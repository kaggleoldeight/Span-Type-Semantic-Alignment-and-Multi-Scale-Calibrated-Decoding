# -*- coding: utf-8 -*-
import os
import json
import platform
import random
import time
from typing import Dict
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from GlobalPointer import GlobalPointer
from config import (
    PRETRAINED_MODEL_DIR, BEST_MODEL_PATH, DEVICE, DATA_DIR,
    TEST_FILE, MAX_LEN, GP_EFFICIENT, ENT_CLS_NUM,
    GP_HEAD_SIZE, RNN_HIDDEN_SIZE, RNN_LAYERS, USE_BIGRU,
    ENABLE_COMPUTE_COST_PROFILING, INFERENCE_COMPUTE_COST_JSON,
    INFERENCE_WARMUP_SAMPLES
)
from data_loader import ent2id

# 确保输出目录存在
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

device = DEVICE
id2ent = {v: k for k, v in ent2id.items()}



def count_model_parameters(model: torch.nn.Module) -> Dict[str, float]:
    """与训练脚本采用相同参数量口径。"""
    total_parameters = sum(p.numel() for p in model.parameters())
    trainable_parameters = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    parameter_bytes = sum(
        p.numel() * p.element_size() for p in model.parameters()
    )
    return {
        'total_parameters': int(total_parameters),
        'trainable_parameters': int(trainable_parameters),
        'frozen_parameters': int(total_parameters - trainable_parameters),
        'parameter_storage_mib': float(parameter_bytes / (1024 ** 2)),
    }


def get_runtime_environment() -> Dict:
    cuda_available = bool(torch.cuda.is_available())
    cuda_device_index = None
    gpu_name = None
    if cuda_available:
        cuda_device_index = int(torch.cuda.current_device())
        gpu_name = torch.cuda.get_device_name(cuda_device_index)
    return {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'pytorch_version': torch.__version__,
        'cuda_available': cuda_available,
        'cuda_runtime_version': torch.version.cuda,
        'cudnn_version': (
            torch.backends.cudnn.version() if cuda_available else None
        ),
        'device': str(device),
        'cuda_device_index': cuda_device_index,
        'gpu_name': gpu_name,
    }


def synchronize_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)


def reset_cuda_peak_memory() -> None:
    if torch.cuda.is_available():
        synchronize_cuda()
        torch.cuda.reset_peak_memory_stats(device)


def get_cuda_peak_memory() -> Dict[str, float]:
    if not torch.cuda.is_available():
        return {
            'peak_allocated_bytes': None,
            'peak_allocated_gib': None,
            'peak_reserved_bytes': None,
            'peak_reserved_gib': None,
        }
    synchronize_cuda()
    allocated = int(torch.cuda.max_memory_allocated(device))
    reserved = int(torch.cuda.max_memory_reserved(device))
    return {
        'peak_allocated_bytes': allocated,
        'peak_allocated_gib': float(allocated / (1024 ** 3)),
        'peak_reserved_bytes': reserved,
        'peak_reserved_gib': float(reserved / (1024 ** 3)),
    }


def save_json(payload: Dict, output_path: str) -> None:
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def set_seed(seed=44):
    """
    设置随机种子，确保预测结果可复现
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to {seed} for reproducibility")


def load_model():
    print(f"Loading tokenizer from {PRETRAINED_MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_DIR, use_fast=True)

    print("Loading model structure...")
    encoder = AutoModel.from_pretrained(PRETRAINED_MODEL_DIR)
    model = GlobalPointer(
        encoder,
        ENT_CLS_NUM,
        GP_HEAD_SIZE,
        rnn_hidden_size=RNN_HIDDEN_SIZE,
        rnn_layers=RNN_LAYERS,
        efficient=GP_EFFICIENT,
        use_bigru=USE_BIGRU,
    ).to(device)

    print(f"Loading weights from {BEST_MODEL_PATH}...")
    # 增加 map_location 防止 GPU/CPU 不匹配报错
    state_dict = torch.load(BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return tokenizer, model


def predict_one(text, tokenizer, model, max_len=256, debug=False):
    # 1. Tokenizer 处理 (关键：必须开启 truncation 和 return_offsets_mapping)
    inputs = tokenizer(
        text,
        max_length=max_len,
        truncation=True,
        return_offsets_mapping=True,
        return_tensors="pt"
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    token_type_ids = inputs["token_type_ids"].to(device)
    offset_mapping = inputs["offset_mapping"][0].cpu().numpy()  # (seq_len, 2)

    # 2. 模型推理
    with torch.no_grad():
        # model 输出形状: (batch, ent_type, seq, seq)
        logits = model(input_ids, attention_mask, token_type_ids)
        scores = logits[0].cpu().numpy()  # 取 batch 第一个 -> (ent_type, seq, seq)

    # 3. 阈值过滤 (GlobalPointer 默认阈值为 0)
    # np.where 返回: (type_indices, start_indices, end_indices)
    pred_indices = np.where(scores > 0)

    entities = []

    # 调试信息：如果开启 debug 且分数都不大于0
    if debug and len(pred_indices[0]) == 0:
        print(f" [Debug] No entities found. Max score: {np.max(scores)}")
        return []

    for label_idx, start_token, end_token in zip(*pred_indices):
        # 过滤掉 [CLS], [SEP], [PAD] 等特殊位置
        # offset_mapping 中，特殊 token 的 span 通常是 (0,0)
        start_char_span = offset_mapping[start_token]
        end_char_span = offset_mapping[end_token]

        # 如果是特殊token (start=0, end=0)，跳过
        if start_char_span[0] == 0 and start_char_span[1] == 0:
            continue
        if end_char_span[0] == 0 and end_char_span[1] == 0:
            continue

        # 映射回原文字符索引
        # offset_mapping 格式：(start_char, end_char)，其中 end_char 是开区间（不包含）
        # 例如：token "苯" 的 offset_mapping 可能是 (22, 23)，表示字符索引 [22, 23)
        char_start = start_char_span[0]  # 实体起始字符索引（闭区间）
        char_end = end_char_span[1]      # 实体结束字符的后一位（开区间，不包含）
        
        # Python 切片 text[start:end] 是左闭右开，正好匹配 offset_mapping 的格式
        # 例如：text[22:27] 会提取索引 22, 23, 24, 25, 26 的字符（共5个字符）
        extracted_text = text[char_start:char_end]

        entities.append({
            "start_idx": int(char_start),
            # 注意：char_end 是开区间位置，需要减1转换为闭区间格式
            # 这样输出的 end_idx 与训练时的格式一致（闭区间，包含最后一个字符）
            "end_idx": int(char_end) - 1,
            "type": id2ent[label_idx],
            "value": extracted_text
        })

        if debug:
            print(
                f" [Debug] Found: {extracted_text} ({id2ent[label_idx]}) Score: {scores[label_idx, start_token, end_token]:.4f}")

    return entities


if __name__ == '__main__':
    # 设置随机种子，确保预测结果可复现
    set_seed(44)
    
    tokenizer, model = load_model()

    parameter_statistics = count_model_parameters(model)
    print("Model parameter statistics:")
    print(
        f"  Total parameters: "
        f"{parameter_statistics['total_parameters']:,}"
    )
    print(
        f"  Trainable parameters: "
        f"{parameter_statistics['trainable_parameters']:,}"
    )

    # 读取测试文件
    test_path = os.path.join(DATA_DIR, TEST_FILE)
    print(f"Reading data from {test_path}")
    data = json.load(open(test_path, encoding='utf-8'))

    results = []
    print("Starting prediction...")

    # 仅用于调试：先测前5条，看看有没有输出
    print("-" * 30)
    print("Debug Check (First 3 samples):")
    for i in range(min(3, len(data))):
        print(f"Text: {data[i]['text'][:30]}...")
        ents = predict_one(data[i]['text'], tokenizer, model, max_len=MAX_LEN, debug=True)
        print(f"Result: {ents}")
        print("-" * 30)

    # 正式计时前预热；预热结果不写入预测文件。
    warmup_count = 0
    if ENABLE_COMPUTE_COST_PROFILING and len(data) > 0:
        warmup_count = min(
            int(INFERENCE_WARMUP_SAMPLES),
            len(data),
        )
        for i in range(warmup_count):
            _ = predict_one(
                data[i]['text'],
                tokenizer,
                model,
                max_len=MAX_LEN,
            )

    if ENABLE_COMPUTE_COST_PROFILING:
        reset_cuda_peak_memory()
    synchronize_cuda()
    inference_start = time.perf_counter()

    # 全量预测
    for d in tqdm(data, desc="Predicting full dataset"):
        ents = predict_one(d['text'], tokenizer, model, max_len=MAX_LEN)
        results.append({
            "text": d['text'],
            "entities": ents
        })

    synchronize_cuda()
    inference_elapsed_seconds = (
        time.perf_counter() - inference_start
    )
    inference_peak_memory = (
        get_cuda_peak_memory()
        if ENABLE_COMPUTE_COST_PROFILING
        else {
            'peak_allocated_bytes': None,
            'peak_allocated_gib': None,
            'peak_reserved_bytes': None,
            'peak_reserved_gib': None,
        }
    )

    out_path = os.path.join(DATA_DIR, 'CMeEE-V2_predict_result.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    if ENABLE_COMPUTE_COST_PROFILING:
        timed_samples = int(len(data))
        throughput = (
            float(timed_samples / inference_elapsed_seconds)
            if inference_elapsed_seconds > 0
            else 0.0
        )
        latency_ms = (
            float(
                inference_elapsed_seconds * 1000.0 / timed_samples
            )
            if timed_samples > 0
            else None
        )
        inference_summary = {
            'measurement_definition': {
                'scope': (
                    'End-to-end single-sample inference. Tokenization, '
                    'CPU-to-device transfer, model forward, device-to-CPU '
                    'transfer, thresholding and entity decoding are included. '
                    'Model/tokenizer loading, warm-up, debug output and result '
                    'file writing are excluded.'
                ),
                'batch_size': 1,
                'warmup_samples': int(warmup_count),
            },
            'environment': get_runtime_environment(),
            'parameter_statistics': parameter_statistics,
            'test_file': test_path,
            'timed_samples': timed_samples,
            'elapsed_seconds': float(inference_elapsed_seconds),
            'throughput_samples_per_second': throughput,
            'latency_ms_per_sample': latency_ms,
            'inference_peak_allocated_bytes': (
                inference_peak_memory['peak_allocated_bytes']
            ),
            'inference_peak_allocated_gib': (
                inference_peak_memory['peak_allocated_gib']
            ),
            'inference_peak_reserved_bytes': (
                inference_peak_memory['peak_reserved_bytes']
            ),
            'inference_peak_reserved_gib': (
                inference_peak_memory['peak_reserved_gib']
            ),
            'prediction_threshold': 0.0,
            'max_length': int(MAX_LEN),
        }
        save_json(
            inference_summary,
            INFERENCE_COMPUTE_COST_JSON,
        )
        print(
            "Inference compute-cost summary saved to: "
            f"{INFERENCE_COMPUTE_COST_JSON}"
        )

    print(f"Done. Saved to {out_path}")
