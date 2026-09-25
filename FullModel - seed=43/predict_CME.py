# -*- coding: utf-8 -*-
"""
Span-Type Semantic Alignment GlobalPointer 预测脚本：
- 加载训练后的 Span-Type GlobalPointer 模型；
- 预测阶段使用 GlobalPointer span logits + 验证集选择的 Span-Type fusion gamma；
- 使用验证集保存的 best_thresholds.json 中的 per-class 阈值；
- 如果阈值文件不存在，回退到 DEFAULT_PRED_THRESHOLD。
"""
import json
import os
import random
from typing import Dict

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from data_loader import ent2id, id2ent
from GlobalPointer import GlobalPointer
from config import (
    BEST_MODEL_PATH,
    BEST_THRESHOLDS_JSON,
    DATA_DIR,
    DEFAULT_PRED_THRESHOLD,
    USE_THRESHOLD_SEARCH,
    USE_SPAN_TYPE_SEMANTIC_FUSION_INNOVATION,
    USE_MULTISCALE_CALIBRATION_THRESHOLD_INNOVATION,
    DEFAULT_SPAN_TYPE_FUSION_GAMMA,
    DEFAULT_MULTISCALE_FUSION_BETA,
    DEVICE,
    END_INDEX_IS_EXCLUSIVE,
    GP_EFFICIENT,
    GP_HEAD_SIZE,
    MAX_LEN,
    PRETRAINED_MODEL_DIR,
    RNN_HIDDEN_SIZE,
    RNN_LAYERS,
    USE_MULTISCALE_FUSION,
    MULTISCALE_LAYERS,
    MULTISCALE_DROPOUT,
    MULTISCALE_DETACH_ENCODER,
    USE_GATED_MULTISCALE_MAIN_FUSION,
    GATED_MULTISCALE_DETACH_ENCODER,
    GATED_MULTISCALE_INIT_ALPHA,
    GATED_MULTISCALE_MAX_ALPHA,
    GATED_MULTISCALE_DROPOUT,
    SPAN_TYPE_DROPOUT,
    SPAN_TYPE_FUSION_CHUNK_SIZE,
    SPAN_TYPE_MAX_WIDTH,
    SPAN_TYPE_MLP_HIDDEN_SIZE,
    SPAN_TYPE_PROJ_DIM,
    SPAN_TYPE_WIDTH_EMB_SIZE,
    TEST_FILE,
    USE_BIGRU,
    USE_SPAN_TYPE_CONTRASTIVE,
    USE_SPAN_TYPE_FUSION,
)

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

device = DEVICE
ENT_CLS_NUM = len(ent2id)


def set_seed(seed: int = 43) -> None:
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
        use_boundary_aware=False,
        use_span_type_contrastive=USE_SPAN_TYPE_CONTRASTIVE,
        span_type_proj_dim=SPAN_TYPE_PROJ_DIM,
        span_type_hidden_size=SPAN_TYPE_MLP_HIDDEN_SIZE,
        span_width_emb_size=SPAN_TYPE_WIDTH_EMB_SIZE,
        span_type_max_width=SPAN_TYPE_MAX_WIDTH,
        span_type_dropout=SPAN_TYPE_DROPOUT,
        use_multiscale_fusion=USE_MULTISCALE_FUSION,
        multiscale_layers=MULTISCALE_LAYERS,
        multiscale_dropout=MULTISCALE_DROPOUT,
        multiscale_detach_encoder=MULTISCALE_DETACH_ENCODER,
        use_gated_multiscale_main_fusion=USE_GATED_MULTISCALE_MAIN_FUSION,
        gated_multiscale_detach_encoder=GATED_MULTISCALE_DETACH_ENCODER,
        gated_multiscale_init_alpha=GATED_MULTISCALE_INIT_ALPHA,
        gated_multiscale_max_alpha=GATED_MULTISCALE_MAX_ALPHA,
        gated_multiscale_dropout=GATED_MULTISCALE_DROPOUT,
    ).to(device)

    print(f"Loading weights from {BEST_MODEL_PATH}...")
    state_dict = torch.load(BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return tokenizer, model


def load_thresholds():
    threshold_vec = np.full((ENT_CLS_NUM,), float(DEFAULT_PRED_THRESHOLD), dtype=np.float32)

    selected_gamma = float(DEFAULT_SPAN_TYPE_FUSION_GAMMA)
    selected_beta = float(DEFAULT_MULTISCALE_FUSION_BETA)
    thresholds: Dict[str, float] = {}

    if os.path.exists(BEST_THRESHOLDS_JSON):
        with open(BEST_THRESHOLDS_JSON, 'r', encoding='utf-8') as f:
            payload = json.load(f)

        selected_gamma = float(payload.get('selected_span_type_fusion_gamma', selected_gamma))
        selected_beta = float(payload.get('selected_multiscale_fusion_beta', selected_beta))

        if USE_THRESHOLD_SEARCH:
            thresholds = payload.get('thresholds', {})
            for idx in range(ENT_CLS_NUM):
                label_name = id2ent[idx]
                if label_name in thresholds:
                    threshold_vec[idx] = float(thresholds[label_name])
        else:
            print(
                f"USE_THRESHOLD_SEARCH=False. Ignore per-class thresholds in {BEST_THRESHOLDS_JSON} "
                f"and use DEFAULT_PRED_THRESHOLD={DEFAULT_PRED_THRESHOLD}."
            )
    else:
        print(
            f"Warning: threshold file not found: {BEST_THRESHOLDS_JSON}. "
            f"Use DEFAULT_PRED_THRESHOLD={DEFAULT_PRED_THRESHOLD}, gamma={selected_gamma}, beta={selected_beta}."
        )

    if not (USE_SPAN_TYPE_SEMANTIC_FUSION_INNOVATION and USE_SPAN_TYPE_CONTRASTIVE and USE_SPAN_TYPE_FUSION):
        selected_gamma = 0.0
    if not (USE_MULTISCALE_CALIBRATION_THRESHOLD_INNOVATION and USE_MULTISCALE_FUSION):
        selected_beta = 0.0

    print(f"Loaded thresholds from {BEST_THRESHOLDS_JSON}: {thresholds if USE_THRESHOLD_SEARCH else 'DEFAULT_ONLY'}")
    print(f"Loaded Span-Type fusion gamma: {selected_gamma}")
    print(f"Loaded Multi-Scale fusion beta: {selected_beta}")
    return threshold_vec, selected_gamma, selected_beta


def get_span_scores(model, input_ids, attention_mask, token_type_ids, gamma: float, beta: float):
    outputs = model(
        input_ids,
        attention_mask,
        token_type_ids,
        return_span_type=(USE_SPAN_TYPE_CONTRASTIVE and USE_SPAN_TYPE_FUSION and float(gamma) != 0.0),
        return_multiscale=(USE_MULTISCALE_FUSION and float(beta) != 0.0),
    )
    if isinstance(outputs, dict):
        span_logits = outputs['span_logits']
        if USE_SPAN_TYPE_CONTRASTIVE and USE_SPAN_TYPE_FUSION and float(gamma) != 0.0:
            span_type_logits = model.compute_span_type_similarity_logits(
                outputs['sequence_output'],
                attention_mask,
                chunk_size=SPAN_TYPE_FUSION_CHUNK_SIZE,
            )
            span_logits = span_logits + float(gamma) * span_type_logits
        if USE_MULTISCALE_FUSION and float(beta) != 0.0 and 'multiscale_logits' in outputs:
            span_logits = span_logits + float(beta) * outputs['multiscale_logits']
        return span_logits
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    return outputs


def predict_one(text, tokenizer, model, thresholds: np.ndarray, gamma: float, beta: float, max_len: int = 256, debug: bool = False):
    inputs = tokenizer(
        text,
        max_length=max_len,
        truncation=True,
        return_offsets_mapping=True,
        return_tensors='pt',
    )

    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    token_type_ids = inputs.get('token_type_ids')
    if token_type_ids is None:
        token_type_ids = torch.zeros_like(input_ids)
    token_type_ids = token_type_ids.to(device)
    offset_mapping = inputs['offset_mapping'][0].cpu().numpy()

    with torch.no_grad():
        span_logits = get_span_scores(model, input_ids, attention_mask, token_type_ids, gamma, beta)
        scores = span_logits[0].cpu().numpy()

    threshold_arr = thresholds.reshape(-1, 1, 1)
    pred_indices = np.where(scores > threshold_arr)

    entities = []
    if debug and len(pred_indices[0]) == 0:
        print(f" [Debug] No entities found. Max score: {np.max(scores)}")
        return []

    for label_idx, start_token, end_token in zip(*pred_indices):
        start_char_span = offset_mapping[start_token]
        end_char_span = offset_mapping[end_token]

        if start_char_span[0] == 0 and start_char_span[1] == 0:
            continue
        if end_char_span[0] == 0 and end_char_span[1] == 0:
            continue

        char_start = int(start_char_span[0])
        char_end = int(end_char_span[1])
        extracted_text = text[char_start:char_end]

        entities.append({
            'start_idx': char_start,
            # 与训练数据索引约定保持一致：右开区间输出 char_end，否则输出闭区间末字符。
            'end_idx': char_end if END_INDEX_IS_EXCLUSIVE else char_end - 1,
            'type': id2ent[int(label_idx)],
            'value': extracted_text,
        })

        if debug:
            print(
                f" [Debug] Found: {extracted_text} ({id2ent[int(label_idx)]}) "
                f"Score: {scores[label_idx, start_token, end_token]:.4f} "
                f"Threshold: {thresholds[int(label_idx)]:.4f}"
            )

    return entities


if __name__ == '__main__':
    set_seed(43)

    tokenizer, model = load_model()
    thresholds, gamma, beta = load_thresholds()

    test_path = os.path.join(DATA_DIR, TEST_FILE)
    print(f"Reading data from {test_path}")
    data = json.load(open(test_path, encoding='utf-8'))

    results = []
    print("Starting prediction...")

    print("-" * 30)
    print("Debug Check (First 3 samples):")
    for i in range(min(3, len(data))):
        print(f"Text: {data[i]['text'][:30]}...")
        ents = predict_one(data[i]['text'], tokenizer, model, thresholds, gamma, beta, max_len=MAX_LEN, debug=True)
        print(f"Result: {ents}")
        print("-" * 30)

    for d in tqdm(data, desc='Predicting full dataset'):
        ents = predict_one(d['text'], tokenizer, model, thresholds, gamma, beta, max_len=MAX_LEN, debug=False)
        item = dict(d)
        item['entities'] = ents
        results.append(item)

    test_stem = os.path.splitext(os.path.basename(TEST_FILE))[0]
    output_path = os.path.join(DATA_DIR, f'{test_stem}_pred.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Prediction finished. Saved to {output_path}")
