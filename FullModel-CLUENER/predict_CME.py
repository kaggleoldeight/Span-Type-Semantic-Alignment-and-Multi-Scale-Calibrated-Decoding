# -*- coding: utf-8 -*-
"""
CLUENER 预测脚本（兼容当前 Span-Type + Multi-Scale GlobalPointer 版本）

功能：
1. 加载当前训练得到的 best_model.pth；
2. 支持读取 JSON 数组与 JSONL 两种测试文件；
3. 使用验证集保存的 best_thresholds.json：
   - USE_THRESHOLD_SEARCH=True 时使用 per-class 阈值；
   - 否则回退 DEFAULT_PRED_THRESHOLD；
4. 若启用 Span-Type / Multi-Scale，则按保存的 gamma / beta 融合；
5. 输出两种文件：
   - <test_stem>_pred_entities.json：包含 entities 列表，便于人工检查；
   - <test_stem>_pred_cluener.jsonl：CLUENER 常见提交格式，字段为 id/text/label。
"""
import json
import os
import random
from collections import defaultdict
from typing import Dict, List

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


def set_seed(seed: int = 42) -> None:
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


def read_records(path: str) -> List[Dict]:
    """兼容 JSON 数组和 JSONL。CLUENER 官方常见为 JSONL。"""
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    if not text:
        return []
    if text[0] == '[':
        return json.loads(text)
    records = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


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

    print(f"Loaded thresholds: {thresholds if USE_THRESHOLD_SEARCH else 'DEFAULT_ONLY'}")
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


def _deduplicate_entities(entities: List[Dict]) -> List[Dict]:
    """按 type/start/end/value 去重；保留当前排序。"""
    seen = set()
    deduped = []
    for e in entities:
        key = (e['type'], int(e['start_idx']), int(e['end_idx']), e['value'])
        if key not in seen:
            seen.add(key)
            deduped.append(e)
    return deduped


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

        # 过滤 [CLS] / [SEP] / [PAD]
        if start_char_span[0] == 0 and start_char_span[1] == 0:
            continue
        if end_char_span[0] == 0 and end_char_span[1] == 0:
            continue

        char_start = int(start_char_span[0])
        char_end_exclusive = int(end_char_span[1])
        if not (0 <= char_start < char_end_exclusive <= len(text)):
            continue

        extracted_text = text[char_start:char_end_exclusive]
        entities.append({
            'start_idx': char_start,
            # CLUENER 标注常见为闭区间；这里保持闭区间 end。
            'end_idx': char_end_exclusive - 1,
            'type': id2ent[int(label_idx)],
            'value': extracted_text,
        })

        if debug:
            print(
                f" [Debug] Found: {extracted_text} ({id2ent[int(label_idx)]}) "
                f"Score: {scores[label_idx, start_token, end_token]:.4f} "
                f"Threshold: {thresholds[int(label_idx)]:.4f}"
            )

    return _deduplicate_entities(entities)


def entities_to_cluener_label(entities: List[Dict]) -> Dict:
    """
    转为 CLUENER 常见 label 格式：
    {
      "name": {"张三": [[0, 1]]},
      "company": {"腾讯": [[5, 6]]}
    }
    """
    label = {}
    for e in entities:
        ent_type = e['type']
        mention = e['value']
        start = int(e['start_idx'])
        end = int(e['end_idx'])
        if ent_type not in label:
            label[ent_type] = {}
        if mention not in label[ent_type]:
            label[ent_type][mention] = []
        span = [start, end]
        if span not in label[ent_type][mention]:
            label[ent_type][mention].append(span)
    return label


if __name__ == '__main__':
    set_seed(42)

    tokenizer, model = load_model()
    thresholds, gamma, beta = load_thresholds()

    test_path = os.path.join(DATA_DIR, TEST_FILE)
    print(f"Reading data from {test_path}")
    data = read_records(test_path)

    print("Starting prediction...")
    print("-" * 30)
    print("Debug Check (First 3 samples):")
    for i in range(min(3, len(data))):
        print(f"Text: {data[i].get('text', '')[:30]}...")
        ents = predict_one(data[i].get('text', ''), tokenizer, model, thresholds, gamma, beta, max_len=MAX_LEN, debug=True)
        print(f"Result: {ents}")
        print("-" * 30)

    entity_results = []
    cluener_results = []

    for idx, d in enumerate(tqdm(data, desc='Predicting full dataset')):
        text = d.get('text', '')
        ents = predict_one(text, tokenizer, model, thresholds, gamma, beta, max_len=MAX_LEN, debug=False)

        entity_item = dict(d)
        entity_item['entities'] = ents
        entity_results.append(entity_item)

        # CLUENER 常见提交/预测格式：保留 id/text，label 为预测结果。
        cluener_item = {}
        if 'id' in d:
            cluener_item['id'] = d['id']
        else:
            cluener_item['id'] = idx
        cluener_item['text'] = text
        cluener_item['label'] = entities_to_cluener_label(ents)
        cluener_results.append(cluener_item)

    test_stem = os.path.splitext(os.path.basename(TEST_FILE))[0]
    entity_output_path = os.path.join(DATA_DIR, f'{test_stem}_pred_entities.json')
    cluener_output_path = os.path.join(DATA_DIR, f'{test_stem}_pred_cluener.jsonl')

    with open(entity_output_path, 'w', encoding='utf-8') as f:
        json.dump(entity_results, f, ensure_ascii=False, indent=2)

    with open(cluener_output_path, 'w', encoding='utf-8') as f:
        for item in cluener_results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Entity-style prediction saved to: {entity_output_path}")
    print(f"CLUENER-style prediction saved to: {cluener_output_path}")
