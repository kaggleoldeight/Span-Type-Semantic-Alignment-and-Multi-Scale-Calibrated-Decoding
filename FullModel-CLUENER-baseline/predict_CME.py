# -*- coding: utf-8 -*-
"""
CLUENER 预测脚本：兼容当前旧版 GlobalPointer 模型。

输出：
1. outputs_CLUENER/CLUENER_predict_entities.json：entities 列表，便于人工检查；
2. outputs_CLUENER/CLUENER_predict_result.jsonl：CLUENER 常见 label 字典格式。
"""
import os
import json
import random
from typing import Dict, List

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from GlobalPointer import GlobalPointer
from config import (
    PRETRAINED_MODEL_DIR, BEST_MODEL_PATH, DEVICE, DATA_DIR, OUTPUT_DIR,
    TEST_FILE, MAX_LEN, GP_EFFICIENT, ENT_CLS_NUM,
    GP_HEAD_SIZE, RNN_HIDDEN_SIZE, RNN_LAYERS, USE_BIGRU
)
from data_loader import ent2id, id2ent

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = DEVICE


def set_seed(seed=42):
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
    """兼容 JSON 数组和 JSONL。"""
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
    ).to(device)

    print(f"Loading weights from {BEST_MODEL_PATH}...")
    state_dict = torch.load(BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return tokenizer, model


def _deduplicate_entities(entities: List[Dict]) -> List[Dict]:
    seen = set()
    out = []
    for e in entities:
        key = (e['type'], int(e['start_idx']), int(e['end_idx']), e['value'])
        if key not in seen:
            seen.add(key)
            out.append(e)
    return out


def predict_one(text, tokenizer, model, max_len=256, debug=False):
    inputs = tokenizer(
        text,
        max_length=max_len,
        truncation=True,
        return_offsets_mapping=True,
        return_tensors="pt",
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    token_type_ids = inputs.get("token_type_ids")
    if token_type_ids is None:
        token_type_ids = torch.zeros_like(input_ids)
    token_type_ids = token_type_ids.to(device)
    offset_mapping = inputs["offset_mapping"][0].cpu().numpy()

    with torch.no_grad():
        logits = model(input_ids, attention_mask, token_type_ids)
        scores = logits[0].cpu().numpy()

    pred_indices = np.where(scores > 0)
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
        char_end_exclusive = int(end_char_span[1])
        if not (0 <= char_start < char_end_exclusive <= len(text)):
            continue

        extracted_text = text[char_start:char_end_exclusive]
        entities.append({
            "start_idx": char_start,
            # CLUENER 常见 span 为闭区间。
            "end_idx": char_end_exclusive - 1,
            "type": id2ent[int(label_idx)],
            "value": extracted_text,
        })

        if debug:
            print(
                f" [Debug] Found: {extracted_text} ({id2ent[int(label_idx)]}) "
                f"Score: {scores[label_idx, start_token, end_token]:.4f}"
            )

    return _deduplicate_entities(entities)


def entities_to_cluener_label(entities: List[Dict]) -> Dict:
    """
    转为 CLUENER 常见 label 格式：
    {"name": {"张三": [[0, 1]]}}
    """
    label = {}
    for e in entities:
        ent_type = e['type']
        mention = e['value']
        span = [int(e['start_idx']), int(e['end_idx'])]
        label.setdefault(ent_type, {}).setdefault(mention, [])
        if span not in label[ent_type][mention]:
            label[ent_type][mention].append(span)
    return label


if __name__ == '__main__':
    set_seed(42)
    tokenizer, model = load_model()

    test_path = os.path.join(DATA_DIR, TEST_FILE)
    print(f"Reading data from {test_path}")
    data = read_records(test_path)

    print("Starting prediction...")
    print("-" * 30)
    print("Debug Check (First 3 samples):")
    for i in range(min(3, len(data))):
        text = data[i].get('text', '')
        print(f"Text: {text[:30]}...")
        ents = predict_one(text, tokenizer, model, max_len=MAX_LEN, debug=True)
        print(f"Result: {ents}")
        print("-" * 30)

    entity_results = []
    cluener_results = []

    for idx, d in enumerate(tqdm(data, desc="Predicting full dataset")):
        text = d.get('text', '')
        ents = predict_one(text, tokenizer, model, max_len=MAX_LEN)

        entity_item = dict(d)
        entity_item['entities'] = ents
        entity_results.append(entity_item)

        cluener_results.append({
            "id": d.get('id', idx),
            "text": text,
            "label": entities_to_cluener_label(ents),
        })

    entity_out = os.path.join(OUTPUT_DIR, 'CLUENER_predict_entities.json')
    cluener_out = os.path.join(OUTPUT_DIR, 'CLUENER_predict_result.jsonl')

    with open(entity_out, 'w', encoding='utf-8') as f:
        json.dump(entity_results, f, ensure_ascii=False, indent=2)

    with open(cluener_out, 'w', encoding='utf-8') as f:
        for item in cluener_results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Entity-style prediction saved to: {entity_out}")
    print(f"CLUENER-style prediction saved to: {cluener_out}")
