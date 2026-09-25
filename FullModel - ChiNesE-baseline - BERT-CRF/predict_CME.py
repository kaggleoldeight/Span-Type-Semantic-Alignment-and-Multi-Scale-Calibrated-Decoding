# -*- coding: utf-8 -*-
"""
ChiNesE BERT-CRF 预测脚本。

输出：
1. outputs_ChiNesE_BERT_CRF/ChiNesE_BERT_CRF_predict_entities.json：pred_entities 列表，便于人工检查；
2. outputs_ChiNesE_BERT_CRF/ChiNesE_BERT_CRF_predict_result.jsonl：ChiNesE 常见 label 字典格式。
"""
import json
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from bert_crf import BertCRF
from data_loader import crf_id2label, ent2id
from config import (
    BEST_MODEL_PATH,
    DATA_DIR,
    DEVICE,
    DROPOUT_PROB,
    MAX_LEN,
    NUM_CRF_LABELS,
    OUTPUT_DIR,
    PRETRAINED_MODEL_DIR,
    SEED,
    TEST_FILE,
)

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
    if not tokenizer.is_fast:
        raise RuntimeError('BERT-CRF 版本需要 fast tokenizer 以获得 offset_mapping。')

    print("Loading BERT-CRF model structure...")
    model = BertCRF(PRETRAINED_MODEL_DIR, num_labels=NUM_CRF_LABELS, dropout_prob=DROPOUT_PROB).to(device)
    print(f"Loading weights from {BEST_MODEL_PATH}...")
    state_dict = torch.load(BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return tokenizer, model


def decode_entities_from_labels(text: str, label_ids: List[int], offset_mapping) -> List[Dict]:
    entities: List[Tuple[int, int, str]] = []
    current_type = None
    current_start = None
    current_end = None

    for idx, label_id in enumerate(label_ids):
        if idx >= len(offset_mapping):
            break
        token_start, token_end = offset_mapping[idx]
        if token_start == 0 and token_end == 0:
            if current_type is not None:
                entities.append((int(current_start), int(current_end), current_type))
                current_type = None
                current_start = None
                current_end = None
            continue

        label = crf_id2label.get(int(label_id), 'O')
        if label == 'O':
            if current_type is not None:
                entities.append((int(current_start), int(current_end), current_type))
                current_type = None
                current_start = None
                current_end = None
            continue

        prefix, ent_type = label.split('-', 1)
        if prefix == 'B':
            if current_type is not None:
                entities.append((int(current_start), int(current_end), current_type))
            current_type = ent_type
            current_start = int(token_start)
            current_end = int(token_end) - 1
        elif prefix == 'I':
            if current_type == ent_type:
                current_end = int(token_end) - 1
            else:
                if current_type is not None:
                    entities.append((int(current_start), int(current_end), current_type))
                current_type = ent_type
                current_start = int(token_start)
                current_end = int(token_end) - 1

    if current_type is not None:
        entities.append((int(current_start), int(current_end), current_type))

    seen = set()
    out = []
    for start, end, ent_type in entities:
        if 0 <= start <= end < len(text) and ent_type in ent2id:
            value = text[start:end + 1]
            key = (start, end, ent_type, value)
            if key not in seen:
                seen.add(key)
                out.append({
                    "start_idx": int(start),
                    "end_idx": int(end),
                    "type": ent_type,
                    "value": value,
                })
    return out


def predict_one(text: str, tokenizer, model, max_len=256, debug=False) -> List[Dict]:
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
    offset_mapping = inputs["offset_mapping"][0].cpu().numpy().tolist()

    with torch.no_grad():
        _, _, decoded = model(input_ids, attention_mask, token_type_ids, labels=None)

    ents = decode_entities_from_labels(text, decoded[0], offset_mapping)
    if debug:
        print(f"Pred entities: {ents}")
    return ents


def entities_to_chinese_label(entities: List[Dict]) -> Dict:
    label = {}
    for e in entities:
        ent_type = e['type']
        mention = e['value']
        span = [int(e['start_idx']), int(e['end_idx'])]
        label.setdefault(ent_type, {}).setdefault(mention, [])
        if span not in label[ent_type][mention]:
            label[ent_type][mention].append(span)
    return label


def main():
    set_seed(SEED)
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
        print("-" * 30)

    entity_results = []
    chinese_results = []
    for idx, d in enumerate(tqdm(data, desc="Predicting full dataset")):
        text = d.get('text', '')
        ents = predict_one(text, tokenizer, model, max_len=MAX_LEN)

        entity_item = dict(d)
        entity_item['pred_entities'] = ents
        entity_results.append(entity_item)

        chinese_results.append({
            "id": d.get('id', idx),
            "text": text,
            "label": entities_to_chinese_label(ents),
        })

    entity_out = os.path.join(OUTPUT_DIR, 'ChiNesE_BERT_CRF_predict_entities.json')
    chinese_out = os.path.join(OUTPUT_DIR, 'ChiNesE_BERT_CRF_predict_result.jsonl')

    with open(entity_out, 'w', encoding='utf-8') as f:
        json.dump(entity_results, f, ensure_ascii=False, indent=2)

    with open(chinese_out, 'w', encoding='utf-8') as f:
        for item in chinese_results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Entity-style prediction saved to: {entity_out}")
    print(f"ChiNesE-style prediction saved to: {chinese_out}")


if __name__ == '__main__':
    main()
