# -*- coding: utf-8 -*-
"""
ChiNesE BERT-CRF 有标签 test 集评估脚本。

流程：
- train_CME.py 用 train/dev 训练并保存 outputs_ChiNesE_BERT_CRF/best_model.pth；
- 本脚本固定加载 best_model.pth；
- 在 test 上只评估一次，不做任何 test 调参；
- 指标使用完整 ChiNesE nested gold entities。
"""
import json
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from bert_crf import BertCRF
from data_loader import EntDataset, crf_id2label, ent2id, id2ent, load_data
from config import (
    BATCH_SIZE,
    BEST_MODEL_PATH,
    DATA_DIR,
    DEVICE,
    DROPOUT_PROB,
    FLAT_NER_STRATEGY,
    NUM_CRF_LABELS,
    NUM_WORKERS,
    OUTPUT_DIR,
    PRETRAINED_MODEL_DIR,
    SEED,
    TEST_CLASS_METRICS_CSV,
    TEST_FILE,
    TEST_METRICS_JSON,
)


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


def decode_entities_from_labels(text: str, label_ids: List[int], offset_mapping) -> List[Tuple[int, int, str]]:
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
    valid = []
    for start, end, ent_type in entities:
        if 0 <= start <= end < len(text) and ent_type in ent2id:
            key = (start, end, ent_type)
            if key not in seen:
                seen.add(key)
                valid.append(key)
    return valid


def update_counts_from_entities(pred_entities, gold_entities, class_tp, class_pred, class_true):
    pred_set = set(pred_entities)
    gold_set = set(gold_entities)
    inter_set = pred_set & gold_set

    for start, end, ent_type in inter_set:
        class_tp[ent2id[ent_type]] += 1
    for start, end, ent_type in pred_set:
        class_pred[ent2id[ent_type]] += 1
    for start, end, ent_type in gold_set:
        class_true[ent2id[ent_type]] += 1

    return len(inter_set), len(pred_set), len(gold_set)


def build_class_metrics(class_tp, class_pred, class_true):
    class_metrics = []
    f1_values = []
    for cls_idx in range(len(class_tp)):
        tp = int(class_tp[cls_idx])
        pred = int(class_pred[cls_idx])
        true = int(class_true[cls_idx])
        precision = tp / pred if pred > 0 else 0.0
        recall = tp / true if true > 0 else 0.0
        f1 = 2 * tp / (pred + true) if (pred + true) > 0 else 0.0
        f1_values.append(f1)
        class_metrics.append({
            'label': id2ent.get(cls_idx, str(cls_idx)),
            'tp': tp,
            'pred': pred,
            'true': true,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        })
    macro_f1 = float(np.mean(f1_values)) if f1_values else 0.0
    return class_metrics, macro_f1


def print_class_metrics(class_metrics):
    print("\nTest class metrics:")
    print(f"{'label':<14}{'TP':>8}{'Pred':>8}{'True':>8}{'Precision':>12}{'Recall':>12}{'F1':>12}")
    for item in class_metrics:
        print(
            f"{item['label']:<14}"
            f"{int(item['tp']):>8}"
            f"{int(item['pred']):>8}"
            f"{int(item['true']):>8}"
            f"{item['precision']:>12.6f}"
            f"{item['recall']:>12.6f}"
            f"{item['f1']:>12.6f}"
        )


def save_test_class_metrics(csv_path: str, class_metrics: List[Dict]) -> None:
    import csv
    parent = os.path.dirname(csv_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fields = ['label', 'tp', 'pred', 'true', 'precision', 'recall', 'f1']
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for item in class_metrics:
            writer.writerow({
                'label': item['label'],
                'tp': int(item['tp']),
                'pred': int(item['pred']),
                'true': int(item['true']),
                'precision': f"{item['precision']:.6f}",
                'recall': f"{item['recall']:.6f}",
                'f1': f"{item['f1']:.6f}",
            })


def load_model():
    print(f"Loading tokenizer from {PRETRAINED_MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_DIR, use_fast=True)
    if not tokenizer.is_fast:
        raise RuntimeError('BERT-CRF 版本需要 fast tokenizer 以获得 offset_mapping。')

    print(f"Loading BERT-CRF structure from {PRETRAINED_MODEL_DIR}...")
    model = BertCRF(PRETRAINED_MODEL_DIR, num_labels=NUM_CRF_LABELS, dropout_prob=DROPOUT_PROB).to(device)
    print("=" * 80)
    print(f"Current PRETRAINED_MODEL_DIR: {PRETRAINED_MODEL_DIR}")
    print(f"encoder model_type: {model.encoder.config.model_type}")
    print(f"hidden_size: {model.encoder.config.hidden_size}")
    print(f"num_hidden_layers: {model.encoder.config.num_hidden_layers}")
    print(f"num_attention_heads: {model.encoder.config.num_attention_heads}")
    print("Model: BERT-CRF flat sequence labeling baseline")
    print("=" * 80)

    print(f"Loading weights from {BEST_MODEL_PATH}...")
    state_dict = torch.load(BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return tokenizer, model


def main():
    set_seed(SEED)
    tokenizer, model = load_model()

    test_path = os.path.join(DATA_DIR, TEST_FILE)
    print(f"Loading labeled test data from: {test_path}")
    test_data = load_data(test_path)
    print(f"Test samples: {len(test_data)}")
    print(f"Flat NER strategy used during training: {FLAT_NER_STRATEGY}")

    test_dataset = EntDataset(test_data, tokenizer=tokenizer, flat_strategy=FLAT_NER_STRATEGY)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=test_dataset.collate,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    total_X, total_Y, total_Z = 0, 0, 0
    class_tp = [0 for _ in range(len(ent2id))]
    class_pred = [0 for _ in range(len(ent2id))]
    class_true = [0 for _ in range(len(ent2id))]

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            raw_text_list, input_ids, attention_mask, segment_ids, labels, offsets, gold_entities = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            segment_ids = segment_ids.to(device)

            _, _, decoded = model(input_ids, attention_mask, segment_ids, labels=None)

            for i, label_ids in enumerate(decoded):
                pred_entities = decode_entities_from_labels(raw_text_list[i], label_ids, offsets[i])
                X, Y, Z = update_counts_from_entities(pred_entities, gold_entities[i], class_tp, class_pred, class_true)
                total_X += X
                total_Y += Y
                total_Z += Z

    precision = total_X / total_Y if total_Y > 0 else 0.0
    recall = total_X / total_Z if total_Z > 0 else 0.0
    micro_f1 = 2 * total_X / (total_Y + total_Z) if (total_Y + total_Z) > 0 else 0.0

    class_metrics, macro_f1 = build_class_metrics(class_tp, class_pred, class_true)
    print_class_metrics(class_metrics)
    print(
        f"\nTest overall: precision={precision:.6f}, recall={recall:.6f}, "
        f"micro_f1={micro_f1:.6f}, macro_f1={macro_f1:.6f}, "
        f"TP={total_X}, Pred={total_Y}, True={total_Z}"
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_test_class_metrics(TEST_CLASS_METRICS_CSV, class_metrics)

    payload = {
        "model": "BERT-CRF",
        "test_file": test_path,
        "pretrained_model_dir": PRETRAINED_MODEL_DIR,
        "best_model_path": BEST_MODEL_PATH,
        "flat_ner_strategy": FLAT_NER_STRATEGY,
        "evaluation_gold": "full ChiNesE nested gold entities",
        "precision": precision,
        "recall": recall,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "tp": int(total_X),
        "pred": int(total_Y),
        "true": int(total_Z),
    }
    with open(TEST_METRICS_JSON, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Saved test class metrics to: {TEST_CLASS_METRICS_CSV}")
    print(f"Saved test metrics json to: {TEST_METRICS_JSON}")


if __name__ == '__main__':
    main()
