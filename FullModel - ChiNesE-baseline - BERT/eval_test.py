# -*- coding: utf-8 -*-
"""
ChiNesE 有标签 test 集评估脚本。

用途：
- train_CME.py 仍然使用 TRAIN_FILE / DEV_FILE 训练和验证；
- eval_test.py 加载 outputs_ChiNesE_bert_globalpointer/best_model.pth；
- 在 TEST_FILE 上按当前旧版代码一致的阈值 logits > 0 计算 P/R/F1 和逐类别指标。

注意：
- 当前上传的旧版代码没有 gamma/beta/per-class threshold 搜索；
- 因此本脚本不做任何 test 调参，只固定使用训练脚本保存的 best_model.pth，并按 logits > 0 评估。
"""
import json
import os
import random
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from data_loader import EntDataset, id2ent, load_data
from GlobalPointer import GlobalPointer, MetricsCalculator
from config import (
    BATCH_SIZE,
    BEST_MODEL_PATH,
    DATA_DIR,
    DEVICE,
    ENT_CLS_NUM,
    GP_EFFICIENT,
    GP_HEAD_SIZE,
    MAX_LEN,
    NUM_WORKERS,
    OUTPUT_DIR,
    PRETRAINED_MODEL_DIR,
    RNN_HIDDEN_SIZE,
    RNN_LAYERS,
    TEST_FILE,
    TEST_CLASS_METRICS_CSV,
    TEST_METRICS_JSON,
    USE_BIGRU,
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


def update_class_counts(logits, labels, class_tp, class_pred, class_true):
    """按实体类别累计 TP / Pred / True，阈值与训练脚本验证阶段一致：logits > 0。"""
    pred_mask = (logits.detach() > 0)
    true_mask = (labels.detach() > 0)

    num_classes = min(pred_mask.size(1), len(class_tp))
    for cls_idx in range(num_classes):
        pred_c = pred_mask[:, cls_idx]
        true_c = true_mask[:, cls_idx]

        class_tp[cls_idx] += int((pred_c & true_c).sum().item())
        class_pred[cls_idx] += int(pred_c.sum().item())
        class_true[cls_idx] += int(true_c.sum().item())


def build_class_metrics(class_tp, class_pred, class_true):
    """生成逐类别 Precision / Recall / F1。"""
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
    """在控制台打印每个类别的指标。"""
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

    print(f"Loading encoder from {PRETRAINED_MODEL_DIR}...")
    encoder = AutoModel.from_pretrained(PRETRAINED_MODEL_DIR)
    print("=" * 80)
    print(f"Current PRETRAINED_MODEL_DIR: {PRETRAINED_MODEL_DIR}")
    print(f"encoder model_type: {encoder.config.model_type}")
    print(f"hidden_size: {encoder.config.hidden_size}")
    print(f"num_hidden_layers: {encoder.config.num_hidden_layers}")
    print(f"num_attention_heads: {encoder.config.num_attention_heads}")
    print("=" * 80)

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


def main():
    set_seed(42)
    tokenizer, model = load_model()

    test_path = os.path.join(DATA_DIR, TEST_FILE)
    print(f"Loading labeled test data from: {test_path}")
    test_data = load_data(test_path)
    print(f"Test samples: {len(test_data)}")

    test_dataset = EntDataset(test_data, tokenizer=tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=test_dataset.collate,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    metrics = MetricsCalculator()
    total_X, total_Y, total_Z = 0, 0, 0
    class_tp = [0 for _ in range(ENT_CLS_NUM)]
    class_pred = [0 for _ in range(ENT_CLS_NUM)]
    class_true = [0 for _ in range(ENT_CLS_NUM)]

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            raw_text_list, input_ids, attention_mask, segment_ids, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            segment_ids = segment_ids.to(device)
            labels = labels.to(device)

            logits = model(input_ids, attention_mask, segment_ids)
            X, Y, Z = metrics.get_evaluate_fpr(logits, labels)
            total_X += X
            total_Y += Y
            total_Z += Z
            update_class_counts(logits, labels, class_tp, class_pred, class_true)

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
        "test_file": test_path,
        "pretrained_model_dir": PRETRAINED_MODEL_DIR,
        "best_model_path": BEST_MODEL_PATH,
        "threshold_rule": "logits > 0",
        "precision": precision,
        "recall": recall,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "tp": int(total_X),
        "pred": int(total_Y),
        "true": int(total_Z),
        "class_metrics": class_metrics,
    }
    with open(TEST_METRICS_JSON, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Saved test class metrics to: {TEST_CLASS_METRICS_CSV}")
    print(f"Saved test metrics json to: {TEST_METRICS_JSON}")


if __name__ == '__main__':
    import torch.multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()
