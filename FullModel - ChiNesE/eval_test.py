# -*- coding: utf-8 -*-
"""
有标签测试集评估脚本，适用于 ChiNesE/new_test.json。

使用流程：
1. python train_CME.py
   - 训练时只在 DEV_FILE/new_eval.json 上选择 best_model、gamma、beta 和 per-class thresholds。
2. python eval_test.py
   - 固定 best_thresholds.json 中的验证集最优参数，在 TEST_FILE/new_test.json 上只评估一次。
   - 不在测试集上重新搜索阈值，避免测试集信息泄漏。
"""
import csv
import json
import os
import random
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from config import (
    BATCH_SIZE,
    BEST_MODEL_PATH,
    BEST_THRESHOLDS_JSON,
    DATA_DIR,
    DEFAULT_MULTISCALE_FUSION_BETA,
    DEFAULT_PRED_THRESHOLD,
    DEFAULT_SPAN_TYPE_FUSION_GAMMA,
    DEVICE,
    GP_EFFICIENT,
    GP_HEAD_SIZE,
    MAX_LEN,
    MULTISCALE_DETACH_ENCODER,
    MULTISCALE_DROPOUT,
    MULTISCALE_LAYERS,
    OUTPUT_DIR,
    PRETRAINED_MODEL_DIR,
    RNN_HIDDEN_SIZE,
    RNN_LAYERS,
    SPAN_TYPE_DROPOUT,
    SPAN_TYPE_FUSION_CHUNK_SIZE,
    SPAN_TYPE_MAX_WIDTH,
    SPAN_TYPE_MLP_HIDDEN_SIZE,
    SPAN_TYPE_PROJ_DIM,
    SPAN_TYPE_WIDTH_EMB_SIZE,
    TEST_FILE,
    USE_BIGRU,
    USE_GATED_MULTISCALE_MAIN_FUSION,
    GATED_MULTISCALE_DETACH_ENCODER,
    GATED_MULTISCALE_INIT_ALPHA,
    GATED_MULTISCALE_MAX_ALPHA,
    GATED_MULTISCALE_DROPOUT,
    USE_MULTISCALE_FUSION,
    USE_SPAN_TYPE_CONTRASTIVE,
    USE_SPAN_TYPE_FUSION,
    USE_SPAN_TYPE_SEMANTIC_FUSION_INNOVATION,
    USE_MULTISCALE_CALIBRATION_THRESHOLD_INNOVATION,
    USE_THRESHOLD_SEARCH,
)
try:
    from config import TEST_CLASS_METRICS_CSV, TEST_METRICS_JSON
except Exception:
    TEST_CLASS_METRICS_CSV = os.path.join(OUTPUT_DIR, 'test_class_metrics.csv')
    TEST_METRICS_JSON = os.path.join(OUTPUT_DIR, 'test_metrics.json')

from data_loader import EntDataset, ent2id, id2ent, load_data
from GlobalPointer import GlobalPointer


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


def resolve_data_path(path: str) -> str:
    if os.path.exists(path):
        return path
    if path.endswith('.json') and os.path.exists(path[:-5]):
        return path[:-5]
    if (not path.endswith('.json')) and os.path.exists(path + '.json'):
        return path + '.json'
    return path


def ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def load_model():
    print(f"Loading tokenizer from {PRETRAINED_MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_DIR, use_fast=True)
    print(f"Loading encoder from {PRETRAINED_MODEL_DIR}...")
    encoder = AutoModel.from_pretrained(PRETRAINED_MODEL_DIR)
    print('=' * 80)
    print(f"Current PRETRAINED_MODEL_DIR: {PRETRAINED_MODEL_DIR}")
    print(f"encoder model_type: {encoder.config.model_type}")
    print(f"hidden_size: {encoder.config.hidden_size}")
    print(f"num_hidden_layers: {encoder.config.num_hidden_layers}")
    print(f"num_attention_heads: {encoder.config.num_attention_heads}")
    print('=' * 80)

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

    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError(f"Best model not found: {BEST_MODEL_PATH}. 请先运行 python train_CME.py")
    print(f"Loading weights from {BEST_MODEL_PATH}...")
    state_dict = torch.load(BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return tokenizer, model


def load_dev_selected_params():
    threshold_vec = np.full((ENT_CLS_NUM,), float(DEFAULT_PRED_THRESHOLD), dtype=np.float32)
    selected_gamma = float(DEFAULT_SPAN_TYPE_FUSION_GAMMA)
    selected_beta = float(DEFAULT_MULTISCALE_FUSION_BETA)
    payload: Dict = {}

    if os.path.exists(BEST_THRESHOLDS_JSON):
        with open(BEST_THRESHOLDS_JSON, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        selected_gamma = float(payload.get('selected_span_type_fusion_gamma', selected_gamma))
        selected_beta = float(payload.get('selected_multiscale_fusion_beta', selected_beta))
        if USE_THRESHOLD_SEARCH:
            thresholds = payload.get('thresholds', {})
            for cls_idx in range(ENT_CLS_NUM):
                label_name = id2ent[cls_idx]
                if label_name in thresholds:
                    threshold_vec[cls_idx] = float(thresholds[label_name])
    else:
        print(f"Warning: threshold file not found: {BEST_THRESHOLDS_JSON}. Use default threshold/gamma/beta.")

    if not (USE_SPAN_TYPE_SEMANTIC_FUSION_INNOVATION and USE_SPAN_TYPE_CONTRASTIVE and USE_SPAN_TYPE_FUSION):
        selected_gamma = 0.0
    if not (USE_MULTISCALE_CALIBRATION_THRESHOLD_INNOVATION and USE_MULTISCALE_FUSION):
        selected_beta = 0.0

    print(f"Use dev-selected gamma={selected_gamma}, beta={selected_beta}")
    print("Use thresholds:", {id2ent[i]: float(threshold_vec[i]) for i in range(ENT_CLS_NUM)})
    return threshold_vec, selected_gamma, selected_beta, payload


def fuse_for_eval(model, outputs, attention_mask, gamma: float, beta: float):
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


def calc_metrics_from_counts(tp: int, pred: int, true: int):
    precision = float(tp / pred) if pred > 0 else 0.0
    recall = float(tp / true) if true > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def evaluate_labeled_test(model, loader, threshold_vec: np.ndarray, gamma: float, beta: float):
    tp = np.zeros((ENT_CLS_NUM,), dtype=np.int64)
    pred = np.zeros((ENT_CLS_NUM,), dtype=np.int64)
    true = np.zeros((ENT_CLS_NUM,), dtype=np.int64)
    threshold_tensor = torch.tensor(threshold_vec, dtype=torch.float32, device=device).view(1, ENT_CLS_NUM, 1, 1)

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc='Testing'):
            raw_text_list, input_ids, attention_mask, segment_ids, labels, _, _ = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            segment_ids = segment_ids.to(device)
            labels = labels.to(device)

            outputs = model(
                input_ids,
                attention_mask,
                segment_ids,
                return_span_type=(USE_SPAN_TYPE_CONTRASTIVE and USE_SPAN_TYPE_FUSION and float(gamma) != 0.0),
                return_multiscale=(USE_MULTISCALE_FUSION and float(beta) != 0.0),
            )
            logits = fuse_for_eval(model, outputs, attention_mask, gamma, beta)
            pred_mask = logits > threshold_tensor
            true_mask = labels.bool()

            for cls_idx in range(ENT_CLS_NUM):
                pred_c = pred_mask[:, cls_idx]
                true_c = true_mask[:, cls_idx]
                tp[cls_idx] += int((pred_c & true_c).sum().item())
                pred[cls_idx] += int(pred_c.sum().item())
                true[cls_idx] += int(true_c.sum().item())

    class_metrics: List[Dict] = []
    for cls_idx in range(ENT_CLS_NUM):
        p, r, f1 = calc_metrics_from_counts(int(tp[cls_idx]), int(pred[cls_idx]), int(true[cls_idx]))
        class_metrics.append({
            'label': id2ent[cls_idx],
            'tp': int(tp[cls_idx]),
            'pred': int(pred[cls_idx]),
            'true': int(true[cls_idx]),
            'precision': p,
            'recall': r,
            'f1': f1,
            'threshold': float(threshold_vec[cls_idx]),
        })

    total_tp = int(tp.sum())
    total_pred = int(pred.sum())
    total_true = int(true.sum())
    precision, recall, f1 = calc_metrics_from_counts(total_tp, total_pred, total_true)
    overall = {
        'tp': total_tp,
        'pred': total_pred,
        'true': total_true,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }
    return overall, class_metrics


def save_test_outputs(overall: Dict, class_metrics: List[Dict], gamma: float, beta: float, threshold_vec: np.ndarray, dev_payload: Dict):
    ensure_parent(TEST_CLASS_METRICS_CSV)
    with open(TEST_CLASS_METRICS_CSV, 'w', newline='', encoding='utf-8') as f:
        fields = ['label', 'tp', 'pred', 'true', 'precision', 'recall', 'f1', 'threshold']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for item in class_metrics:
            writer.writerow({
                'label': item['label'],
                'tp': item['tp'],
                'pred': item['pred'],
                'true': item['true'],
                'precision': f"{item['precision']:.6f}",
                'recall': f"{item['recall']:.6f}",
                'f1': f"{item['f1']:.6f}",
                'threshold': f"{item['threshold']:.6f}",
            })

    payload = {
        'test_file': os.path.join(DATA_DIR, TEST_FILE),
        'best_model_path': BEST_MODEL_PATH,
        'best_thresholds_json': BEST_THRESHOLDS_JSON,
        'selected_span_type_fusion_gamma': float(gamma),
        'selected_multiscale_fusion_beta': float(beta),
        'thresholds': {id2ent[i]: float(threshold_vec[i]) for i in range(ENT_CLS_NUM)},
        'overall': overall,
        'class_metrics': class_metrics,
        'dev_selected_payload': dev_payload,
        'note': 'Thresholds/gamma/beta are selected on dev only. Test is used once for final evaluation.',
    }
    ensure_parent(TEST_METRICS_JSON)
    with open(TEST_METRICS_JSON, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Saved test class metrics to: {TEST_CLASS_METRICS_CSV}")
    print(f"Saved test metrics json to: {TEST_METRICS_JSON}")


def main():
    set_seed(42)
    tokenizer, model = load_model()
    test_path = resolve_data_path(os.path.join(DATA_DIR, TEST_FILE))
    print(f"Loading labeled test data from: {test_path}")
    test_data = load_data(test_path)
    print(f"Test samples: {len(test_data)}")
    test_dataset = EntDataset(test_data, tokenizer=tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=test_dataset.collate,
        shuffle=False,
        num_workers=0,
    )

    threshold_vec, gamma, beta, dev_payload = load_dev_selected_params()
    overall, class_metrics = evaluate_labeled_test(model, test_loader, threshold_vec, gamma, beta)

    print("\nTest class metrics:")
    print(f"{'label':<14}{'TP':>8}{'Pred':>8}{'True':>8}{'Precision':>12}{'Recall':>12}{'F1':>12}{'Threshold':>12}")
    for item in class_metrics:
        print(
            f"{item['label']:<14}{item['tp']:>8}{item['pred']:>8}{item['true']:>8}"
            f"{item['precision']:>12.6f}{item['recall']:>12.6f}{item['f1']:>12.6f}{item['threshold']:>12.3f}"
        )
    print(
        f"\nTest overall: precision={overall['precision']:.6f}, "
        f"recall={overall['recall']:.6f}, f1={overall['f1']:.6f}, "
        f"TP={overall['tp']}, Pred={overall['pred']}, True={overall['true']}"
    )
    save_test_outputs(overall, class_metrics, gamma, beta, threshold_vec, dev_payload)


if __name__ == '__main__':
    main()
