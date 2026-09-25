# -*- coding: utf-8 -*-
"""
训练日志与参数保存、模型保存工具
"""
import csv
import json
import os
from typing import Dict, List


def ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def save_params(params: Dict, path: str) -> None:
    ensure_parent(path)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(params, f, ensure_ascii=False, indent=2)


def append_epoch_log(csv_path: str, epoch: int, loss: float, f1: float, precision: float, recall: float) -> None:
    ensure_parent(csv_path)
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['epoch', 'loss', 'f1', 'precision', 'recall'])
        writer.writerow([epoch, f'{loss:.6f}', f'{f1:.6f}', f'{precision:.6f}', f'{recall:.6f}'])
        f.flush()

def append_step_log(csv_path: str, epoch: int, step: int, loss: float, avg_loss: float) -> None:
    """
    实时逐步写入训练日志（按step），包含本步loss与当前平均loss。
    """
    ensure_parent(csv_path)
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['epoch', 'step', 'loss', 'avg_loss'])
        writer.writerow([epoch, step, f'{loss:.6f}', f'{avg_loss:.6f}'])
        f.flush()



def append_class_metrics(
    csv_path: str,
    epoch: int,
    class_metrics: List[Dict],
) -> None:
    """追加保存每个epoch的验证集逐类别指标。"""
    ensure_parent(csv_path)
    file_exists = os.path.exists(csv_path)
    fields = [
        'epoch', 'label', 'tp', 'fp', 'fn', 'pred', 'true', 'support',
        'precision', 'recall', 'f1',
    ]
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not file_exists:
            writer.writeheader()
        for item in class_metrics:
            writer.writerow({
                'epoch': int(epoch),
                'label': item['label'],
                'tp': int(item['tp']),
                'fp': int(item['fp']),
                'fn': int(item['fn']),
                'pred': int(item['pred']),
                'true': int(item['true']),
                'support': int(item['support']),
                'precision': f"{item['precision']:.6f}",
                'recall': f"{item['recall']:.6f}",
                'f1': f"{item['f1']:.6f}",
            })
        f.flush()


def save_best_class_metrics_csv(
    csv_path: str,
    epoch: int,
    class_metrics: List[Dict],
) -> None:
    """覆盖保存最佳验证集F1对应轮次的逐类别指标。"""
    ensure_parent(csv_path)
    fields = [
        'epoch', 'label', 'tp', 'fp', 'fn', 'pred', 'true', 'support',
        'precision', 'recall', 'f1',
    ]
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for item in class_metrics:
            writer.writerow({
                'epoch': int(epoch),
                'label': item['label'],
                'tp': int(item['tp']),
                'fp': int(item['fp']),
                'fn': int(item['fn']),
                'pred': int(item['pred']),
                'true': int(item['true']),
                'support': int(item['support']),
                'precision': f"{item['precision']:.6f}",
                'recall': f"{item['recall']:.6f}",
                'f1': f"{item['f1']:.6f}",
            })


def save_best_class_metrics_json(
    json_path: str,
    epoch: int,
    overall_metrics: Dict,
    class_metrics: List[Dict],
) -> None:
    """保存最佳轮次的总体指标、宏平均指标和逐类别指标。"""
    macro_precision = (
        sum(float(item['precision']) for item in class_metrics)
        / len(class_metrics)
        if class_metrics else 0.0
    )
    macro_recall = (
        sum(float(item['recall']) for item in class_metrics)
        / len(class_metrics)
        if class_metrics else 0.0
    )
    macro_f1 = (
        sum(float(item['f1']) for item in class_metrics)
        / len(class_metrics)
        if class_metrics else 0.0
    )
    payload = {
        'epoch': int(epoch),
        'evaluation_split': 'validation',
        'matching_rule': (
            'Strict entity-level exact match: entity type, start token and '
            'end token must all match.'
        ),
        'decision_threshold': 0.0,
        'overall_micro': {
            'precision': float(overall_metrics['precision']),
            'recall': float(overall_metrics['recall']),
            'f1': float(overall_metrics['f1']),
        },
        'overall_macro': {
            'precision': float(macro_precision),
            'recall': float(macro_recall),
            'f1': float(macro_f1),
        },
        'classes': [
            {
                'label': item['label'],
                'tp': int(item['tp']),
                'fp': int(item['fp']),
                'fn': int(item['fn']),
                'pred': int(item['pred']),
                'true': int(item['true']),
                'support': int(item['support']),
                'precision': float(item['precision']),
                'recall': float(item['recall']),
                'f1': float(item['f1']),
            }
            for item in class_metrics
        ],
    }
    save_params(payload, json_path)



def save_best_model(model, path: str) -> None:
    import torch
    ensure_parent(path)
    torch.save(model.state_dict(), path)

def save_topk_model(model, f1_score: float, epoch: int, out_dir: str, meta_path: str, k: int = 5) -> None:
    """
    保存当前模型到out_dir，并使用meta文件维护前k个最优模型（按f1降序）。
    超出k的旧模型会被自动删除。
    """
    import torch
    ensure_parent(meta_path)
    os.makedirs(out_dir, exist_ok=True)

    # 生成文件名：f1保留4位小数，含epoch
    filename = f'best_f1_{f1_score:.4f}_epoch_{epoch}.pth'
    filepath = os.path.join(out_dir, filename)
    torch.save(model.state_dict(), filepath)

    # 读取现有meta
    records = []
    if os.path.exists(meta_path):
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                records = json.load(f)
        except Exception:
            records = []

    # 加入新纪录并排序
    records.append({"path": filepath, "f1": float(f1_score), "epoch": int(epoch)})
    records.sort(key=lambda x: x["f1"], reverse=True)

    # 保留前k，删除其余文件
    to_keep = records[:k]
    to_delete = records[k:]
    for item in to_delete:
        try:
            if os.path.exists(item["path"]):
                os.remove(item["path"])
        except Exception:
            pass

    # 写回meta
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(to_keep, f, ensure_ascii=False, indent=2)


