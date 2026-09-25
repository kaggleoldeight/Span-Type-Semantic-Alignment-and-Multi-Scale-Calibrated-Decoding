# -*- coding: utf-8 -*-
"""
ChiNesE 数据集 BERT-CRF 训练脚本。

重要说明：
- BERT-CRF 是 flat 序列标注基线；
- ChiNesE 是嵌套 NER 数据集，训练时通过 innermost/outermost 策略转为单层 BIO；
- 验证时仍使用完整 gold entities 计算 P/R/F1。
"""
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from bert_crf import BertCRF
from data_loader import EntDataset, crf_id2label, ent2id, id2ent, load_data
from train_logger import (
    append_class_metrics,
    append_epoch_log,
    append_step_log,
    save_best_model,
    save_class_metrics,
    save_params,
    save_topk_model,
)
from config import (
    BATCH_SIZE,
    BEST_CLASS_METRICS_CSV,
    BEST_MODEL_PATH,
    BEST_MODELS_DIR,
    BEST_MODELS_META,
    CLASS_METRICS_CSV,
    CRF_LEARNING_RATE,
    DATA_DIR,
    DEV_FILE,
    DEVICE,
    DROPOUT_PROB,
    EARLY_STOP_PATIENCE,
    EPOCHS,
    FLAT_NER_STRATEGY,
    LEARNING_RATE,
    MAX_LEN,
    NUM_CRF_LABELS,
    NUM_WORKERS,
    OUTPUT_DIR,
    PRETRAINED_MODEL_DIR,
    SEED,
    TRAIN_FILE,
    TRAIN_LOG_CSV,
    TRAIN_PARAMS_JSON,
    TRAIN_STEP_LOG_CSV,
    USE_EARLY_STOPPING,
    WARMUP_RATIO,
    WEIGHT_DECAY,
)

train_path = os.path.join(DATA_DIR, TRAIN_FILE)
eval_path = os.path.join(DATA_DIR, DEV_FILE)
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
    """把 CRF 解码标签还原为字符级实体 span。"""
    entities: List[Tuple[int, int, str]] = []
    current_type = None
    current_start = None
    current_end = None

    for idx, label_id in enumerate(label_ids):
        if idx >= len(offset_mapping):
            break
        token_start, token_end = offset_mapping[idx]

        # special token 的 offset 通常为 (0, 0)，不参与实体还原。
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
                # 非法 I 开头或类型切换，按 B 处理，避免整段丢失。
                if current_type is not None:
                    entities.append((int(current_start), int(current_end), current_type))
                current_type = ent_type
                current_start = int(token_start)
                current_end = int(token_end) - 1

    if current_type is not None:
        entities.append((int(current_start), int(current_end), current_type))

    valid_entities = []
    seen = set()
    for start, end, ent_type in entities:
        if 0 <= start <= end < len(text) and ent_type in ent2id:
            key = (start, end, ent_type)
            if key not in seen:
                seen.add(key)
                valid_entities.append(key)
    return valid_entities


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
    for cls_idx in range(len(class_tp)):
        tp = int(class_tp[cls_idx])
        pred = int(class_pred[cls_idx])
        true = int(class_true[cls_idx])
        precision = tp / pred if pred > 0 else 0.0
        recall = tp / true if true > 0 else 0.0
        f1 = 2 * tp / (pred + true) if (pred + true) > 0 else 0.0
        class_metrics.append({
            'label': id2ent.get(cls_idx, str(cls_idx)),
            'tp': tp,
            'pred': pred,
            'true': true,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        })
    return class_metrics


def print_class_metrics(epoch, class_metrics):
    print(f"\nEpoch {epoch} class metrics:")
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


def evaluate(model, data_loader):
    model.eval()
    total_X, total_Y, total_Z = 0, 0, 0
    class_tp = [0 for _ in range(len(ent2id))]
    class_pred = [0 for _ in range(len(ent2id))]
    class_true = [0 for _ in range(len(ent2id))]

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Valing"):
            raw_text_list, input_ids, attention_mask, segment_ids, labels, offsets, gold_entities = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            segment_ids = segment_ids.to(device)
            labels = labels.to(device)

            _, _, decoded = model(input_ids, attention_mask, segment_ids, labels=None)

            for i, label_ids in enumerate(decoded):
                pred_entities = decode_entities_from_labels(raw_text_list[i], label_ids, offsets[i])
                X, Y, Z = update_counts_from_entities(pred_entities, gold_entities[i], class_tp, class_pred, class_true)
                total_X += X
                total_Y += Y
                total_Z += Z

    precision = total_X / total_Y if total_Y > 0 else 0.0
    recall = total_X / total_Z if total_Z > 0 else 0.0
    f1 = 2 * total_X / (total_Y + total_Z) if (total_Y + total_Z) > 0 else 0.0
    class_metrics = build_class_metrics(class_tp, class_pred, class_true)
    return precision, recall, f1, class_metrics, total_X, total_Y, total_Z


def main():
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading tokenizer from {PRETRAINED_MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_DIR, use_fast=True)
    if not tokenizer.is_fast:
        raise RuntimeError('BERT-CRF 版本需要 fast tokenizer 以获得 offset_mapping，请使用 AutoTokenizer(..., use_fast=True)。')

    train_data = load_data(train_path)
    eval_data = load_data(eval_path)
    print(f"Train samples: {len(train_data)}")
    print(f"Dev samples: {len(eval_data)}")
    print(f"Flat NER strategy for CRF training: {FLAT_NER_STRATEGY}")

    train_dataset = EntDataset(train_data, tokenizer=tokenizer, flat_strategy=FLAT_NER_STRATEGY)
    eval_dataset = EntDataset(eval_data, tokenizer=tokenizer, flat_strategy=FLAT_NER_STRATEGY)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=train_dataset.collate,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=eval_dataset.collate,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    print(f"Loading BERT encoder from {PRETRAINED_MODEL_DIR}...")
    model = BertCRF(PRETRAINED_MODEL_DIR, num_labels=NUM_CRF_LABELS, dropout_prob=DROPOUT_PROB).to(device)
    print("=" * 80)
    print(f"Current PRETRAINED_MODEL_DIR: {PRETRAINED_MODEL_DIR}")
    print(f"encoder model_type: {model.encoder.config.model_type}")
    print(f"hidden_size: {model.encoder.config.hidden_size}")
    print(f"num_hidden_layers: {model.encoder.config.num_hidden_layers}")
    print(f"num_attention_heads: {model.encoder.config.num_attention_heads}")
    print("Model: BERT-CRF flat sequence labeling baseline")
    print("=" * 80)

    optimizer_grouped_parameters = [
        {'params': model.encoder.parameters(), 'lr': LEARNING_RATE},
        {'params': list(model.classifier.parameters()) + list(model.crf.parameters()), 'lr': CRF_LEARNING_RATE},
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, weight_decay=WEIGHT_DECAY)

    total_training_steps = len(train_loader) * EPOCHS
    warmup_steps = int(WARMUP_RATIO * total_training_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    save_params({
        "model": "BERT-CRF",
        "pretrained_model_dir": PRETRAINED_MODEL_DIR,
        "train_file": train_path,
        "dev_file": eval_path,
        "flat_ner_strategy": FLAT_NER_STRATEGY,
        "max_len": MAX_LEN,
        "batch_size": BATCH_SIZE,
        "num_crf_labels": NUM_CRF_LABELS,
        "learning_rate": LEARNING_RATE,
        "crf_learning_rate": CRF_LEARNING_RATE,
        "dropout_prob": DROPOUT_PROB,
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        "epochs": EPOCHS,
        "use_early_stopping": USE_EARLY_STOPPING,
        "early_stop_patience": EARLY_STOP_PATIENCE,
        "evaluation_gold": "full ChiNesE nested gold entities",
        "output_dir": OUTPUT_DIR,
    }, TRAIN_PARAMS_JSON)

    best_f1 = 0.0
    no_improve_epochs = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Train {epoch + 1}/{EPOCHS}")

        for step, batch in enumerate(pbar):
            raw_text_list, input_ids, attention_mask, segment_ids, labels, offsets, gold_entities = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            segment_ids = segment_ids.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            loss, _, _ = model(input_ids, attention_mask, segment_ids, labels=labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += float(loss.item())
            avg_loss = total_loss / (step + 1)
            pbar.set_postfix({"loss": f"{avg_loss:.6f}"})
            append_step_log(TRAIN_STEP_LOG_CSV, epoch, step + 1, loss.item(), avg_loss)

        avg_loss = total_loss / max(len(train_loader), 1)
        precision, recall, f1, class_metrics, total_X, total_Y, total_Z = evaluate(model, eval_loader)
        print_class_metrics(epoch, class_metrics)
        append_class_metrics(CLASS_METRICS_CSV, epoch, class_metrics)
        append_epoch_log(TRAIN_LOG_CSV, epoch, avg_loss, f1, precision, recall)

        is_best = f1 > best_f1
        if is_best:
            save_best_model(model, BEST_MODEL_PATH)
            save_class_metrics(BEST_CLASS_METRICS_CSV, epoch, class_metrics)
            best_f1 = f1
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        save_topk_model(model, f1, epoch, BEST_MODELS_DIR, BEST_MODELS_META, k=5)

        print(
            f"Epoch {epoch} -> F1: {f1:.6f}  Precision: {precision:.6f}  "
            f"Recall: {recall:.6f}  Best_F1: {best_f1:.6f}  "
            f"TP={total_X} Pred={total_Y} True={total_Z}"
        )

        if USE_EARLY_STOPPING and no_improve_epochs >= EARLY_STOP_PATIENCE:
            print(f"Early stopping triggered. No F1 improvement for {EARLY_STOP_PATIENCE} epochs.")
            break

    try:
        from plot_metrics import plot_curves
        plot_curves()
    except Exception as e:
        print(f"Plot skipped: {e}")


if __name__ == '__main__':
    import torch.multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()
