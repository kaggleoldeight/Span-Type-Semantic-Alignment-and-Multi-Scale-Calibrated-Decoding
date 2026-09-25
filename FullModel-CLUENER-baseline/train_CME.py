# -*- coding: utf-8 -*-
"""
@Time: 2021/8/27 16:12
@Auth: Xhw0
@File: entity_extract.py
@Description: 实体抽取.
"""
import random
import numpy as np
import os
from data_loader import EntDataset, load_data, id2ent
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
import torch
from GlobalPointer import GlobalPointer, MetricsCalculator
from tqdm import tqdm
from config import (
    PRETRAINED_MODEL_DIR,
    DATA_DIR,
    TRAIN_FILE,
    DEV_FILE,
    DEVICE,
    BATCH_SIZE,
    ENT_CLS_NUM,
    GP_HEAD_SIZE,
    GP_EFFICIENT,
    LEARNING_RATE,
    NUM_WORKERS,
    EPOCHS,
    EARLY_STOP_PATIENCE,
    USE_EARLY_STOPPING,
    WEIGHT_DECAY,
    WARMUP_RATIO,
    RNN_HIDDEN_SIZE,
    RNN_LAYERS,
    USE_BIGRU,
)
from config import TRAIN_LOG_CSV, TRAIN_STEP_LOG_CSV, TRAIN_PARAMS_JSON, BEST_MODEL_PATH, BEST_MODELS_DIR, BEST_MODELS_META, CLASS_METRICS_CSV, BEST_CLASS_METRICS_CSV
from train_logger import append_epoch_log, append_step_log, append_class_metrics, save_class_metrics, save_params, save_best_model, save_topk_model
from transformers import get_linear_schedule_with_warmup
from adversarial import AdversarialTrainer
from config import (
    ADV_TRAIN_METHOD,
    ADV_EMB_NAME,
    FGM_EPSILON,
    PGD_EPSILON,
    PGD_ALPHA,
    PGD_STEPS,
)

train_cme_path = f'{DATA_DIR}/{TRAIN_FILE}'
eval_cme_path = f'{DATA_DIR}/{DEV_FILE}'
device = DEVICE


def set_seed(seed=42):
    """
    设置随机种子，确保实验可复现
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 设置PyTorch的确定性模式（可能影响性能，但确保可复现）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # 设置环境变量
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to {seed} for reproducibility")


def multilabel_categorical_crossentropy(y_pred, y_true):
    y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
    y_pred_neg = y_pred - y_true * 1e12  # mask the pred outputs of pos classes
    y_pred_pos = y_pred - (1 - y_true) * 1e12 # mask the pred outputs of neg classes
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return (neg_loss + pos_loss).mean()

def loss_fun(y_true, y_pred):
    """
    y_true:(batch_size, ent_type_size, seq_len, seq_len)
    y_pred:(batch_size, ent_type_size, seq_len, seq_len)
    """
    batch_size, ent_type_size = y_pred.shape[:2]
    y_true = y_true.reshape(batch_size * ent_type_size, -1)
    y_pred = y_pred.reshape(batch_size * ent_type_size, -1)
    loss = multilabel_categorical_crossentropy(y_true, y_pred)
    return loss


def update_class_counts(logits, labels, class_tp, class_pred, class_true):
    """按实体类别累计 TP / Pred / True，阈值与整体评估保持一致：logits > 0。"""
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
    """生成逐类别 Precision / Recall / F1，便于保存 CSV 和打印。"""
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
    """在控制台打印每个类别的指标。"""
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

def main():
    # 设置随机种子，确保可复现
    set_seed(42)
    
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_DIR, use_fast=True)

    ner_train = EntDataset(load_data(train_cme_path), tokenizer=tokenizer)
    ner_loader_train = DataLoader(ner_train , batch_size=BATCH_SIZE, collate_fn=ner_train.collate, shuffle=True, num_workers=NUM_WORKERS)
    ner_evl = EntDataset(load_data(eval_cme_path), tokenizer=tokenizer)
    ner_loader_evl = DataLoader(ner_evl , batch_size=BATCH_SIZE, collate_fn=ner_evl.collate, shuffle=False, num_workers=NUM_WORKERS)

    encoder = AutoModel.from_pretrained(PRETRAINED_MODEL_DIR)
    model = GlobalPointer(
        encoder,
        ENT_CLS_NUM,
        GP_HEAD_SIZE,
        rnn_hidden_size=RNN_HIDDEN_SIZE,
        rnn_layers=RNN_LAYERS,
        efficient=GP_EFFICIENT,
        use_bigru=USE_BIGRU,
    ).to(device)  # 实体类型数由 config.ENT_CLS_NUM 控制，CLUENER 为 10 类

    # 分层学习率：预训练层较小 lr，新加任务层较大学习率
    pretrained_params = list(model.encoder.parameters())
    new_params = []
    if getattr(model, "bigru", None) is not None:
        new_params += list(model.bigru.parameters())

    if GP_EFFICIENT:
        new_params += list(model.q_dense.parameters())
        new_params += list(model.k_dense.parameters())
        new_params += list(model.classifier.parameters())
    else:
        new_params += list(model.dense.parameters())

    optimizer_grouped_parameters = [
        {'params': pretrained_params, 'lr': LEARNING_RATE},
        {'params': new_params, 'lr': LEARNING_RATE * 20},
    ]

    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        weight_decay=WEIGHT_DECAY,
    )

    # 学习率调度：warmup + 线性衰减
    total_training_steps = len(ner_loader_train) * EPOCHS
    warmup_steps = int(WARMUP_RATIO * total_training_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps
    )

    # 对抗训练器（按配置启用）
    use_adversarial = ADV_TRAIN_METHOD.upper() in ['FGM', 'PGD']
    adversarial_trainer = None
    if use_adversarial:
        adversarial_trainer = AdversarialTrainer(
            model,
            method=ADV_TRAIN_METHOD,
            emb_name=ADV_EMB_NAME
        )

    metrics = MetricsCalculator()
    max_f, max_recall = 0.0, 0.0
    no_improve_epochs = 0
    save_params({
        "pretrained_model_dir": PRETRAINED_MODEL_DIR,
        "train_file": train_cme_path,
        "dev_file": eval_cme_path,
        "batch_size": BATCH_SIZE,
        "ent_cls_num": ENT_CLS_NUM,
        "gp_head_size": GP_HEAD_SIZE,
        "learning_rate": LEARNING_RATE,
        "num_workers": NUM_WORKERS,
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        "rnn_hidden_size": RNN_HIDDEN_SIZE,
        "rnn_layers": RNN_LAYERS,
        "use_bigru": USE_BIGRU,
        "adv_train_method": ADV_TRAIN_METHOD,
        "adv_emb_name": ADV_EMB_NAME,
        "fgm_epsilon": FGM_EPSILON,
        "pgd_epsilon": PGD_EPSILON,
        "pgd_alpha": PGD_ALPHA,
        "pgd_steps": PGD_STEPS,
        "use_early_stopping": USE_EARLY_STOPPING,
        "early_stop_patience": EARLY_STOP_PATIENCE,
        "class_metrics_csv": CLASS_METRICS_CSV,
        "best_class_metrics_csv": BEST_CLASS_METRICS_CSV,
    }, TRAIN_PARAMS_JSON)
    for eo in range(EPOCHS):
        total_loss = 0.
        pbar = tqdm(ner_loader_train, desc="Train {}/{}".format(eo+1, EPOCHS))
        for idx, batch in enumerate(pbar):
            raw_text_list, input_ids, attention_mask, segment_ids, labels = batch
            input_ids, attention_mask, segment_ids, labels = input_ids.to(device), attention_mask.to(device), segment_ids.to(device), labels.to(device)
            optimizer.zero_grad()

            # 正常前向与损失
            logits = model(input_ids, attention_mask, segment_ids)
            loss = loss_fun(logits, labels)
            loss.backward()

            # 对抗训练
            if use_adversarial and adversarial_trainer is not None:
                if ADV_TRAIN_METHOD.upper() == 'FGM':
                    adversarial_trainer.attack(epsilon=FGM_EPSILON)
                    logits_adv = model(input_ids, attention_mask, segment_ids)
                    loss_adv = loss_fun(logits_adv, labels)
                    loss_adv.backward()
                    adversarial_trainer.restore()
                elif ADV_TRAIN_METHOD.upper() == 'PGD':
                    # 步骤1: 备份干净样本的梯度 g_clean
                    grad_backup = {}
                    for name, param in model.named_parameters():
                        if param.requires_grad and param.grad is not None:
                            grad_backup[name] = param.grad.clone()
                    
                    # 【修复点】：由于 BiGRU(cuDNN) 的限制，全程保持默认的 train() 模式
                    # 步骤2: K步迭代寻找最坏扰动
                    for t in range(PGD_STEPS):
                        adversarial_trainer.attack(
                            epsilon=PGD_EPSILON,
                            alpha=PGD_ALPHA,
                            is_first_attack=(t == 0)
                        )
                        if t < PGD_STEPS - 1:
                            logits_adv = model(input_ids, attention_mask, segment_ids)
                            loss_adv_temp = loss_fun(logits_adv, labels)
                            # 清零临时梯度
                            model.zero_grad() 
                            loss_adv_temp.backward()
                    
                    # 步骤3: 找准最坏扰动后，计算最终对抗梯度
                    logits_adv = model(input_ids, attention_mask, segment_ids)
                    loss_adv = loss_fun(logits_adv, labels)
                    model.zero_grad()
                    loss_adv.backward()
                    
                    # 步骤4: 平滑梯度。将干净梯度和对抗梯度取平均
                    for name, param in model.named_parameters():
                        if param.requires_grad and name in grad_backup:
                            if param.grad is None:
                                param.grad = grad_backup[name].clone()
                            else:
                                param.grad = (param.grad + grad_backup[name]) / 2.0
                    
                    # 步骤5: 恢复原始 Embedding 权重
                    adversarial_trainer.restore()

            # PGD/对抗训练可能放大梯度尺度，做梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()
            total_loss+=loss.item()

            avg_loss = total_loss / (idx + 1)
            pbar.set_postfix({"loss": "{:.6f}".format(avg_loss)})
            # 实时写入step日志
            append_step_log(TRAIN_STEP_LOG_CSV, eo, idx + 1, loss.item(), avg_loss)

        with torch.no_grad():
            total_X, total_Y, total_Z = 0, 0, 0
            class_tp = [0 for _ in range(ENT_CLS_NUM)]
            class_pred = [0 for _ in range(ENT_CLS_NUM)]
            class_true = [0 for _ in range(ENT_CLS_NUM)]

            model.eval()
            for batch in tqdm(ner_loader_evl, desc="Valing"):
                raw_text_list, input_ids, attention_mask, segment_ids, labels = batch
                input_ids, attention_mask, segment_ids, labels = input_ids.to(device), attention_mask.to(
                    device), segment_ids.to(device), labels.to(device)
                logits = model(input_ids, attention_mask, segment_ids)
                X, Y, Z = metrics.get_evaluate_fpr(logits, labels)
                total_X += X
                total_Y += Y
                total_Z += Z
                update_class_counts(logits, labels, class_tp, class_pred, class_true)

            class_metrics = build_class_metrics(class_tp, class_pred, class_true)
            print_class_metrics(eo, class_metrics)
            append_class_metrics(CLASS_METRICS_CSV, eo, class_metrics)

            avg_precision = total_X / total_Y if total_Y > 0 else 0.0
            avg_recall = total_X / total_Z if total_Z > 0 else 0.0
            avg_f1 = 2 * total_X / (total_Y + total_Z) if (total_Y + total_Z) > 0 else 0.0

            append_epoch_log(TRAIN_LOG_CSV, eo, avg_loss, avg_f1, avg_precision, avg_recall)

            is_best = avg_f1 > max_f
            if is_best:
                save_best_model(model, BEST_MODEL_PATH)
                save_class_metrics(BEST_CLASS_METRICS_CSV, eo, class_metrics)
                max_f = avg_f1
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
            # 统一格式输出当前轮指标与历史最佳F1
            print("Epoch {} -> F1: {:.6f}  Precision: {:.6f}  Recall: {:.6f}  Best_F1: {:.6f}".format(
                eo, avg_f1, avg_precision, avg_recall, max_f
            ))
            model.train()

            save_topk_model(model, avg_f1, eo, BEST_MODELS_DIR, BEST_MODELS_META, k=5)

        # 基于F1的早停（可通过配置开关禁用）
        if USE_EARLY_STOPPING and no_improve_epochs >= EARLY_STOP_PATIENCE:
            print("Early stopping triggered. No F1 improvement for {} epochs.".format(EARLY_STOP_PATIENCE))
            break

    from plot_metrics import plot_curves
    plot_curves()


if __name__ == '__main__':
    import torch.multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()
