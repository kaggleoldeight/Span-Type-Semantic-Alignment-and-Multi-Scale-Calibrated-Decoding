# -*- coding: utf-8 -*-
"""
@Time: 2021/8/27 16:12
@Auth: Xhw0
@File: entity_extract.py
@Description: 实体抽取.
"""
import csv
import platform
import random
import time
from typing import Dict, List
import numpy as np
import os
from data_loader import EntDataset, id2ent, load_data
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
    CLASS_METRICS_CSV,
    BEST_CLASS_METRICS_CSV,
    BEST_CLASS_METRICS_JSON,
    ENABLE_COMPUTE_COST_PROFILING,
    TRAIN_COMPUTE_COST_CSV,
    TRAIN_COMPUTE_COST_JSON,
)
from config import TRAIN_LOG_CSV, TRAIN_STEP_LOG_CSV, TRAIN_PARAMS_JSON, BEST_MODEL_PATH, BEST_MODELS_DIR, BEST_MODELS_META
from train_logger import (
    append_class_metrics,
    append_epoch_log,
    append_step_log,
    save_best_class_metrics_csv,
    save_best_class_metrics_json,
    save_best_model,
    save_params,
    save_topk_model,
)
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



def count_model_parameters(model: torch.nn.Module) -> Dict[str, float]:
    """统计参数量；不改变模型参数或requires_grad状态。"""
    total_parameters = sum(p.numel() for p in model.parameters())
    trainable_parameters = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    parameter_bytes = sum(
        p.numel() * p.element_size() for p in model.parameters()
    )
    return {
        'total_parameters': int(total_parameters),
        'trainable_parameters': int(trainable_parameters),
        'frozen_parameters': int(total_parameters - trainable_parameters),
        'parameter_storage_mib': float(parameter_bytes / (1024 ** 2)),
    }


def get_runtime_environment() -> Dict:
    """保存计算代价测量所使用的软硬件环境。"""
    cuda_available = bool(torch.cuda.is_available())
    cuda_device_index = None
    gpu_name = None
    if cuda_available:
        cuda_device_index = int(torch.cuda.current_device())
        gpu_name = torch.cuda.get_device_name(cuda_device_index)
    return {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'pytorch_version': torch.__version__,
        'cuda_available': cuda_available,
        'cuda_runtime_version': torch.version.cuda,
        'cudnn_version': (
            torch.backends.cudnn.version() if cuda_available else None
        ),
        'device': str(device),
        'cuda_device_index': cuda_device_index,
        'gpu_name': gpu_name,
    }


def synchronize_cuda() -> None:
    """同步CUDA异步任务，防止计时被低估。"""
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)


def reset_cuda_peak_memory() -> None:
    """重置当前训练轮次的CUDA峰值统计。"""
    if torch.cuda.is_available():
        synchronize_cuda()
        torch.cuda.reset_peak_memory_stats(device)


def get_cuda_peak_memory() -> Dict[str, float]:
    """读取PyTorch allocated与reserved峰值显存。"""
    if not torch.cuda.is_available():
        return {
            'peak_allocated_bytes': None,
            'peak_allocated_gib': None,
            'peak_reserved_bytes': None,
            'peak_reserved_gib': None,
        }
    synchronize_cuda()
    allocated = int(torch.cuda.max_memory_allocated(device))
    reserved = int(torch.cuda.max_memory_reserved(device))
    return {
        'peak_allocated_bytes': allocated,
        'peak_allocated_gib': float(allocated / (1024 ** 3)),
        'peak_reserved_bytes': reserved,
        'peak_reserved_gib': float(reserved / (1024 ** 3)),
    }


def append_training_cost_csv(csv_path: str, record: Dict) -> None:
    """逐轮追加训练计算代价。"""
    parent = os.path.dirname(csv_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fields = [
        'epoch',
        'train_phase_seconds',
        'full_epoch_wall_seconds',
        'train_samples',
        'train_batches',
        'samples_per_second',
        'peak_allocated_gib',
        'peak_reserved_gib',
    ]
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not file_exists:
            writer.writeheader()
        writer.writerow({field: record.get(field) for field in fields})
        f.flush()


def build_training_cost_summary(
    parameter_statistics: Dict,
    epoch_records: List[Dict],
    total_run_wall_seconds: float,
) -> Dict:
    """
    平均单轮训练时间只统计原训练循环，不包括验证、模型保存和绘图。
    """
    epoch_times = [
        float(item['train_phase_seconds']) for item in epoch_records
    ]
    throughputs = [
        float(item['samples_per_second']) for item in epoch_records
    ]
    allocated_peaks = [
        float(item['peak_allocated_gib'])
        for item in epoch_records
        if item.get('peak_allocated_gib') is not None
    ]
    reserved_peaks = [
        float(item['peak_reserved_gib'])
        for item in epoch_records
        if item.get('peak_reserved_gib') is not None
    ]
    return {
        'measurement_definition': {
            'average_epoch_training_time': (
                'Mean wall-clock time of the original training phase. '
                'Data loading, forward/backward computation, configured '
                'adversarial steps, gradient clipping, optimizer update and '
                'scheduler update are included. Validation, checkpoint I/O '
                'and plotting are excluded.'
            ),
            'training_peak_memory': (
                'Maximum torch.cuda allocated memory in the training phase '
                'after resetting peak statistics at the start of each epoch.'
            ),
        },
        'environment': get_runtime_environment(),
        'parameter_statistics': parameter_statistics,
        'completed_epochs': int(len(epoch_records)),
        'average_epoch_training_seconds': (
            float(np.mean(epoch_times)) if epoch_times else None
        ),
        'std_epoch_training_seconds': (
            float(np.std(epoch_times, ddof=1))
            if len(epoch_times) > 1
            else 0.0 if epoch_times else None
        ),
        'average_training_throughput_samples_per_second': (
            float(np.mean(throughputs)) if throughputs else None
        ),
        'training_peak_allocated_gib': (
            max(allocated_peaks) if allocated_peaks else None
        ),
        'training_peak_reserved_gib': (
            max(reserved_peaks) if reserved_peaks else None
        ),
        'total_run_wall_seconds': float(total_run_wall_seconds),
        'epoch_records': epoch_records,
    }


def build_class_metrics(
    class_tp: np.ndarray,
    class_pred: np.ndarray,
    class_true: np.ndarray,
) -> List[Dict]:
    """由验证集严格实体匹配计数生成逐类别指标。"""
    class_metrics = []
    for class_id in range(ENT_CLS_NUM):
        tp = int(class_tp[class_id])
        pred = int(class_pred[class_id])
        true = int(class_true[class_id])
        fp = pred - tp
        fn = true - tp
        precision = tp / pred if pred > 0 else 0.0
        recall = tp / true if true > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
        class_metrics.append({
            'label': id2ent[class_id],
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'pred': pred,
            'true': true,
            'support': true,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        })
    return class_metrics


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
    ).to(device) #9个实体类型

    parameter_statistics = count_model_parameters(model)
    print("Model parameter statistics:")
    print(
        f"  Total parameters: "
        f"{parameter_statistics['total_parameters']:,}"
    )
    print(
        f"  Trainable parameters: "
        f"{parameter_statistics['trainable_parameters']:,}"
    )
    print(
        f"  Frozen parameters: "
        f"{parameter_statistics['frozen_parameters']:,}"
    )
    print(
        f"  Parameter storage: "
        f"{parameter_statistics['parameter_storage_mib']:.2f} MiB"
    )

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
        "best_class_metrics_json": BEST_CLASS_METRICS_JSON,
        "compute_cost_profiling_enabled": ENABLE_COMPUTE_COST_PROFILING,
        "train_compute_cost_csv": TRAIN_COMPUTE_COST_CSV,
        "train_compute_cost_json": TRAIN_COMPUTE_COST_JSON,
        "parameter_statistics": parameter_statistics,
    }, TRAIN_PARAMS_JSON)
    epoch_cost_records: List[Dict] = []
    training_run_start = time.perf_counter()

    # 防止本次运行的逐轮代价与历史运行混合。
    if (
        ENABLE_COMPUTE_COST_PROFILING
        and os.path.exists(TRAIN_COMPUTE_COST_CSV)
    ):
        os.remove(TRAIN_COMPUTE_COST_CSV)

    for eo in range(EPOCHS):
        full_epoch_start = time.perf_counter()
        if ENABLE_COMPUTE_COST_PROFILING:
            reset_cuda_peak_memory()
        synchronize_cuda()
        train_phase_start = time.perf_counter()

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

        synchronize_cuda()
        train_phase_seconds = time.perf_counter() - train_phase_start
        train_peak_memory = (
            get_cuda_peak_memory()
            if ENABLE_COMPUTE_COST_PROFILING
            else {
                'peak_allocated_gib': None,
                'peak_reserved_gib': None,
            }
        )

        with torch.no_grad():
            total_X, total_Y, total_Z = 0, 0, 0
            class_tp = np.zeros(ENT_CLS_NUM, dtype=np.int64)
            class_pred = np.zeros(ENT_CLS_NUM, dtype=np.int64)
            class_true = np.zeros(ENT_CLS_NUM, dtype=np.int64)
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

                pred_mask = logits > 0
                true_mask = labels > 0
                for class_id in range(ENT_CLS_NUM):
                    pred_class = pred_mask[:, class_id]
                    true_class = true_mask[:, class_id]
                    class_tp[class_id] += int(
                        (pred_class & true_class).sum().item()
                    )
                    class_pred[class_id] += int(
                        pred_class.sum().item()
                    )
                    class_true[class_id] += int(
                        true_class.sum().item()
                    )
            avg_precision = total_X / total_Y if total_Y > 0 else 0.0
            avg_recall = total_X / total_Z if total_Z > 0 else 0.0
            avg_f1 = 2 * total_X / (total_Y + total_Z) if (total_Y + total_Z) > 0 else 0.0

            class_metrics = build_class_metrics(
                class_tp,
                class_pred,
                class_true,
            )
            append_epoch_log(TRAIN_LOG_CSV, eo, avg_loss, avg_f1, avg_precision, avg_recall)
            append_class_metrics(
                CLASS_METRICS_CSV,
                eo,
                class_metrics,
            )

            is_best = avg_f1 > max_f
            if is_best:
                save_best_model(model, BEST_MODEL_PATH)
                save_best_class_metrics_csv(
                    BEST_CLASS_METRICS_CSV,
                    eo,
                    class_metrics,
                )
                save_best_class_metrics_json(
                    BEST_CLASS_METRICS_JSON,
                    eo,
                    {
                        'precision': avg_precision,
                        'recall': avg_recall,
                        'f1': avg_f1,
                    },
                    class_metrics,
                )
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

        if ENABLE_COMPUTE_COST_PROFILING:
            full_epoch_wall_seconds = (
                time.perf_counter() - full_epoch_start
            )
            train_samples = int(len(ner_train))
            samples_per_second = (
                float(train_samples / train_phase_seconds)
                if train_phase_seconds > 0
                else 0.0
            )
            epoch_cost_record = {
                'epoch': int(eo + 1),
                'train_phase_seconds': float(train_phase_seconds),
                'full_epoch_wall_seconds': float(
                    full_epoch_wall_seconds
                ),
                'train_samples': train_samples,
                'train_batches': int(len(ner_loader_train)),
                'samples_per_second': samples_per_second,
                'peak_allocated_gib': train_peak_memory.get(
                    'peak_allocated_gib'
                ),
                'peak_reserved_gib': train_peak_memory.get(
                    'peak_reserved_gib'
                ),
            }
            epoch_cost_records.append(epoch_cost_record)
            append_training_cost_csv(
                TRAIN_COMPUTE_COST_CSV,
                epoch_cost_record,
            )
            save_params(
                build_training_cost_summary(
                    parameter_statistics,
                    epoch_cost_records,
                    time.perf_counter() - training_run_start,
                ),
                TRAIN_COMPUTE_COST_JSON,
            )

        # 基于F1的早停（可通过配置开关禁用）
        if USE_EARLY_STOPPING and no_improve_epochs >= EARLY_STOP_PATIENCE:
            print("Early stopping triggered. No F1 improvement for {} epochs.".format(EARLY_STOP_PATIENCE))
            break

    if ENABLE_COMPUTE_COST_PROFILING:
        final_compute_summary = build_training_cost_summary(
            parameter_statistics,
            epoch_cost_records,
            time.perf_counter() - training_run_start,
        )
        save_params(
            final_compute_summary,
            TRAIN_COMPUTE_COST_JSON,
        )
        print(
            "Training compute-cost summary saved to: "
            f"{TRAIN_COMPUTE_COST_JSON}"
        )

    from plot_metrics import plot_curves
    plot_curves()


if __name__ == '__main__':
    import torch.multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()
