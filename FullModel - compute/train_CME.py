# -*- coding: utf-8 -*-
"""
实体抽取训练脚本：Span-Type Semantic Alignment GlobalPointer

保留：RoBERTa + BiGRU + GlobalPointer + PGD + per-class 阈值搜索。
新增：Span-Type 语义对齐辅助损失。
"""
import csv
import json
import os
import platform
import random
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from adversarial import AdversarialTrainer
from data_loader import EntDataset, ent2id, get_label_descriptions, id2ent, load_data
from GlobalPointer import GlobalPointer
from train_logger import (
    append_class_metrics,
    append_epoch_log,
    append_step_log,
    save_best_model,
    save_class_metrics,
    save_params,
    save_thresholds_json,
    save_topk_model,
)
from config import (
    ADV_EMB_NAME,
    ADV_TRAIN_METHOD,
    BATCH_SIZE,
    USE_SPAN_TYPE_SEMANTIC_FUSION_INNOVATION,
    USE_MULTISCALE_CALIBRATION_THRESHOLD_INNOVATION,
    EFFECTIVE_USE_SPAN_TYPE_SEMANTIC_FUSION,
    EFFECTIVE_USE_MULTISCALE_CALIBRATION_THRESHOLD,
    BEST_CLASS_METRICS_CSV,
    BEST_MODEL_PATH,
    BEST_MODELS_DIR,
    BEST_MODELS_META,
    BEST_THRESHOLDS_JSON,
    CLASS_METRICS_CSV,
    CUSTOM_LABEL_DESCRIPTIONS,
    DATA_DIR,
    DATASET_NAME,
    DEFAULT_PRED_THRESHOLD,
    DEV_FILE,
    DEVICE,
    EARLY_STOP_PATIENCE,
    EPOCHS,
    FGM_EPSILON,
    GP_EFFICIENT,
    GP_HEAD_SIZE,
    LAST_THRESHOLDS_JSON,
    LEARNING_RATE,
    NEW_LAYER_LR_MULTIPLIER,
    USE_MULTISCALE_FUSION,
    MULTISCALE_LAYERS,
    MULTISCALE_DROPOUT,
    MULTISCALE_AUX_LOSS_WEIGHT,
    MULTISCALE_LR_MULTIPLIER,
    MULTISCALE_FUSION_BETA_GRID,
    DEFAULT_MULTISCALE_FUSION_BETA,
    MULTISCALE_DETACH_ENCODER,
    MULTISCALE_USE_IN_ADVERSARIAL,
    USE_GATED_MULTISCALE_MAIN_FUSION,
    GATED_MULTISCALE_DETACH_ENCODER,
    GATED_MULTISCALE_INIT_ALPHA,
    GATED_MULTISCALE_MAX_ALPHA,
    GATED_MULTISCALE_DROPOUT,
    GATED_MULTISCALE_LR_MULTIPLIER,
    USE_MULTISCALE_NEG_CALIBRATION,
    MULTISCALE_NEG_CALIB_WEIGHT,
    MULTISCALE_NEG_TOPK,
    MULTISCALE_NEG_MARGIN,
    NUM_WORKERS,
    PGD_ALPHA,
    PGD_EPSILON,
    PGD_STEPS,
    PRETRAINED_MODEL_DIR,
    RNN_HIDDEN_SIZE,
    RNN_LAYERS,
    SPAN_TYPE_DROPOUT,
    SPAN_TYPE_FUSION_CHUNK_SIZE,
    SPAN_TYPE_FUSION_GAMMA_GRID,
    SPAN_TYPE_HARD_NEG_TOPK,
    SPAN_TYPE_LOSS_WEIGHT,
    SPAN_TYPE_LR_MULTIPLIER,
    SPAN_TYPE_MAX_NEG_PER_BATCH,
    SPAN_TYPE_MAX_POS_PER_BATCH,
    SPAN_TYPE_MAX_WIDTH,
    SPAN_TYPE_MLP_HIDDEN_SIZE,
    SPAN_TYPE_NEG_LOSS_WEIGHT,
    SPAN_TYPE_NEG_MARGIN,
    SPAN_TYPE_PROJ_DIM,
    SPAN_TYPE_TEMPERATURE,
    SPAN_TYPE_USE_IN_ADVERSARIAL,
    SPAN_TYPE_WIDTH_EMB_SIZE,
    THRESHOLD_SEARCH_MAX,
    THRESHOLD_SEARCH_MIN,
    THRESHOLD_SEARCH_MODE,
    THRESHOLD_SEARCH_STEP,
    TRAIN_FILE,
    TRAIN_LOG_CSV,
    TRAIN_PARAMS_JSON,
    TRAIN_STEP_LOG_CSV,
    USE_BIGRU,
    USE_EARLY_STOPPING,
    USE_SPAN_TYPE_CONTRASTIVE,
    USE_SPAN_TYPE_FUSION,
    USE_THRESHOLD_SEARCH,
    WARMUP_RATIO,
    WEIGHT_DECAY,
    ENABLE_COMPUTE_COST_PROFILING,
    TRAIN_COMPUTE_COST_CSV,
    TRAIN_COMPUTE_COST_JSON,
)

train_cme_path = f'{DATA_DIR}/{TRAIN_FILE}'
eval_cme_path = f'{DATA_DIR}/{DEV_FILE}'
device = DEVICE
ENT_CLS_NUM = len(ent2id)



def count_model_parameters(model: torch.nn.Module) -> Dict[str, float]:
    """统计模型总参数量、可训练参数量及参数张量存储量。"""
    total_parameters = sum(p.numel() for p in model.parameters())
    trainable_parameters = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    parameter_storage_bytes = sum(
        p.numel() * p.element_size() for p in model.parameters()
    )
    return {
        'total_parameters': int(total_parameters),
        'trainable_parameters': int(trainable_parameters),
        'frozen_parameters': int(total_parameters - trainable_parameters),
        'parameter_storage_bytes': int(parameter_storage_bytes),
        'parameter_storage_mib': float(
            parameter_storage_bytes / (1024 ** 2)
        ),
    }


def get_runtime_environment() -> Dict:
    """记录计算代价测量所使用的软硬件环境。"""
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
    """CUDA 异步执行，正式计时前后同步以保证墙钟时间准确。"""
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)


def reset_cuda_peak_memory() -> None:
    """重置当前轮次的 CUDA 峰值显存统计。"""
    if torch.cuda.is_available():
        synchronize_cuda()
        torch.cuda.reset_peak_memory_stats(device)


def get_cuda_peak_memory() -> Dict[str, float]:
    """读取 PyTorch allocated 与 reserved 峰值显存。"""
    if not torch.cuda.is_available():
        return {
            'peak_allocated_bytes': None,
            'peak_allocated_gib': None,
            'peak_reserved_bytes': None,
            'peak_reserved_gib': None,
        }

    synchronize_cuda()
    peak_allocated_bytes = int(torch.cuda.max_memory_allocated(device))
    peak_reserved_bytes = int(torch.cuda.max_memory_reserved(device))
    return {
        'peak_allocated_bytes': peak_allocated_bytes,
        'peak_allocated_gib': float(
            peak_allocated_bytes / (1024 ** 3)
        ),
        'peak_reserved_bytes': peak_reserved_bytes,
        'peak_reserved_gib': float(
            peak_reserved_bytes / (1024 ** 3)
        ),
    }


def append_training_cost_csv(csv_path: str, record: Dict) -> None:
    """逐轮追加计算代价，避免程序中断后丢失已完成轮次。"""
    parent = os.path.dirname(csv_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    fields = [
        'epoch',
        'train_phase_seconds',
        'full_epoch_wall_seconds',
        'train_samples',
        'train_batches',
        'training_throughput_samples_per_second',
        'peak_allocated_bytes',
        'peak_allocated_gib',
        'peak_reserved_bytes',
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
    平均单轮训练时间只统计原训练阶段：包含数据加载、前向与反向、
    当前配置下的辅助损失和对抗训练、梯度裁剪、优化器及调度器更新；
    不包含验证、gamma/beta/阈值搜索、模型保存和日志写盘。
    """
    epoch_times = [
        float(item['train_phase_seconds']) for item in epoch_records
    ]
    throughputs = [
        float(item['training_throughput_samples_per_second'])
        for item in epoch_records
    ]
    allocated_values = [
        int(item['peak_allocated_bytes'])
        for item in epoch_records
        if item.get('peak_allocated_bytes') is not None
    ]
    reserved_values = [
        int(item['peak_reserved_bytes'])
        for item in epoch_records
        if item.get('peak_reserved_bytes') is not None
    ]
    peak_allocated_bytes = (
        max(allocated_values) if allocated_values else None
    )
    peak_reserved_bytes = (
        max(reserved_values) if reserved_values else None
    )

    return {
        'measurement_definition': {
            'average_epoch_training_time': (
                'Mean wall-clock time of the training phase only. Data '
                'loading, forward/backward computation, configured '
                'auxiliary losses and adversarial steps, gradient clipping, '
                'optimizer update and scheduler update are included. '
                'Validation, gamma/beta/threshold search, checkpoint I/O and '
                'logging I/O are excluded.'
            ),
            'training_peak_memory': (
                'Maximum torch.cuda allocated memory during each training '
                'phase after resetting peak statistics at the beginning of '
                'the epoch. The overall value is the maximum across all '
                'completed epochs.'
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
        'training_peak_allocated_bytes': peak_allocated_bytes,
        'training_peak_allocated_gib': (
            float(peak_allocated_bytes / (1024 ** 3))
            if peak_allocated_bytes is not None
            else None
        ),
        'training_peak_reserved_bytes': peak_reserved_bytes,
        'training_peak_reserved_gib': (
            float(peak_reserved_bytes / (1024 ** 3))
            if peak_reserved_bytes is not None
            else None
        ),
        'total_run_wall_seconds': float(total_run_wall_seconds),
        'epoch_records': epoch_records,
    }


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


def multilabel_categorical_crossentropy(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return (neg_loss + pos_loss).mean()


def globalpointer_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    batch_size, ent_type_size = y_pred.shape[:2]
    y_true = y_true.reshape(batch_size * ent_type_size, -1)
    y_pred = y_pred.reshape(batch_size * ent_type_size, -1)
    return multilabel_categorical_crossentropy(y_pred, y_true)


# 保留原函数名，避免外部脚本 import loss_fun 失效。
def loss_fun(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return globalpointer_loss(y_pred, y_true)


def extract_span_logits(outputs) -> torch.Tensor:
    if isinstance(outputs, dict):
        return outputs['span_logits']
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    return outputs


def build_label_descriptions() -> List[str]:
    desc_map = get_label_descriptions()
    if CUSTOM_LABEL_DESCRIPTIONS is not None:
        desc_map.update(CUSTOM_LABEL_DESCRIPTIONS)
    return [desc_map.get(id2ent[i], id2ent[i]) for i in range(ENT_CLS_NUM)]


def _random_select(indices: torch.Tensor, max_num: int) -> torch.Tensor:
    if indices.size(0) <= max_num:
        return indices
    perm = torch.randperm(indices.size(0), device=indices.device)[:max_num]
    return indices[perm]


def _select_hard_negative_spans(
    span_logits: torch.Tensor,
    labels: torch.Tensor,
    max_neg: int,
    topk: int,
) -> torch.Tensor:
    """选择当前模型高分但标签为 0 的 hard negative span。"""
    if max_neg <= 0 or topk <= 0:
        return span_logits.new_zeros((0, 4), dtype=torch.long)

    scores = span_logits.detach().clone()
    scores = scores.masked_fill(labels.bool(), -1e12)
    flat_scores = scores.reshape(-1)
    k = min(int(topk), flat_scores.numel())
    if k <= 0:
        return span_logits.new_zeros((0, 4), dtype=torch.long)

    top_values, top_indices = torch.topk(flat_scores, k=k, largest=True)
    # 排除被 mask 后的无效位置。
    valid = top_values > -1e11
    top_indices = top_indices[valid]
    if top_indices.numel() == 0:
        return span_logits.new_zeros((0, 4), dtype=torch.long)

    bsz, num_cls, seq_len, _ = span_logits.shape
    end = top_indices % seq_len
    tmp = top_indices // seq_len
    start = tmp % seq_len
    tmp = tmp // seq_len
    label = tmp % num_cls
    batch = tmp // num_cls
    neg_indices = torch.stack([batch, label, start, end], dim=1).long()
    neg_indices = _random_select(neg_indices, max_neg)
    return neg_indices


def span_type_semantic_loss(
    model: GlobalPointer,
    outputs,
    labels: torch.Tensor,
    use_span_type: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Span-Type 语义对齐损失：
    - 正样本 span：用 CE 让 span 表示靠近正确实体类型语义向量；
    - hard negative span：用 margin 约束其不要靠近任何实体类型语义向量。
    """
    span_logits = extract_span_logits(outputs)
    zero = span_logits.sum() * 0.0
    if not (use_span_type and USE_SPAN_TYPE_CONTRASTIVE and isinstance(outputs, dict)):
        return zero, zero, zero

    sequence_output = outputs['sequence_output']
    pos_indices = labels.nonzero(as_tuple=False)  # (N, 4): batch, label, start, end
    if pos_indices.numel() > 0:
        pos_indices = _random_select(pos_indices, SPAN_TYPE_MAX_POS_PER_BATCH)
        pos_span_repr = model.get_span_representations(sequence_output, pos_indices)
        type_repr = model.get_type_representations()
        pos_logits = torch.matmul(pos_span_repr, type_repr.t()) / SPAN_TYPE_TEMPERATURE
        pos_targets = pos_indices[:, 1].long()
        pos_loss = F.cross_entropy(pos_logits, pos_targets)
    else:
        type_repr = model.get_type_representations()
        pos_loss = zero

    neg_indices = _select_hard_negative_spans(
        span_logits=span_logits,
        labels=labels,
        max_neg=SPAN_TYPE_MAX_NEG_PER_BATCH,
        topk=SPAN_TYPE_HARD_NEG_TOPK,
    )
    if neg_indices.numel() > 0:
        neg_span_repr = model.get_span_representations(sequence_output, neg_indices)
        neg_sim = torch.matmul(neg_span_repr, type_repr.t())
        neg_loss = torch.relu(neg_sim.max(dim=1).values - SPAN_TYPE_NEG_MARGIN).pow(2).mean()
    else:
        neg_loss = zero

    total_span_type_loss = pos_loss + SPAN_TYPE_NEG_LOSS_WEIGHT * neg_loss
    return total_span_type_loss, pos_loss.detach(), neg_loss.detach()


def multiscale_auxiliary_loss(outputs, labels: torch.Tensor, use_multiscale: bool = True) -> torch.Tensor:
    """Multi-Scale auxiliary GlobalPointer loss。标签复用主 GlobalPointer span labels。"""
    span_logits = extract_span_logits(outputs)
    zero = span_logits.sum() * 0.0
    if not (use_multiscale and USE_MULTISCALE_FUSION and isinstance(outputs, dict) and 'multiscale_logits' in outputs):
        return zero
    return globalpointer_loss(outputs['multiscale_logits'], labels)




def multiscale_negative_calibration_loss(outputs, labels: torch.Tensor, use_multiscale: bool = True) -> torch.Tensor:
    """
    实验B：Multi-Scale negative calibration loss。

    只惩罚非实体 span 上 multiscale_logits 相对主分支 span_logits 过高的 top-k 位置：
        penalty = relu(multiscale_logits - detach(span_logits) - margin)^2

    事实动机：实验A的最佳结果主要提升Recall，但Precision下降；因此实验B不改变解码，
    只在训练中压制 multiscale 分支对负 span 的过度加分，目标是减少FP。
    """
    span_logits = extract_span_logits(outputs)
    zero = span_logits.sum() * 0.0
    if not (
        use_multiscale
        and USE_MULTISCALE_FUSION
        and USE_MULTISCALE_NEG_CALIBRATION
        and isinstance(outputs, dict)
        and 'multiscale_logits' in outputs
        and MULTISCALE_NEG_CALIB_WEIGHT > 0
        and MULTISCALE_NEG_TOPK > 0
    ):
        return zero

    multiscale_logits = outputs['multiscale_logits']
    main_logits = span_logits.detach()
    labels_bool = labels.detach().bool()

    # GlobalPointer 已对 padding 和下三角 span 做 mask；这里额外过滤非有限值，避免 -inf - -inf 产生 NaN。
    valid_mask = (~labels_bool) & torch.isfinite(multiscale_logits) & torch.isfinite(main_logits)
    if not bool(valid_mask.any().item()):
        return zero

    gap = multiscale_logits - main_logits - float(MULTISCALE_NEG_MARGIN)
    gap_values = gap.masked_select(valid_mask)
    if gap_values.numel() == 0:
        return zero

    k = min(int(MULTISCALE_NEG_TOPK), int(gap_values.numel()))
    hard_gap = torch.topk(gap_values, k=k, largest=True).values
    penalty = torch.relu(hard_gap).pow(2)
    if not bool((penalty > 0).any().item()):
        return zero
    return penalty.mean()


def compute_total_loss(
    model: GlobalPointer,
    outputs,
    labels: torch.Tensor,
    use_span_type: bool = True,
    use_multiscale: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    span_logits = extract_span_logits(outputs)
    gp_loss = globalpointer_loss(span_logits, labels)
    st_loss, st_pos_loss, st_neg_loss = span_type_semantic_loss(model, outputs, labels, use_span_type=use_span_type)
    ms_loss = multiscale_auxiliary_loss(outputs, labels, use_multiscale=use_multiscale)
    ms_neg_loss = multiscale_negative_calibration_loss(outputs, labels, use_multiscale=use_multiscale)
    total_loss = (
        gp_loss
        + SPAN_TYPE_LOSS_WEIGHT * st_loss
        + MULTISCALE_AUX_LOSS_WEIGHT * ms_loss
        + MULTISCALE_NEG_CALIB_WEIGHT * ms_neg_loss
    )
    return (
        total_loss,
        gp_loss.detach(),
        st_loss.detach(),
        st_pos_loss.detach(),
        st_neg_loss.detach(),
        ms_loss.detach(),
        ms_neg_loss.detach(),
    )


def decode_logits_for_eval(outputs) -> torch.Tensor:
    return extract_span_logits(outputs)


def build_span_type_gamma_grid() -> List[float]:
    """构造 Span-Type fusion gamma 搜索列表，必须包含 0.0 作为安全回退。"""
    if not (USE_SPAN_TYPE_CONTRASTIVE and USE_SPAN_TYPE_FUSION):
        return [0.0]
    values = [float(x) for x in SPAN_TYPE_FUSION_GAMMA_GRID]
    values.append(0.0)
    return sorted(set(round(v, 6) for v in values))


def build_multiscale_beta_grid() -> List[float]:
    """构造 Multi-Scale fusion beta 搜索列表，必须包含 0.0 作为安全回退。"""
    if not USE_MULTISCALE_FUSION:
        return [0.0]
    values = [float(x) for x in MULTISCALE_FUSION_BETA_GRID]
    values.append(0.0)
    values.append(float(DEFAULT_MULTISCALE_FUSION_BETA))
    return sorted(set(round(v, 6) for v in values))


def compute_span_type_fusion_logits(
    model: GlobalPointer,
    outputs,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    计算 Span-Type 相似度 logits。若未启用 fusion 或当前 outputs 不含 sequence_output，返回 None。
    该 logits 只在验证/预测阶段用于 soft fusion，不作为主损失直接训练。
    """
    if not (USE_SPAN_TYPE_CONTRASTIVE and USE_SPAN_TYPE_FUSION and isinstance(outputs, dict)):
        return None
    return model.compute_span_type_similarity_logits(
        outputs['sequence_output'],
        attention_mask,
        chunk_size=SPAN_TYPE_FUSION_CHUNK_SIZE,
    )


def compute_multiscale_fusion_logits(outputs) -> torch.Tensor:
    """返回 Multi-Scale auxiliary logits。若未启用或当前 outputs 不含该分支，返回 None。"""
    if not (USE_MULTISCALE_FUSION and isinstance(outputs, dict)):
        return None
    return outputs.get('multiscale_logits')


def fuse_logits(
    span_logits: torch.Tensor,
    span_type_logits: torch.Tensor,
    gamma: float,
    multiscale_logits: torch.Tensor = None,
    beta: float = 0.0,
) -> torch.Tensor:
    """final_logits = gp_logits + gamma * span_type_logits + beta * multiscale_logits。"""
    logits = span_logits
    if span_type_logits is not None and float(gamma) != 0.0:
        logits = logits + float(gamma) * span_type_logits
    if multiscale_logits is not None and float(beta) != 0.0:
        logits = logits + float(beta) * multiscale_logits
    return logits


def fuse_logits_with_span_type(
    span_logits: torch.Tensor,
    span_type_logits: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """兼容旧函数名。"""
    return fuse_logits(span_logits, span_type_logits, gamma)


def build_threshold_grid() -> List[float]:
    if not USE_THRESHOLD_SEARCH:
        return [float(DEFAULT_PRED_THRESHOLD)]

    values = []
    cur = float(THRESHOLD_SEARCH_MIN)
    max_v = float(THRESHOLD_SEARCH_MAX)
    step = float(THRESHOLD_SEARCH_STEP)
    while cur <= max_v + 1e-9:
        values.append(round(cur, 6))
        cur += step
    if DEFAULT_PRED_THRESHOLD not in values:
        values.append(float(DEFAULT_PRED_THRESHOLD))
    return sorted(set(values))


def init_threshold_stats(num_thresholds: int, num_classes: int) -> Dict[str, torch.Tensor]:
    return {
        'tp': torch.zeros(num_thresholds, num_classes, dtype=torch.long),
        'pred': torch.zeros(num_thresholds, num_classes, dtype=torch.long),
        'true': torch.zeros(num_classes, dtype=torch.long),
    }


def update_threshold_stats(
    stats: Dict[str, torch.Tensor],
    logits: torch.Tensor,
    labels: torch.Tensor,
    thresholds: torch.Tensor,
) -> None:
    logits = logits.detach()
    labels_bool = labels.detach().bool()
    num_classes = logits.size(1)

    for cls_idx in range(num_classes):
        scores_c = logits[:, cls_idx].reshape(-1)
        true_c = labels_bool[:, cls_idx].reshape(-1)
        stats['true'][cls_idx] += true_c.sum().cpu()

        pred_matrix = scores_c.unsqueeze(0) > thresholds.unsqueeze(1)
        pred_counts = pred_matrix.sum(dim=1).cpu()
        tp_counts = (pred_matrix & true_c.unsqueeze(0)).sum(dim=1).cpu()

        stats['pred'][:, cls_idx] += pred_counts
        stats['tp'][:, cls_idx] += tp_counts


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def select_thresholds_and_metrics(
    stats: Dict[str, torch.Tensor],
    threshold_values: List[float],
    mode: str,
) -> Tuple[Dict[str, float], List[Dict], Dict[str, float]]:
    tp = stats['tp']
    pred = stats['pred']
    true = stats['true']
    num_thresholds, num_classes = tp.shape

    mode = mode.lower()
    if mode not in {'global', 'per_class'}:
        raise ValueError("THRESHOLD_SEARCH_MODE 仅支持 'global' 或 'per_class'")

    selected_indices = torch.zeros(num_classes, dtype=torch.long)
    if mode == 'global':
        tp_total = tp.sum(dim=1).float()
        pred_total = pred.sum(dim=1).float()
        true_total = true.sum().float()
        f1_scores = torch.where(
            (pred_total + true_total) > 0,
            2 * tp_total / (pred_total + true_total),
            torch.zeros_like(tp_total),
        )
        best_idx = int(torch.argmax(f1_scores).item())
        selected_indices[:] = best_idx
    else:
        for cls_idx in range(num_classes):
            tp_c = tp[:, cls_idx].float()
            pred_c = pred[:, cls_idx].float()
            true_c = true[cls_idx].float()
            f1_c = torch.where(
                (pred_c + true_c) > 0,
                2 * tp_c / (pred_c + true_c),
                torch.zeros_like(tp_c),
            )
            selected_indices[cls_idx] = int(torch.argmax(f1_c).item())

    class_metrics = []
    total_tp = 0
    total_pred = 0
    total_true = 0
    thresholds_by_label = {}

    for cls_idx in range(num_classes):
        idx = int(selected_indices[cls_idx].item())
        label_name = id2ent.get(cls_idx, str(cls_idx))
        cls_tp = int(tp[idx, cls_idx].item())
        cls_pred = int(pred[idx, cls_idx].item())
        cls_true = int(true[cls_idx].item())
        precision = _safe_div(cls_tp, cls_pred)
        recall = _safe_div(cls_tp, cls_true)
        f1 = _safe_div(2 * cls_tp, cls_pred + cls_true)
        threshold = float(threshold_values[idx])

        thresholds_by_label[label_name] = threshold
        total_tp += cls_tp
        total_pred += cls_pred
        total_true += cls_true

        class_metrics.append({
            'label': label_name,
            'tp': cls_tp,
            'pred': cls_pred,
            'true': cls_true,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'threshold': threshold,
        })

    overall_metrics = {
        'tp': total_tp,
        'pred': total_pred,
        'true': total_true,
        'precision': _safe_div(total_tp, total_pred),
        'recall': _safe_div(total_tp, total_true),
        'f1': _safe_div(2 * total_tp, total_pred + total_true),
    }
    return thresholds_by_label, class_metrics, overall_metrics


def make_threshold_payload(
    epoch: int,
    thresholds_by_label: Dict[str, float],
    overall_metrics: Dict[str, float],
    threshold_values: List[float],
    selected_gamma: float,
    selected_beta: float,
) -> Dict:
    return {
        'epoch': int(epoch),
        'dataset_name': DATASET_NAME,
        'mode': THRESHOLD_SEARCH_MODE,
        'use_span_type_semantic_fusion_innovation': bool(USE_SPAN_TYPE_SEMANTIC_FUSION_INNOVATION),
        'use_multiscale_calibration_threshold_innovation': bool(USE_MULTISCALE_CALIBRATION_THRESHOLD_INNOVATION),
        'effective_use_span_type_semantic_fusion': bool(EFFECTIVE_USE_SPAN_TYPE_SEMANTIC_FUSION),
        'effective_use_multiscale_calibration_threshold': bool(EFFECTIVE_USE_MULTISCALE_CALIBRATION_THRESHOLD),
        'use_span_type_contrastive': bool(USE_SPAN_TYPE_CONTRASTIVE),
        'use_span_type_fusion': bool(USE_SPAN_TYPE_FUSION),
        'selected_span_type_fusion_gamma': float(selected_gamma),
        'selected_multiscale_fusion_beta': float(selected_beta),
        'span_type_fusion_gamma_grid': [float(x) for x in build_span_type_gamma_grid()],
        'multiscale_fusion_beta_grid': [float(x) for x in build_multiscale_beta_grid()],
        'use_multiscale_fusion': bool(USE_MULTISCALE_FUSION),
        'multiscale_layers': [int(x) for x in MULTISCALE_LAYERS],
        'multiscale_aux_loss_weight': float(MULTISCALE_AUX_LOSS_WEIGHT),
        'use_multiscale_neg_calibration': bool(USE_MULTISCALE_NEG_CALIBRATION),
        'multiscale_neg_calib_weight': float(MULTISCALE_NEG_CALIB_WEIGHT),
        'multiscale_neg_topk': int(MULTISCALE_NEG_TOPK),
        'multiscale_neg_margin': float(MULTISCALE_NEG_MARGIN),
        'use_gated_multiscale_main_fusion': bool(USE_GATED_MULTISCALE_MAIN_FUSION),
        'gated_multiscale_detach_encoder': bool(GATED_MULTISCALE_DETACH_ENCODER),
        'gated_multiscale_init_alpha': float(GATED_MULTISCALE_INIT_ALPHA),
        'gated_multiscale_max_alpha': float(GATED_MULTISCALE_MAX_ALPHA),
        'span_type_loss_weight': float(SPAN_TYPE_LOSS_WEIGHT),
        'span_type_temperature': float(SPAN_TYPE_TEMPERATURE),
        'span_type_neg_margin': float(SPAN_TYPE_NEG_MARGIN),
        'default_threshold': float(DEFAULT_PRED_THRESHOLD),
        'search_min': float(THRESHOLD_SEARCH_MIN),
        'search_max': float(THRESHOLD_SEARCH_MAX),
        'search_step': float(THRESHOLD_SEARCH_STEP),
        'searched_thresholds': [float(x) for x in threshold_values],
        'thresholds': {k: float(v) for k, v in thresholds_by_label.items()},
        'overall': {k: float(v) for k, v in overall_metrics.items()},
    }


def main() -> None:
    set_seed(42)

    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_DIR, use_fast=True)

    ner_train = EntDataset(load_data(train_cme_path), tokenizer=tokenizer)
    ner_loader_train = DataLoader(
        ner_train,
        batch_size=BATCH_SIZE,
        collate_fn=ner_train.collate,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    ner_evl = EntDataset(load_data(eval_cme_path), tokenizer=tokenizer)
    ner_loader_evl = DataLoader(
        ner_evl,
        batch_size=BATCH_SIZE,
        collate_fn=ner_evl.collate,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

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

    if USE_SPAN_TYPE_CONTRASTIVE:
        label_descs = build_label_descriptions()
        print('Span-Type label descriptions:')
        for idx, desc in enumerate(label_descs):
            print(f'  {idx}:{id2ent[idx]} -> {desc}')
        model.initialize_type_embeddings(tokenizer, label_descs, device=device)

    parameter_statistics = count_model_parameters(model)
    print('Model parameter statistics:')
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

    pretrained_params = list(model.encoder.parameters())

    task_params = []
    if getattr(model, 'bigru', None) is not None:
        task_params += list(model.bigru.parameters())
    if GP_EFFICIENT:
        task_params += list(model.q_dense.parameters())
        task_params += list(model.k_dense.parameters())
        task_params += list(model.classifier.parameters())
    else:
        task_params += list(model.dense.parameters())

    span_type_params = model.get_span_type_parameters() if USE_SPAN_TYPE_CONTRASTIVE else []
    multiscale_params = model.get_multiscale_parameters() if USE_MULTISCALE_FUSION else []
    gated_multiscale_params = (
        model.get_gated_multiscale_parameters()
        if USE_GATED_MULTISCALE_MAIN_FUSION and hasattr(model, 'get_gated_multiscale_parameters')
        else []
    )

    optimizer_grouped_parameters = [
        {'params': pretrained_params, 'lr': LEARNING_RATE},
        {'params': task_params, 'lr': LEARNING_RATE * NEW_LAYER_LR_MULTIPLIER},
    ]
    if span_type_params:
        optimizer_grouped_parameters.append({
            'params': span_type_params,
            'lr': LEARNING_RATE * SPAN_TYPE_LR_MULTIPLIER,
        })
    if multiscale_params:
        optimizer_grouped_parameters.append({
            'params': multiscale_params,
            'lr': LEARNING_RATE * MULTISCALE_LR_MULTIPLIER,
        })
    if gated_multiscale_params:
        optimizer_grouped_parameters.append({
            'params': gated_multiscale_params,
            'lr': LEARNING_RATE * GATED_MULTISCALE_LR_MULTIPLIER,
        })

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, weight_decay=WEIGHT_DECAY)

    total_training_steps = len(ner_loader_train) * EPOCHS
    warmup_steps = int(WARMUP_RATIO * total_training_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    use_adversarial = ADV_TRAIN_METHOD.upper() in ['FGM', 'PGD']
    adversarial_trainer = None
    if use_adversarial:
        adversarial_trainer = AdversarialTrainer(model, method=ADV_TRAIN_METHOD, emb_name=ADV_EMB_NAME)

    threshold_values = build_threshold_grid()
    threshold_tensor = torch.tensor(threshold_values, dtype=torch.float32, device=device)
    gamma_values = build_span_type_gamma_grid()
    beta_values = build_multiscale_beta_grid()

    print('Innovation switches:')
    print(f'  Span-Type semantic fusion innovation: {USE_SPAN_TYPE_SEMANTIC_FUSION_INNOVATION} '
          f'(effective={EFFECTIVE_USE_SPAN_TYPE_SEMANTIC_FUSION}, gamma_grid={gamma_values})')
    print(f'  Multi-Scale calibration/threshold innovation: {USE_MULTISCALE_CALIBRATION_THRESHOLD_INNOVATION} '
          f'(effective={EFFECTIVE_USE_MULTISCALE_CALIBRATION_THRESHOLD}, beta_grid={beta_values}, '
          f'use_threshold_search={USE_THRESHOLD_SEARCH})')

    max_f = 0.0
    no_improve_epochs = 0
    save_params({
        'dataset_name': DATASET_NAME,
        'pretrained_model_dir': PRETRAINED_MODEL_DIR,
        'train_file': train_cme_path,
        'dev_file': eval_cme_path,
        'batch_size': BATCH_SIZE,
        'ent_cls_num': ENT_CLS_NUM,
        'ent2id': ent2id,
        'gp_head_size': GP_HEAD_SIZE,
        'learning_rate': LEARNING_RATE,
        'new_layer_lr_multiplier': NEW_LAYER_LR_MULTIPLIER,
        'span_type_lr_multiplier': SPAN_TYPE_LR_MULTIPLIER,
        'num_workers': NUM_WORKERS,
        'weight_decay': WEIGHT_DECAY,
        'warmup_ratio': WARMUP_RATIO,
        'rnn_hidden_size': RNN_HIDDEN_SIZE,
        'rnn_layers': RNN_LAYERS,
        'use_bigru': USE_BIGRU,
        'use_span_type_semantic_fusion_innovation': USE_SPAN_TYPE_SEMANTIC_FUSION_INNOVATION,
        'use_multiscale_calibration_threshold_innovation': USE_MULTISCALE_CALIBRATION_THRESHOLD_INNOVATION,
        'effective_use_span_type_semantic_fusion': EFFECTIVE_USE_SPAN_TYPE_SEMANTIC_FUSION,
        'effective_use_multiscale_calibration_threshold': EFFECTIVE_USE_MULTISCALE_CALIBRATION_THRESHOLD,
        'use_span_type_contrastive': USE_SPAN_TYPE_CONTRASTIVE,
        'use_span_type_fusion': USE_SPAN_TYPE_FUSION,
        'span_type_fusion_gamma_grid': gamma_values,
        'use_multiscale_fusion': USE_MULTISCALE_FUSION,
        'multiscale_layers': MULTISCALE_LAYERS,
        'multiscale_dropout': MULTISCALE_DROPOUT,
        'multiscale_aux_loss_weight': MULTISCALE_AUX_LOSS_WEIGHT,
        'multiscale_lr_multiplier': MULTISCALE_LR_MULTIPLIER,
        'multiscale_fusion_beta_grid': beta_values,
        'multiscale_detach_encoder': MULTISCALE_DETACH_ENCODER,
        'multiscale_use_in_adversarial': MULTISCALE_USE_IN_ADVERSARIAL,
        'use_multiscale_neg_calibration': USE_MULTISCALE_NEG_CALIBRATION,
        'multiscale_neg_calib_weight': MULTISCALE_NEG_CALIB_WEIGHT,
        'multiscale_neg_topk': MULTISCALE_NEG_TOPK,
        'multiscale_neg_margin': MULTISCALE_NEG_MARGIN,
        'span_type_fusion_chunk_size': SPAN_TYPE_FUSION_CHUNK_SIZE,
        'span_type_loss_weight': SPAN_TYPE_LOSS_WEIGHT,
        'span_type_neg_loss_weight': SPAN_TYPE_NEG_LOSS_WEIGHT,
        'span_type_temperature': SPAN_TYPE_TEMPERATURE,
        'span_type_proj_dim': SPAN_TYPE_PROJ_DIM,
        'span_type_max_pos_per_batch': SPAN_TYPE_MAX_POS_PER_BATCH,
        'span_type_max_neg_per_batch': SPAN_TYPE_MAX_NEG_PER_BATCH,
        'span_type_hard_neg_topk': SPAN_TYPE_HARD_NEG_TOPK,
        'span_type_neg_margin': SPAN_TYPE_NEG_MARGIN,
        'span_type_use_in_adversarial': SPAN_TYPE_USE_IN_ADVERSARIAL,
        'label_descriptions': build_label_descriptions(),
        'use_threshold_search': USE_THRESHOLD_SEARCH,
        'threshold_search_mode': THRESHOLD_SEARCH_MODE,
        'threshold_search_min': THRESHOLD_SEARCH_MIN,
        'threshold_search_max': THRESHOLD_SEARCH_MAX,
        'threshold_search_step': THRESHOLD_SEARCH_STEP,
        'threshold_values': threshold_values,
        'adv_train_method': ADV_TRAIN_METHOD,
        'adv_emb_name': ADV_EMB_NAME,
        'fgm_epsilon': FGM_EPSILON,
        'pgd_epsilon': PGD_EPSILON,
        'pgd_alpha': PGD_ALPHA,
        'pgd_steps': PGD_STEPS,
        'use_early_stopping': USE_EARLY_STOPPING,
        'early_stop_patience': EARLY_STOP_PATIENCE,
        'use_gated_multiscale_main_fusion': USE_GATED_MULTISCALE_MAIN_FUSION,
        'gated_multiscale_detach_encoder': GATED_MULTISCALE_DETACH_ENCODER,
        'gated_multiscale_init_alpha': GATED_MULTISCALE_INIT_ALPHA,
        'gated_multiscale_max_alpha': GATED_MULTISCALE_MAX_ALPHA,
        'gated_multiscale_dropout': GATED_MULTISCALE_DROPOUT,
        'gated_multiscale_lr_multiplier': GATED_MULTISCALE_LR_MULTIPLIER,
        'compute_cost_profiling_enabled': ENABLE_COMPUTE_COST_PROFILING,
        'train_compute_cost_csv': TRAIN_COMPUTE_COST_CSV,
        'train_compute_cost_json': TRAIN_COMPUTE_COST_JSON,
        'parameter_statistics': parameter_statistics,
    }, TRAIN_PARAMS_JSON)

    epoch_cost_records: List[Dict] = []
    training_run_start = time.perf_counter()

    # 当前运行重新生成逐轮计算代价文件，避免与历史运行混合。
    if (
        ENABLE_COMPUTE_COST_PROFILING
        and os.path.exists(TRAIN_COMPUTE_COST_CSV)
    ):
        os.remove(TRAIN_COMPUTE_COST_CSV)

    for eo in range(EPOCHS):
        model.train()
        full_epoch_start = time.perf_counter()
        if ENABLE_COMPUTE_COST_PROFILING:
            reset_cuda_peak_memory()
        synchronize_cuda()
        train_phase_start = time.perf_counter()

        total_loss_sum = 0.0
        total_gp_loss = 0.0
        total_st_loss = 0.0
        total_st_pos_loss = 0.0
        total_st_neg_loss = 0.0
        total_ms_loss = 0.0
        total_ms_neg_loss = 0.0
        pbar = tqdm(ner_loader_train, desc='Train {}/{}'.format(eo + 1, EPOCHS))

        for idx, batch in enumerate(pbar):
            (
                raw_text_list,
                input_ids,
                attention_mask,
                segment_ids,
                labels,
                start_boundary_labels,
                end_boundary_labels,
            ) = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            segment_ids = segment_ids.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(
                input_ids,
                attention_mask,
                segment_ids,
                return_span_type=USE_SPAN_TYPE_CONTRASTIVE,
                return_multiscale=USE_MULTISCALE_FUSION,
            )
            loss, gp_loss, st_loss, st_pos_loss, st_neg_loss, ms_loss, ms_neg_loss = compute_total_loss(
                model,
                outputs,
                labels,
                use_span_type=True,
                use_multiscale=USE_MULTISCALE_FUSION,
            )
            loss.backward()

            if use_adversarial and adversarial_trainer is not None:
                if ADV_TRAIN_METHOD.upper() == 'FGM':
                    adversarial_trainer.attack(epsilon=FGM_EPSILON)
                    outputs_adv = model(
                        input_ids,
                        attention_mask,
                        segment_ids,
                        return_span_type=(USE_SPAN_TYPE_CONTRASTIVE and SPAN_TYPE_USE_IN_ADVERSARIAL),
                        return_multiscale=(USE_MULTISCALE_FUSION and MULTISCALE_USE_IN_ADVERSARIAL),
                    )
                    loss_adv, _, _, _, _, _, _ = compute_total_loss(
                        model,
                        outputs_adv,
                        labels,
                        use_span_type=SPAN_TYPE_USE_IN_ADVERSARIAL,
                        use_multiscale=MULTISCALE_USE_IN_ADVERSARIAL,
                    )
                    loss_adv.backward()
                    adversarial_trainer.restore()

                elif ADV_TRAIN_METHOD.upper() == 'PGD':
                    grad_backup = {}
                    for name, param in model.named_parameters():
                        if param.requires_grad and param.grad is not None:
                            grad_backup[name] = param.grad.clone()

                    for t in range(PGD_STEPS):
                        adversarial_trainer.attack(
                            epsilon=PGD_EPSILON,
                            alpha=PGD_ALPHA,
                            is_first_attack=(t == 0),
                        )
                        if t < PGD_STEPS - 1:
                            outputs_adv = model(
                                input_ids,
                                attention_mask,
                                segment_ids,
                                return_span_type=(USE_SPAN_TYPE_CONTRASTIVE and SPAN_TYPE_USE_IN_ADVERSARIAL),
                                return_multiscale=(USE_MULTISCALE_FUSION and MULTISCALE_USE_IN_ADVERSARIAL),
                            )
                            loss_adv_temp, _, _, _, _, _, _ = compute_total_loss(
                                model,
                                outputs_adv,
                                labels,
                                use_span_type=SPAN_TYPE_USE_IN_ADVERSARIAL,
                                use_multiscale=MULTISCALE_USE_IN_ADVERSARIAL,
                            )
                            model.zero_grad()
                            loss_adv_temp.backward()

                    outputs_adv = model(
                        input_ids,
                        attention_mask,
                        segment_ids,
                        return_span_type=(USE_SPAN_TYPE_CONTRASTIVE and SPAN_TYPE_USE_IN_ADVERSARIAL),
                        return_multiscale=(USE_MULTISCALE_FUSION and MULTISCALE_USE_IN_ADVERSARIAL),
                    )
                    loss_adv, _, _, _, _, _, _ = compute_total_loss(
                        model,
                        outputs_adv,
                        labels,
                        use_span_type=SPAN_TYPE_USE_IN_ADVERSARIAL,
                        use_multiscale=MULTISCALE_USE_IN_ADVERSARIAL,
                    )
                    model.zero_grad()
                    loss_adv.backward()

                    for name, param in model.named_parameters():
                        if param.requires_grad and name in grad_backup:
                            if param.grad is None:
                                param.grad = grad_backup[name].clone()
                            else:
                                param.grad = (param.grad + grad_backup[name]) / 2.0
                    adversarial_trainer.restore()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss_sum += loss.item()
            total_gp_loss += gp_loss.item()
            total_st_loss += st_loss.item()
            total_st_pos_loss += st_pos_loss.item()
            total_st_neg_loss += st_neg_loss.item()
            total_ms_loss += ms_loss.item()
            total_ms_neg_loss += ms_neg_loss.item()

            avg_loss = total_loss_sum / (idx + 1)
            avg_gp_loss = total_gp_loss / (idx + 1)
            avg_st_loss = total_st_loss / (idx + 1)
            avg_st_pos_loss = total_st_pos_loss / (idx + 1)
            avg_st_neg_loss = total_st_neg_loss / (idx + 1)
            avg_ms_loss = total_ms_loss / (idx + 1)
            avg_ms_neg_loss = total_ms_neg_loss / (idx + 1)
            pbar.set_postfix({
                'loss': '{:.6f}'.format(avg_loss),
                'gp': '{:.6f}'.format(avg_gp_loss),
                'st': '{:.4f}'.format(avg_st_loss),
                'pos': '{:.4f}'.format(avg_st_pos_loss),
                'neg': '{:.4f}'.format(avg_st_neg_loss),
                'ms': '{:.4f}'.format(avg_ms_loss),
                'ms_neg': '{:.4f}'.format(avg_ms_neg_loss),
            })
            append_step_log(TRAIN_STEP_LOG_CSV, eo, idx + 1, loss.item(), avg_loss)

        synchronize_cuda()
        train_phase_seconds = time.perf_counter() - train_phase_start
        train_peak_memory = (
            get_cuda_peak_memory()
            if ENABLE_COMPUTE_COST_PROFILING
            else {
                'peak_allocated_bytes': None,
                'peak_allocated_gib': None,
                'peak_reserved_bytes': None,
                'peak_reserved_gib': None,
            }
        )

        with torch.no_grad():
            model.eval()
            stats_grid = [
                [init_threshold_stats(len(threshold_values), ENT_CLS_NUM) for _ in beta_values]
                for _ in gamma_values
            ]

            for batch in tqdm(ner_loader_evl, desc='Valing'):
                (
                    raw_text_list,
                    input_ids,
                    attention_mask,
                    segment_ids,
                    labels,
                    start_boundary_labels,
                    end_boundary_labels,
                ) = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                segment_ids = segment_ids.to(device)
                labels = labels.to(device)

                outputs = model(
                    input_ids,
                    attention_mask,
                    segment_ids,
                    return_span_type=(USE_SPAN_TYPE_CONTRASTIVE and USE_SPAN_TYPE_FUSION),
                    return_multiscale=USE_MULTISCALE_FUSION,
                )
                span_logits = decode_logits_for_eval(outputs)
                span_type_logits = compute_span_type_fusion_logits(model, outputs, attention_mask)
                multiscale_logits = compute_multiscale_fusion_logits(outputs)

                for gamma_idx, gamma in enumerate(gamma_values):
                    for beta_idx, beta in enumerate(beta_values):
                        logits = fuse_logits(span_logits, span_type_logits, gamma, multiscale_logits, beta)
                        update_threshold_stats(stats_grid[gamma_idx][beta_idx], logits, labels, threshold_tensor)

            best_pack = None
            for gamma_idx, gamma in enumerate(gamma_values):
                for beta_idx, beta in enumerate(beta_values):
                    cur_thresholds, cur_class_metrics, cur_overall_metrics = select_thresholds_and_metrics(
                        stats_grid[gamma_idx][beta_idx],
                        threshold_values,
                        THRESHOLD_SEARCH_MODE,
                    )
                    cur_f1 = cur_overall_metrics['f1']
                    if best_pack is None or cur_f1 > best_pack['overall']['f1']:
                        best_pack = {
                            'gamma': float(gamma),
                            'beta': float(beta),
                            'thresholds': cur_thresholds,
                            'class_metrics': cur_class_metrics,
                            'overall': cur_overall_metrics,
                        }

            selected_gamma = best_pack['gamma']
            selected_beta = best_pack['beta']
            thresholds_by_label = best_pack['thresholds']
            class_metrics = best_pack['class_metrics']
            overall_metrics = best_pack['overall']

            f1 = overall_metrics['f1']
            precision = overall_metrics['precision']
            recall = overall_metrics['recall']
            avg_epoch_loss = total_loss_sum / max(1, len(ner_loader_train))
            append_epoch_log(TRAIN_LOG_CSV, eo, avg_epoch_loss, f1, precision, recall)
            append_class_metrics(CLASS_METRICS_CSV, eo, class_metrics)

            payload = make_threshold_payload(eo, thresholds_by_label, overall_metrics, threshold_values, selected_gamma, selected_beta)
            save_thresholds_json(LAST_THRESHOLDS_JSON, payload)

            print(f"\nEpoch {eo} class metrics: selected_gamma={selected_gamma:.3f}, selected_beta={selected_beta:.3f}")
            print(f"{'label':<8}{'TP':>8}{'Pred':>8}{'True':>8}{'Precision':>12}{'Recall':>12}{'F1':>12}{'Threshold':>12}")
            for item in class_metrics:
                print(
                    f"{item['label']:<8}{item['tp']:>8}{item['pred']:>8}{item['true']:>8}"
                    f"{item['precision']:>12.6f}{item['recall']:>12.6f}{item['f1']:>12.6f}{item['threshold']:>12.3f}"
                )
            print(
                f"Epoch: {eo}, loss: {avg_epoch_loss:.6f}, gamma: {selected_gamma:.3f}, beta: {selected_beta:.3f}, "
                f"precision: {precision:.6f}, recall: {recall:.6f}, f1: {f1:.6f}"
            )

            if f1 > max_f:
                max_f = f1
                no_improve_epochs = 0
                save_best_model(model, BEST_MODEL_PATH)
                save_class_metrics(BEST_CLASS_METRICS_CSV, eo, class_metrics)
                save_thresholds_json(BEST_THRESHOLDS_JSON, payload)
                save_topk_model(model, f1, eo, BEST_MODELS_DIR, BEST_MODELS_META, k=5)
                print(f"Best model updated: f1={max_f:.6f}, saved to {BEST_MODEL_PATH}")
            else:
                no_improve_epochs += 1
                print(f"No improvement for {no_improve_epochs} epoch(s). Best f1={max_f:.6f}")

            if ENABLE_COMPUTE_COST_PROFILING:
                full_epoch_wall_seconds = (
                    time.perf_counter() - full_epoch_start
                )
                train_samples = int(len(ner_train))
                training_throughput = (
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
                    'training_throughput_samples_per_second': (
                        training_throughput
                    ),
                    'peak_allocated_bytes': train_peak_memory.get(
                        'peak_allocated_bytes'
                    ),
                    'peak_allocated_gib': train_peak_memory.get(
                        'peak_allocated_gib'
                    ),
                    'peak_reserved_bytes': train_peak_memory.get(
                        'peak_reserved_bytes'
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

            if USE_EARLY_STOPPING and no_improve_epochs >= EARLY_STOP_PATIENCE:
                print(f"Early stopping triggered at epoch {eo}.")
                break

    if ENABLE_COMPUTE_COST_PROFILING:
        final_compute_cost = build_training_cost_summary(
            parameter_statistics,
            epoch_cost_records,
            time.perf_counter() - training_run_start,
        )
        save_params(final_compute_cost, TRAIN_COMPUTE_COST_JSON)
        print(
            'Training compute-cost summary saved to: '
            f'{TRAIN_COMPUTE_COST_JSON}'
        )

    print(f"Training finished. Best F1: {max_f:.6f}")


if __name__ == '__main__':
    main()
