# -*- coding: utf-8 -*-
"""
集中配置文件：预训练模型、数据路径与训练/预测超参
"""
import os
import torch

# 路径配置
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PRETRAINED_MODEL_DIR = os.path.join(PROJECT_DIR, 'pretrained_models', 'chinese_roberta_wwm_large_ext')

DATA_DIR = os.path.join(PROJECT_DIR, 'datasets')
TRAIN_FILE = 'CMeEE-V2_train.json'
DEV_FILE = 'CMeEE-V2_dev.json'
TEST_FILE = 'CMeEE-V2_test.json'

# 输出
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)
SAVE_MODEL_PATH = os.path.join(OUTPUT_DIR, 'ent_model.pth')
# 最优模型与日志/图表
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, 'best_model.pth')
BEST_MODELS_DIR = os.path.join(OUTPUT_DIR, 'best_models')
BEST_MODELS_META = os.path.join(BEST_MODELS_DIR, 'best_models.json')
os.makedirs(BEST_MODELS_DIR, exist_ok=True)
TRAIN_LOG_CSV = os.path.join(OUTPUT_DIR, 'train_log.csv')
TRAIN_STEP_LOG_CSV = os.path.join(OUTPUT_DIR, 'train_step_log.csv')
TRAIN_PARAMS_JSON = os.path.join(OUTPUT_DIR, 'train_params.json')
# 兼容旧版总图路径（仍保留，但默认不再使用）
METRICS_FIG_PATH = os.path.join(OUTPUT_DIR, 'metrics_curve.png')
# 分图输出路径
METRICS_F1_FIG_PATH = os.path.join(OUTPUT_DIR, 'f1_curve.png')
METRICS_PRECISION_FIG_PATH = os.path.join(OUTPUT_DIR, 'precision_curve.png')
METRICS_RECALL_FIG_PATH = os.path.join(OUTPUT_DIR, 'recall_curve.png')
METRICS_LOSS_FIG_PATH = os.path.join(OUTPUT_DIR, 'loss_curve.png')

# =========================
# 仅用于结果记录的输出路径
# =========================
# 验证集逐类别指标：每轮明细、最佳轮次CSV、最佳轮次JSON
CLASS_METRICS_CSV = os.path.join(OUTPUT_DIR, 'class_metrics.csv')
BEST_CLASS_METRICS_CSV = os.path.join(OUTPUT_DIR, 'best_class_metrics.csv')
BEST_CLASS_METRICS_JSON = os.path.join(OUTPUT_DIR, 'best_class_metrics.json')

# 计算代价：训练逐轮明细、训练汇总、推断汇总
ENABLE_COMPUTE_COST_PROFILING = True
TRAIN_COMPUTE_COST_CSV = os.path.join(OUTPUT_DIR, 'train_compute_cost.csv')
TRAIN_COMPUTE_COST_JSON = os.path.join(OUTPUT_DIR, 'train_compute_cost.json')
INFERENCE_COMPUTE_COST_JSON = os.path.join(OUTPUT_DIR, 'inference_compute_cost.json')

# 正式推断计时前的预热样本数；预热不计入吞吐量和单样本延迟。
INFERENCE_WARMUP_SAMPLES = 10

# 设备
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 数据与模型超参
MAX_LEN = 256
BATCH_SIZE = 16
NUM_WORKERS = 4  # Windows 上多进程建议适度降低
ENT_CLS_NUM = 9
GP_HEAD_SIZE = 64
GP_EFFICIENT = False  # 是否启用高效版 GlobalPointer
LEARNING_RATE = 1e-5
EPOCHS = 10
EARLY_STOP_PATIENCE = 2
# 早停开关：False 时禁用早停，训练将固定跑满 EPOCHS
USE_EARLY_STOPPING = False


# 优化器与调度器超参
# AdamW 的权重衰减，用于正则化抑制过拟合
WEIGHT_DECAY = 0.01
# 预热比例：训练前 warmup 占总步数的比例（例如 0.1 表示 10% 步数做 warmup）
WARMUP_RATIO = 0.1

# RNN 层超参
# RNN 隐藏层大小（单向），双向 RNN 的输出维度将是 RNN_HIDDEN_SIZE * 2
RNN_HIDDEN_SIZE = 256
# RNN 层数（BiLSTM 和 BiGRU 各使用此层数）
RNN_LAYERS = 1

# GlobalPointer 中是否启用 BiGRU（用于消融实验）
USE_BIGRU = False

# 对抗训练配置
# 方法：'NONE'（不使用）、'FGM' 或 'PGD'
ADV_TRAIN_METHOD = 'NONE'
# Embedding 层参数名（HuggingFace Bert/RoBERTa 默认如下）
ADV_EMB_NAME = 'encoder.embeddings.word_embeddings'
# FGM 参数
FGM_EPSILON = 1.0
# PGD 参数
PGD_EPSILON = 0.3
PGD_ALPHA = 0.1
PGD_STEPS = 3

