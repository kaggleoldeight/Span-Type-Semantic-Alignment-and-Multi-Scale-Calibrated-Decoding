# -*- coding: utf-8 -*-
"""
集中配置文件：ChiNesE 数据集版本

本版本基于当前旧版 GlobalPointer 训练代码生成：
- 数据集切换为 ChiNesE；
- 实体类别数改为 10；
- 输出目录改为 outputs_ChiNesE，避免覆盖 CMeEE-V2 / CLUENER 结果；
- 保留当前代码的 GlobalPointer、BiGRU 开关、对抗训练开关和逐类别指标输出；
- ChiNesE 的标注格式为 label -> mention -> [[start, end], ...]，span 为字符级闭区间。
"""
import os
import torch

# =========================
# 路径配置
# =========================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PRETRAINED_MODEL_DIR = os.path.join(PROJECT_DIR, 'pretrained_models', 'chinese_roberta_wwm_large_ext')

DATA_DIR = os.path.join(PROJECT_DIR, 'datasets')


def _pick_existing_file(candidates):
    """按候选文件名自动选择存在的文件；若都不存在，返回第一个候选名，便于报错定位。"""
    for filename in candidates:
        if os.path.exists(os.path.join(DATA_DIR, filename)):
            return filename
    return candidates[0]


# ChiNesE / Mulco 项目常见命名可能是 new_train/new_eval/new_test 或 *.json。
DATASET_NAME = 'ChiNesE'
TRAIN_FILE = _pick_existing_file([
    'new_train.json', 'new_train',
    'ChiNesE_train.json', 'chinese_train.json',
    'train.json', 'train.jsonl',
])
DEV_FILE = _pick_existing_file([
    'new_eval.json', 'new_eval',
    'new_dev.json', 'new_valid.json',
    'ChiNesE_dev.json', 'ChiNesE_eval.json',
    'dev.json', 'valid.json', 'eval.json', 'dev.jsonl',
])
TEST_FILE = _pick_existing_file([
    'new_test.json', 'new_test',
    'ChiNesE_test.json', 'chinese_test.json',
    'test.json', 'test.jsonl',
])

# =========================
# 输出路径
# =========================
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'outputs_ChiNesE')
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAVE_MODEL_PATH = os.path.join(OUTPUT_DIR, 'ent_model.pth')
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, 'best_model.pth')
BEST_MODELS_DIR = os.path.join(OUTPUT_DIR, 'best_models')
BEST_MODELS_META = os.path.join(BEST_MODELS_DIR, 'best_models.json')
os.makedirs(BEST_MODELS_DIR, exist_ok=True)

TRAIN_LOG_CSV = os.path.join(OUTPUT_DIR, 'train_log.csv')
TRAIN_STEP_LOG_CSV = os.path.join(OUTPUT_DIR, 'train_step_log.csv')
TRAIN_PARAMS_JSON = os.path.join(OUTPUT_DIR, 'train_params.json')

# 逐类别指标输出：保存每轮各实体类别 TP/Pred/True/Precision/Recall/F1
CLASS_METRICS_CSV = os.path.join(OUTPUT_DIR, 'class_metrics.csv')
BEST_CLASS_METRICS_CSV = os.path.join(OUTPUT_DIR, 'best_class_metrics.csv')

# 有标签 test 集最终评估输出
TEST_CLASS_METRICS_CSV = os.path.join(OUTPUT_DIR, 'test_class_metrics.csv')
TEST_METRICS_JSON = os.path.join(OUTPUT_DIR, 'test_metrics.json')

# 绘图输出路径
METRICS_FIG_PATH = os.path.join(OUTPUT_DIR, 'metrics_curve.png')
METRICS_F1_FIG_PATH = os.path.join(OUTPUT_DIR, 'f1_curve.png')
METRICS_PRECISION_FIG_PATH = os.path.join(OUTPUT_DIR, 'precision_curve.png')
METRICS_RECALL_FIG_PATH = os.path.join(OUTPUT_DIR, 'recall_curve.png')
METRICS_LOSS_FIG_PATH = os.path.join(OUTPUT_DIR, 'loss_curve.png')

# =========================
# 设备
# =========================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# =========================
# 数据与模型超参
# =========================
MAX_LEN = 256
BATCH_SIZE = 16
NUM_WORKERS = 4

# ChiNesE 实体类别数为 10：
# Person / Location / Organization / Time / Work / Food / Product / Medicine / Event / Creature
ENT_CLS_NUM = 10
GP_HEAD_SIZE = 64
GP_EFFICIENT = False
LEARNING_RATE = 1e-5
EPOCHS = 10
EARLY_STOP_PATIENCE = 2
USE_EARLY_STOPPING = False

# 优化器与调度器超参
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1

# RNN 层超参
RNN_HIDDEN_SIZE = 256
RNN_LAYERS = 1

# GlobalPointer 中是否启用 BiGRU（用于消融实验）
# 保留你当前上传代码配置：默认关闭 BiGRU。
USE_BIGRU = False

# 对抗训练配置
# 保留你当前上传代码配置：默认不使用对抗训练。
ADV_TRAIN_METHOD = 'NONE'
ADV_EMB_NAME = 'encoder.embeddings.word_embeddings'
FGM_EPSILON = 1.0
PGD_EPSILON = 0.3
PGD_ALPHA = 0.1
PGD_STEPS = 3
