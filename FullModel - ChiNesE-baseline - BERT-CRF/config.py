# -*- coding: utf-8 -*-
"""
集中配置文件：ChiNesE 数据集 BERT-CRF 基线版本

说明：
- 本版本用于构建 BERT-CRF flat NER baseline；
- ChiNesE 是嵌套 NER 数据集，而普通 CRF 序列标注一次只能给每个 token 一个标签；
- 因此训练时需要把嵌套实体转换为单层 BIO 标签，本代码默认采用 innermost 策略；
- 验证和测试时仍然使用 ChiNesE 的完整 gold entities 计算 P/R/F1，用于体现 flat 序列标注在嵌套场景下的局限性。
"""
import os
import torch

# =========================
# 路径配置
# =========================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# BERT-CRF 基线：使用 bert-base-chinese。
# 请保证本地目录存在：/root/autodl-tmp/nlp/pretrained_models/bert-base-chinese
PRETRAINED_MODEL_DIR = os.path.join(PROJECT_DIR, 'pretrained_models', 'bert-base-chinese')

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
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'outputs_ChiNesE_BERT_CRF')
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAVE_MODEL_PATH = os.path.join(OUTPUT_DIR, 'ent_model.pth')
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, 'best_model.pth')
BEST_MODELS_DIR = os.path.join(OUTPUT_DIR, 'best_models')
BEST_MODELS_META = os.path.join(BEST_MODELS_DIR, 'best_models.json')
os.makedirs(BEST_MODELS_DIR, exist_ok=True)

TRAIN_LOG_CSV = os.path.join(OUTPUT_DIR, 'train_log.csv')
TRAIN_STEP_LOG_CSV = os.path.join(OUTPUT_DIR, 'train_step_log.csv')
TRAIN_PARAMS_JSON = os.path.join(OUTPUT_DIR, 'train_params.json')
CLASS_METRICS_CSV = os.path.join(OUTPUT_DIR, 'class_metrics.csv')
BEST_CLASS_METRICS_CSV = os.path.join(OUTPUT_DIR, 'best_class_metrics.csv')
TEST_CLASS_METRICS_CSV = os.path.join(OUTPUT_DIR, 'test_class_metrics.csv')
TEST_METRICS_JSON = os.path.join(OUTPUT_DIR, 'test_metrics.json')

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
# 数据与训练超参
# =========================
MAX_LEN = 256
BATCH_SIZE = 16
NUM_WORKERS = 4

# ChiNesE 10 类实体：Person / Location / Organization / Time / Work / Food / Product / Medicine / Event / Creature
ENT_CLS_NUM = 10

# BERT-CRF 是序列标注模型：O + 10类 * BIO = 21 个序列标签
NUM_CRF_LABELS = 1 + ENT_CLS_NUM * 2

LEARNING_RATE = 1e-5
CRF_LEARNING_RATE = 2e-4
EPOCHS = 10
EARLY_STOP_PATIENCE = 2
USE_EARLY_STOPPING = False
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
DROPOUT_PROB = 0.1

# 嵌套实体转 flat BIO 的策略：
# innermost：优先保留更短的内层实体；outermost：优先保留更长的外层实体。
# ChiNesE 原论文中也使用 innermost / outermost 作为序列标注类对比思路。
FLAT_NER_STRATEGY = 'innermost'

# 随机种子
SEED = 42
