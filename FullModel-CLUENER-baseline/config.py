# -*- coding: utf-8 -*-
"""
集中配置文件：CLUENER 数据集版本

本版本基于当前旧版 GlobalPointer 训练代码生成：
- 数据集切换为 CLUENER；
- 实体类别数改为 10；
- 输出目录改为 outputs_CLUENER，避免覆盖 CMeEE-V2 结果；
- 保留逐类别指标输出：class_metrics.csv / best_class_metrics.csv。
"""
import os
import torch

# 路径配置
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PRETRAINED_MODEL_DIR = os.path.join(PROJECT_DIR, 'pretrained_models', 'chinese_roberta_wwm_large_ext')

DATA_DIR = os.path.join(PROJECT_DIR, 'datasets')

# CLUENER 官方常见文件为 JSONL：train.json / dev.json / test.json。
# 如果你使用 CLUENER_train.json 等命名，也会自动优先识别。
def _pick_existing_file(candidates):
    for filename in candidates:
        if os.path.exists(os.path.join(DATA_DIR, filename)):
            return filename
    return candidates[0]

DATASET_NAME = 'CLUENER'
TRAIN_FILE = _pick_existing_file(['CLUENER_train.json', 'cluener_train.json', 'train.json', 'train.jsonl'])
DEV_FILE = _pick_existing_file(['CLUENER_dev.json', 'cluener_dev.json', 'dev.json', 'dev.jsonl'])
TEST_FILE = _pick_existing_file(['CLUENER_test.json', 'cluener_test.json', 'test.json', 'test.jsonl'])

# 输出
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'outputs_CLUENER')
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
# 兼容旧版总图路径（仍保留，但默认不再使用）
METRICS_FIG_PATH = os.path.join(OUTPUT_DIR, 'metrics_curve.png')
# 分图输出路径
METRICS_F1_FIG_PATH = os.path.join(OUTPUT_DIR, 'f1_curve.png')
METRICS_PRECISION_FIG_PATH = os.path.join(OUTPUT_DIR, 'precision_curve.png')
METRICS_RECALL_FIG_PATH = os.path.join(OUTPUT_DIR, 'recall_curve.png')
METRICS_LOSS_FIG_PATH = os.path.join(OUTPUT_DIR, 'loss_curve.png')

# 设备
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 数据与模型超参
MAX_LEN = 256
BATCH_SIZE = 16
NUM_WORKERS = 4

# CLUENER 实体类别数为 10：address/book/company/game/government/movie/name/organization/position/scene
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
# 保留你当前代码配置：默认关闭 BiGRU。
USE_BIGRU = False

# 对抗训练配置
# 保留你当前代码配置：默认不使用对抗训练。
ADV_TRAIN_METHOD = 'NONE'
ADV_EMB_NAME = 'encoder.embeddings.word_embeddings'
FGM_EPSILON = 1.0
PGD_EPSILON = 0.3
PGD_ALPHA = 0.1
PGD_STEPS = 3
