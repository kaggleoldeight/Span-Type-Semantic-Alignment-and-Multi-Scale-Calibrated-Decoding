# -*- coding: utf-8 -*-
"""
集中配置文件：预训练模型、数据路径与训练/预测超参

Span-Type Semantic Alignment GlobalPointer 版本：
- 保留 RoBERTa/BiGRU/GlobalPointer/PGD 主链路；
- 默认关闭 Boundary-Aware，避免与新实验变量混淆；
- 新增 Span-Type 语义对齐辅助损失：真实 span 向对应实体类型语义向量靠近，
  高分负 span 远离所有实体类型语义向量；
- 保留验证集 per-class 阈值搜索与逐类别指标保存。
"""
import os
import torch

# =========================
# 路径配置
# =========================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PRETRAINED_MODEL_DIR = os.path.join(PROJECT_DIR, 'pretrained_models', 'chinese_roberta_wwm_large_ext')

DATA_DIR = os.path.join(PROJECT_DIR, 'datasets')

# =========================
# CLUENER 数据文件配置
# =========================
# 事实依据：
# - 你当前 data_loader.py 已支持 CLUENER 常见 JSONL 格式：
#   {'text': str, 'label': {type: {mention: [[start, end], ...]}}}
# - CLUENER 常见标注 span 为闭区间，因此 END_INDEX_IS_EXCLUSIVE=False。
#
# 为避免不同数据集文件命名不一致，这里会自动选择已存在的文件：
# 优先 CLUENER_train.json / CLUENER_dev.json / CLUENER_test.json；
# 若不存在，则回退到官方常见 train.json / dev.json / test.json。
def _pick_existing_file(candidates):
    for filename in candidates:
        if os.path.exists(os.path.join(DATA_DIR, filename)):
            return filename
    return candidates[0]

TRAIN_FILE = _pick_existing_file(['CLUENER_train.json', 'cluener_train.json', 'train.json', 'train.jsonl'])
DEV_FILE = _pick_existing_file(['CLUENER_dev.json', 'cluener_dev.json', 'dev.json', 'dev.jsonl'])
TEST_FILE = _pick_existing_file(['CLUENER_test.json', 'cluener_test.json', 'test.json', 'test.jsonl'])

# 数据集名称：'CMeEE-V2' 或 'CLUENER'
DATASET_NAME = 'CLUENER'

# CMeEE-V2 的 end_idx 为开区间；CLUENER 常见格式为闭区间。
END_INDEX_IS_EXCLUSIVE = False

# 如果你使用自定义数据集，可在这里覆盖标签映射；为 None 时根据 DATASET_NAME 自动选择。
CUSTOM_ENT2ID = None

# =========================
# 输出路径
# =========================
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

CLASS_METRICS_CSV = os.path.join(OUTPUT_DIR, 'class_metrics.csv')
BEST_CLASS_METRICS_CSV = os.path.join(OUTPUT_DIR, 'best_class_metrics.csv')
BEST_THRESHOLDS_JSON = os.path.join(OUTPUT_DIR, 'best_thresholds.json')
LAST_THRESHOLDS_JSON = os.path.join(OUTPUT_DIR, 'last_thresholds.json')

METRICS_FIG_PATH = os.path.join(OUTPUT_DIR, 'metrics_curve.png')
METRICS_F1_FIG_PATH = os.path.join(OUTPUT_DIR, 'f1_curve.png')
METRICS_PRECISION_FIG_PATH = os.path.join(OUTPUT_DIR, 'precision_curve.png')
METRICS_RECALL_FIG_PATH = os.path.join(OUTPUT_DIR, 'recall_curve.png')
METRICS_LOSS_FIG_PATH = os.path.join(OUTPUT_DIR, 'loss_curve.png')

# =========================
# 设备与基础超参
# =========================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MAX_LEN = 256
BATCH_SIZE = 16
NUM_WORKERS = 4

# ENT_CLS_NUM 仅作配置记录；训练/预测实际会以 data_loader.ent2id 的长度为准。
ENT_CLS_NUM = 10
GP_HEAD_SIZE = 64
GP_EFFICIENT = False
LEARNING_RATE = 1e-5
EPOCHS = 10
EARLY_STOP_PATIENCE = 2
USE_EARLY_STOPPING = False

WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
NEW_LAYER_LR_MULTIPLIER = 20.0

RNN_HIDDEN_SIZE = 256
RNN_LAYERS = 1
USE_BIGRU = True

# =========================
# 两个论文创新点总开关（用于消融实验）
# =========================
# 创新点1：Span-Type 语义对齐与自适应融合解码机制
# True  = 启用 Span-Type 语义对齐辅助损失 + Span-Type logits 融合 + gamma 搜索；
# False = 完全关闭 Span-Type 相关模块，gamma 固定为 0.0。
USE_SPAN_TYPE_SEMANTIC_FUSION_INNOVATION = True

# 创新点2：多尺度层级特征融合、负样本校准与联合阈值优化机制
# True  = 启用 Multi-Scale 辅助分支 + Gated 主分支融合 + 负样本校准 + beta/类别阈值搜索；
# False = 完全关闭 Multi-Scale/Gated/负样本校准，并关闭 per-class 阈值搜索，beta 固定为 0.0。
USE_MULTISCALE_CALIBRATION_THRESHOLD_INNOVATION = True

# =========================
# Boundary-Aware 兼容配置：默认关闭
# =========================
USE_BOUNDARY_AWARE = False
BOUNDARY_LOSS_WEIGHT_START = 0.0
BOUNDARY_LOSS_WEIGHT_END = 0.0
USE_BOUNDARY_RERANK = False
BOUNDARY_RERANK_ALPHA = 0.0
BOUNDARY_RERANK_BETA = 0.0
BOUNDARY_RERANK_MODE = 'centered_prob'
BOUNDARY_HEAD_LR_MULTIPLIER = 5.0

# =========================
# Span-Type Semantic Alignment 配置
# =========================
USE_SPAN_TYPE_CONTRASTIVE = True
SPAN_TYPE_LOSS_WEIGHT = 0.05
SPAN_TYPE_NEG_LOSS_WEIGHT = 0.20
SPAN_TYPE_TEMPERATURE = 0.07
SPAN_TYPE_PROJ_DIM = 256
SPAN_TYPE_MLP_HIDDEN_SIZE = 256
SPAN_TYPE_DROPOUT = 0.20
SPAN_TYPE_WIDTH_EMB_SIZE = 32
SPAN_TYPE_MAX_WIDTH = 64
SPAN_TYPE_MAX_POS_PER_BATCH = 128
SPAN_TYPE_MAX_NEG_PER_BATCH = 256
SPAN_TYPE_HARD_NEG_TOPK = 256
SPAN_TYPE_NEG_MARGIN = 0.20
SPAN_TYPE_LR_MULTIPLIER = 10.0

# 为控制 PGD 训练成本，默认只在普通前向中加入 Span-Type 损失；
# 如显存与速度允许，可设为 True，让对抗分支也包含该辅助损失。
SPAN_TYPE_USE_IN_ADVERSARIAL = False

# =========================
# Span-Type Semantic Fusion 配置
# =========================
# 事实依据：上一版 Span-Type 只作为辅助损失，预测阶段仍只使用 GlobalPointer logits，
# 因此相对强基线提升较小。本版让 Span-Type 相似度直接参与验证/预测阶段的 span logits，
# 并在验证集搜索融合系数 gamma，gamma=0.0 等价于回退到上一版/强基线解码。
USE_SPAN_TYPE_FUSION = True
SPAN_TYPE_FUSION_GAMMA_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
DEFAULT_SPAN_TYPE_FUSION_GAMMA = 0.0
SPAN_TYPE_FUSION_CHUNK_SIZE = 8192

# =========================
# Layer-wise Multi-Scale Span Fusion 配置
# =========================
# 与此前 CNN adapter 不同，本模块不改写 RoBERTa->BiGRU->GlobalPointer 主干。
# 它从 RoBERTa 多个 hidden layer 取表示，训练一个轻量 auxiliary GlobalPointer 分支，
# 验证/预测阶段用 beta 融合：final_logits = gp_logits + gamma * span_type_logits + beta * multiscale_logits。
# beta=0.0 等价于完全回退到当前 Span-Type Fusion 强基线。
USE_MULTISCALE_FUSION = True
MULTISCALE_LAYERS = [-1, -2, -4, -6]
MULTISCALE_DROPOUT = 0.10
MULTISCALE_AUX_LOSS_WEIGHT = 0.01
MULTISCALE_LR_MULTIPLIER = 5.0
MULTISCALE_FUSION_BETA_GRID = [0.0, 0.03, 0.05, 0.08, 0.10, 0.15]
DEFAULT_MULTISCALE_FUSION_BETA = 0.0
# 为降低负迁移风险，multiscale 辅助损失默认不反向更新 encoder，只训练辅助分支参数。
MULTISCALE_DETACH_ENCODER = True
# 为控制 PGD 成本，默认对抗分支不计算 multiscale 辅助损失。
MULTISCALE_USE_IN_ADVERSARIAL = False


# =========================
# Gated Multi-Scale Main Fusion 配置（实验A）
# =========================
# 实验C：在实验A/B基础上允许 gated residual 主分支融合反向更新 encoder。
# 目的：让多尺度主融合不仅训练门控/投影层，也能调整 RoBERTa 多层表示；风险是可能扰动强主干。
USE_GATED_MULTISCALE_MAIN_FUSION = True
GATED_MULTISCALE_DETACH_ENCODER = False
GATED_MULTISCALE_INIT_ALPHA = 0.0
GATED_MULTISCALE_MAX_ALPHA = 0.10
GATED_MULTISCALE_DROPOUT = 0.10
GATED_MULTISCALE_LR_MULTIPLIER = 5.0


# =========================
# Multi-Scale Negative Calibration 配置（实验B）
# =========================
# 实验C沿用实验B/B2的负样本校准，但使用更强权重0.005。
# 目的：在 encoder 可被 gated fusion 更新时，进一步约束 multiscale_logits 对非实体 span 的过高响应。
USE_MULTISCALE_NEG_CALIBRATION = True
MULTISCALE_NEG_CALIB_WEIGHT = 0.005
MULTISCALE_NEG_TOPK = 256
MULTISCALE_NEG_MARGIN = 0.0

# 类型描述用于初始化实体类型语义向量。
# CMeEE-V2 标签。
CME_LABEL_DESCRIPTIONS = {
    'bod': '身体部位，解剖结构或人体组织器官',
    'dis': '疾病名称，诊断名称或病理状态',
    'sym': '症状和体征，患者表现或临床症状',
    'mic': '微生物，细菌病毒真菌等病原体',
    'pro': '医疗程序，治疗操作手术或诊疗过程',
    'ite': '医学检验检查项目，实验室检查或影像检查项目',
    'dep': '医院科室，临床科室或医疗部门',
    'dru': '药物名称，药品或治疗用药',
    'equ': '医疗设备，器械仪器或检查治疗设备',
}

# CLUENER 常见标签。
CLUENER_LABEL_DESCRIPTIONS = {
    'address': '地址，地点位置道路行政区划或具体场所',
    'book': '书名，图书小说教材文献等作品名称',
    'company': '公司企业商业机构或品牌组织',
    'game': '游戏名称，电子游戏网络游戏或娱乐游戏',
    'government': '政府机构，行政机关公共部门或官方组织',
    'movie': '电影电视剧综艺等影视作品名称',
    'name': '人名，人物姓名或称呼',
    'organization': '组织机构，学校协会团队或非商业机构',
    'position': '职位职务头衔职业身份',
    'scene': '景点景区自然或人文旅游场景名称',
}

# 自定义标签描述会覆盖上面默认描述；为 None 时自动使用对应数据集描述。
CUSTOM_LABEL_DESCRIPTIONS = None

# =========================
# 验证集阈值搜索配置
# =========================
USE_THRESHOLD_SEARCH = True
THRESHOLD_SEARCH_MODE = 'per_class'  # 'global' 或 'per_class'
THRESHOLD_SEARCH_MIN = -1.5
THRESHOLD_SEARCH_MAX = 2.0
THRESHOLD_SEARCH_STEP = 0.05
DEFAULT_PRED_THRESHOLD = 0.0

# =========================
# 根据两个论文创新点总开关生成最终生效配置
# =========================
# 说明：上面的细粒度开关保留，便于进一步做子模块消融；
# 这里的两个总开关优先级更高，用于一键关闭对应创新点的完整链路。
if not USE_SPAN_TYPE_SEMANTIC_FUSION_INNOVATION:
    USE_SPAN_TYPE_CONTRASTIVE = False
    USE_SPAN_TYPE_FUSION = False
    SPAN_TYPE_LOSS_WEIGHT = 0.0
    SPAN_TYPE_NEG_LOSS_WEIGHT = 0.0
    SPAN_TYPE_USE_IN_ADVERSARIAL = False
    SPAN_TYPE_FUSION_GAMMA_GRID = [0.0]
    DEFAULT_SPAN_TYPE_FUSION_GAMMA = 0.0

if not USE_MULTISCALE_CALIBRATION_THRESHOLD_INNOVATION:
    USE_MULTISCALE_FUSION = False
    USE_GATED_MULTISCALE_MAIN_FUSION = False
    USE_MULTISCALE_NEG_CALIBRATION = False
    MULTISCALE_AUX_LOSS_WEIGHT = 0.0
    MULTISCALE_NEG_CALIB_WEIGHT = 0.0
    MULTISCALE_USE_IN_ADVERSARIAL = False
    MULTISCALE_FUSION_BETA_GRID = [0.0]
    DEFAULT_MULTISCALE_FUSION_BETA = 0.0
    # “联合阈值优化”被归入第二个创新点，因此关闭该创新点时使用固定默认阈值。
    USE_THRESHOLD_SEARCH = False

# 这两个变量只用于日志记录，反映最终实际生效状态。
EFFECTIVE_USE_SPAN_TYPE_SEMANTIC_FUSION = bool(
    USE_SPAN_TYPE_SEMANTIC_FUSION_INNOVATION
    and USE_SPAN_TYPE_CONTRASTIVE
    and USE_SPAN_TYPE_FUSION
)
EFFECTIVE_USE_MULTISCALE_CALIBRATION_THRESHOLD = bool(
    USE_MULTISCALE_CALIBRATION_THRESHOLD_INNOVATION
    and (USE_MULTISCALE_FUSION or USE_GATED_MULTISCALE_MAIN_FUSION or USE_MULTISCALE_NEG_CALIBRATION or USE_THRESHOLD_SEARCH)
)

# =========================
# 对抗训练配置
# =========================
ADV_TRAIN_METHOD = 'PGD'  # 'NONE'、'FGM' 或 'PGD'
ADV_EMB_NAME = 'encoder.embeddings.word_embeddings'
FGM_EPSILON = 1.0
PGD_EPSILON = 0.3
PGD_ALPHA = 0.1
PGD_STEPS = 3
