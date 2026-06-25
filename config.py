# -*- coding: utf-8 -*-
"""
训练端总配置
=============
所有路径、超参、阈值集中在此，其他文件统一从这里导入。

⚠️ 修改前请参考《可调参数清单.md》了解各参数的含义和调整建议。
"""

import os

# ============================================================ #
#  路径配置
# ============================================================ #

# 项目根目录（自动定位为本文件所在目录）
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 原始数据目录（采集机传过来的 sample 文件夹）
# 结构: data/sample_00001/, data/sample_00002/, ...
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data')

# 启用白名单
WHITELIST_ENABLED = True
# 因为whitelist.json 在 data 文件夹里
WHITELIST_PATH = os.path.join(DATA_ROOT, 'whitelist.json')

# 预处理 cache 文件（preprocess.py 生成，dataset.py 读取）
CACHE_PATH = os.path.join(PROJECT_ROOT, 'data', 'cache.npz')

# 训练输出根目录（日志、checkpoint、TensorBoard）
TRAIN_OUTPUT_ROOT = os.path.join(PROJECT_ROOT, 'train')
os.makedirs(TRAIN_OUTPUT_ROOT, exist_ok=True)

# 子目录
CHECKPOINT_DIR = os.path.join(TRAIN_OUTPUT_ROOT, 'checkpoints')
TENSORBOARD_DIR = os.path.join(TRAIN_OUTPUT_ROOT, 'tensorboard')
LOG_DIR = os.path.join(TRAIN_OUTPUT_ROOT, 'logs')




# ============================================================ #
#  数据集划分
# ============================================================ #

# 训练集物体（已采集 / 计划采集），用于 train + val（80/20 按 sample 划分）
TRAIN_OBJECTS = [
    'apple',          # 苹果
    'orange',         # 假橙子
    'fakepear',           # 假梨
    'mug',            # 马克杯
    'tennisball',    # 网球
    'bottlewater',   # 装水塑料瓶
    'mouse',          # 鼠标
    'fakeapple',     # 假苹果
]

# 测试集物体（未见物体，用于实验 E5 泛化测试）
# 注: 现在为空占位，等采集完成后填入
TEST_OBJECTS = [
    # 'tea_cup',
    # 'cucumber',
    # 'plush_toy',
    # 'chopsticks',
    # 'stainless_cup',
]

# 训练/验证按 sample 划分比例
VAL_RATIO = 0.2
SPLIT_SEED = 42

# ============================================================ #
#  物理量全局标准化配置
# ============================================================ #
ENABLE_PHYSICAL_FEATURE_NORMALIZATION = True  # 是否启用Pt/Pv标准化
# 标准化统计量保存路径（自动关联训练集划分，避免不同划分用错统计量）
import hashlib
train_obj_str = '_'.join(sorted(TRAIN_OBJECTS))  # 训练物体列表转字符串
train_obj_hash = hashlib.md5(train_obj_str.encode()).hexdigest()[:8]  # 哈希防冲突
NORM_STATS_PATH = os.path.join(
    TRAIN_OUTPUT_ROOT,
    f'normalization_stats_seed{SPLIT_SEED}_{train_obj_hash}.npz'
)

# ============================================================ #
#  标签编码
# ============================================================ #

LABEL2INT = {
    'success':       0,
    'radial_offset': 1,
    'too_shallow':   2,
    'too_deep':      3,
    'grasp_miss':    4,
}
INT2LABEL = {v: k for k, v in LABEL2INT.items()}
NUM_CLASSES = len(LABEL2INT)


# ============================================================ #
#  触觉数据处理
# ============================================================ #

# 三指基线（空载本底），校准时减去
TACTILE_BASELINE_PER_FINGER = [5988.0, 5994.0, 5992.0]

# 触觉传感器采样频率
TACTILE_FREQ = 50.0   # Hz

# 时间窗口：'lifted' 之后 1 秒，共 50 帧
WINDOW_ANCHOR = 'lifted'
WINDOW_OFFSET_SEC = 0.0
WINDOW_DURATION_SEC = 1.0

# 聚合方式：50帧 → mean+max+std 沿通道拼接 → (9, 15, 5)
TACTILE_AGG_METHODS = ['mean', 'max', 'std']
TACTILE_INPUT_CHANNELS = 3 * len(TACTILE_AGG_METHODS)


# ============================================================ #
#  触觉物理量计算
# ============================================================ #

# 三指物理位置（夹爪坐标系，单位 mm）
FINGER_POSITIONS = [
    [-50.0,   0.0],
    [ 25.0, -43.3],
    [ 25.0,  43.3],
]

# 接触面积比 的"激活点"阈值（校准后单点力 > 该值才算激活）
ACTIVATION_THRESHOLD = 80.0


# ============================================================ #
#  视觉数据处理
# ============================================================ #

VISION_TIMING = 'before'         # 'before' or 'after'
VISION_INPUT_SIZE = 224

# 深度归一化范围（mm）
DEPTH_MIN_MM = 200.0
DEPTH_MAX_MM = 1000.0


# ============================================================ #
#  网络结构
# ============================================================ #

VISUAL_FEAT_DIM = 256
TACTILE_FEAT_DIM = 128
FUSION_DIM = 256
PV_DIM = 4
PT_DIM = 6
USE_PRETRAINED_RESNET = True


# ============================================================ #
#  训练超参
# ============================================================ #

BATCH_SIZE = 32

# 学习率（小数据集用 1e-3）
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# Scheduler
EPOCHS = 50
LR_SCHEDULER_T_MAX = 50

# 两个输出头的损失权重
LOSS_WEIGHT_SUCCESS = 1.0
LOSS_WEIGHT_FAILURE = 1.0

# 标签不平衡处理：损失加权（基于类别频次自动计算，在 train.py 中实现）
USE_CLASS_WEIGHTS = True

# 早停（val loss 连续 N epochs 不下降则停）
EARLY_STOP_ENABLED = True
EARLY_STOP_PATIENCE = 10
EARLY_STOP_MIN_DELTA = 1e-4

# DataLoader
NUM_WORKERS = 4
PIN_MEMORY = True

# Checkpoint
SAVE_EVERY_N_EPOCHS = 5
KEEP_BEST_N = 3
SAVE_LAST = True


# ============================================================ #
#  数据增强（训练时实时做，验证/测试不增强）
# ============================================================ #

AUG_ENABLED = True
AUG_BRIGHTNESS = 0.2
AUG_CONTRAST = 0.2
AUG_SATURATION = 0.2
AUG_ROTATION_DEG = 5.0


# ============================================================ #
#  随机种子
# ============================================================ #

GLOBAL_SEED = 42




# ============================================================ #
#  调试 / 打印
# ============================================================ #

if __name__ == '__main__':
    import json

    print("=" * 60)
    print("训练端配置 (config.py)")
    print("=" * 60)
    cfg = {k: v for k, v in globals().items()
           if k.isupper() and not k.startswith('_')}
    print(json.dumps(cfg, indent=2, ensure_ascii=False, default=str))
    print("=" * 60)
    print(f"DATA_ROOT 是否存在: {os.path.exists(DATA_ROOT)}")
    print(f"CACHE_PATH 是否存在: {os.path.exists(CACHE_PATH)}")
    print("=" * 60)
