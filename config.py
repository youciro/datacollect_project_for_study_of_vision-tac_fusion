# -*- coding: utf-8 -*-
"""
视触觉融合抓取数据采集 —— 全局配置
====================================
所有可调参数集中在这里，改这一个文件即可。
"""

# ====================== 硬件连接 ======================

# --- 机械臂 ---
ARM_IP = '192.168.100.1'
ARM_PORT = 8899
ARM_SPEED = 0.1                # 运动速度比例 0~1
Z_MIN_LIMIT = 0.204           # 机械臂末端Z轴最低限制（米），根据实际填

# --- 触觉传感器 ---
TACTILE_PORT = 'COM14'         # 串口号
TACTILE_BAUDRATE = 1000000     # 波特率
TACTILE_FREQ = 50              # 期望采样频率 Hz
TACTILE_ENDIAN_BIG = True      # 大端序
TACTILE_NOISE_HIGH = 5000     # 噪声上限
TACTILE_NOISE_LOW = 10        # 噪声下限
TACTILE_FINGER_ORDER = (0, 1, 2)

# --- 相机 ---
# 相机无需额外参数，CameraReader 内部用默认配置


# ====================== 采集流程参数 ======================

# 触觉时序保存范围：按空格那一刻为中心，前后各取多少秒
TACTILE_PRE_SECONDS = 2.0      # 接触前
TACTILE_POST_SECONDS = 2.0     # 提起后
# 总时序长度 = 4 秒 ≈ 200 帧 (50Hz)

# 接触检测：触觉总力突变阈值（用于自动标记接触瞬间时间戳）
CONTACT_FORCE_THRESHOLD = 100 #250   # 总力跃变超过此值判定为"接触发生"
CONTACT_DETECT_WINDOW = 0.1        # 在多少秒窗口内检测跃变

# 提起动作
LIFT_HEIGHT = 0.05             # 提起高度（米）
LIFT_STABLE_SECONDS = 2.0      # 提起后稳定保持秒数


# ====================== 物体放置网格 ======================
# eye-on-base + 预定网格点方案
# 桌面预定 9 个网格点，采集时在 GUI 下拉选当前用哪个点

GRID_POINTS = [
    (0.102, -0.374), (0.102, -0.477), (0.102, -0.576),
    (0.003, -0.380), (0.003, -0.479), (0.003, -0.576),
    (-0.096, -0.380), (-0.096, -0.485), (-0.096, -0.582),
]

# 预定抓取角度（度），GUI 里选
GRASP_ANGLES = [0, 45, 90]

# 各类物体的抓取高度（米），根据实测填入
OBJECT_HEIGHTS = {
    'apple':      0.210, #0.219浅,#205深
    'bottlewater':   0.344,
    'orange':      0.223,#0.210,深#浅,0.240,#
    'fakeapple': 0.242,#deep0.214 normal0.225 shalow0.242
    'ball':     0.13,
    'tool':     0.11,
    'soft_toy': 0.14,
    'other':    0.13,
}


# ====================== 标注标签 ======================

# 抓取结果标签
RESULT_LABELS = [
    'success',        # 成功抓起并稳定保持
    'radial_offset',  # 径向偏移：力心偏移，夹偏了
    'too_shallow',    # 轴向偏浅：接触面积小、力弱，抓得太浅
    'too_deep',       # 轴向偏深：力突然增大不稳定，插入过深
    'grasp_miss',     # 抓空：触觉全0，完全没接触到物体
]

# 物体类别清单（按实际实验物体修改）
OBJECT_CLASSES = [
    'apple',    # 塑料正方块
    'bottlewater',             # 易拉罐
    'orange',             # 马克杯
    'fakeapple',            #plasticapple
    'cardboard_box',   # 纸盒
    'tennis_ball',     # 网球
    'apple_model',     # 苹果模型
    'apple',           # 真苹果
    'sponge',          # 海绵
    'rubber_doll',     # 橡胶娃娃
    'plastic_bottle',  # 空塑料瓶
    'pen',             # 笔
    'water_bottle',    # 装水瓶
    'smooth_mug',      # 滑面马克杯
    'other',           # 备用
]


# ====================== 数据保存 ======================

# 数据保存根目录（写死，按需修改这一行）
SAVE_ROOT = 'D:/grasp_data'

# 单个样本文件夹命名格式
SAMPLE_DIR_FORMAT = 'sample_{:05d}'

# 进度文件名（断点续传用）
PROGRESS_FILE = 'progress.json'


# ====================== GUI ======================

GUI_REFRESH_MS = 100           # GUI 刷新间隔毫秒
GUI_WINDOW_TITLE = '视触觉抓取数据采集'
