# -*- coding: utf-8 -*-
"""
触觉 & 视觉物理量计算函数
============================
- 触觉 Pt (6维): 三指力总和 / 均衡度 / 合成力心 cx,cy / 接触面积比 / 力分布熵
- 视觉 Pv (4维): 边缘强度 / 深度方差 / 深度有效率 / 亮度

所有函数都是纯函数, 无副作用, 便于单元测试。
"""

import numpy as np
import cv2

import config


# ============================================================ #
#  触觉数据预处理
# ============================================================ #

def calibrate_tactile(tactile_raw, baseline_per_finger=None, clip_negative=True):
    """
    触觉基线校准: 减去空载本底
    
    Args:
        tactile_raw: shape=(..., 3, 15, 5) 原始触觉数据 (任意前缀维度)
        baseline_per_finger: shape=(3,) 每指空载总力, 默认从 config 读
        clip_negative: 是否把负值裁到 0
        
    Returns:
        校准后的触觉数据, shape 与输入相同
    """
    if baseline_per_finger is None:
        baseline_per_finger = config.TACTILE_BASELINE_PER_FINGER
    
    baseline = np.asarray(baseline_per_finger, dtype=np.float32)
    # 每个点平均: 总力 / 75 (15行 × 5列)
    per_point = baseline / 75.0   # (3,)
    # reshape 成可广播形状
    per_point = per_point.reshape(3, 1, 1)   # (3, 1, 1)
    
    out = tactile_raw.astype(np.float32) - per_point
    if clip_negative:
        out = np.clip(out, 0, None)
    return out


def aggregate_tactile_window(tactile_window, methods=None):
    """
    对触觉时序窗口做统计聚合, 得到 CNN 的输入
    
    Args:
        tactile_window: shape=(T, 3, 15, 5) 已校准的触觉时序
        methods: list of str, 聚合方式, 默认从 config 读
        
    Returns:
        shape=(3*len(methods), 15, 5), 多个统计量沿通道维度拼接
        例如 methods=['mean','max','std'] -> (9, 15, 5)
    """
    if methods is None:
        methods = config.TACTILE_AGG_METHODS
    
    parts = []
    for m in methods:
        if m == 'mean':
            parts.append(tactile_window.mean(axis=0))
        elif m == 'max':
            parts.append(tactile_window.max(axis=0))
        elif m == 'std':
            parts.append(tactile_window.std(axis=0))
        else:
            raise ValueError(f"未知的聚合方式: {m}")
    
    # 沿"指"维度拼接: (3,15,5) × N → (3*N, 15, 5)
    return np.concatenate(parts, axis=0).astype(np.float32)


# ============================================================ #
#  触觉物理量 Pt (6维)
# ============================================================ #

def extract_tactile_features(tactile_mean,
                              finger_positions=None,
                              threshold=None):
    """
    从校准后的单帧触觉数据提取 6 维物理量
    
    Args:
        tactile_mean: shape=(3, 15, 5), 校准后的"代表"触觉
                      (推荐用聚合后的 mean 通道, 即 (9,15,5)[:3])
        finger_positions: shape=(3, 2), 三指 (x,y) 坐标, mm; 默认从 config 读
        threshold: float, 激活点判定阈值; 默认从 config 读
        
    Returns:
        np.ndarray, shape=(6,), 顺序:
            [F_total, balance, cx, cy, contact_ratio, entropy]
    """
    if finger_positions is None:
        finger_positions = np.asarray(config.FINGER_POSITIONS, dtype=np.float32)
    else:
        finger_positions = np.asarray(finger_positions, dtype=np.float32)
    if threshold is None:
        threshold = config.ACTIVATION_THRESHOLD
    
    assert tactile_mean.shape == (3, 15, 5), \
        f"期望 shape=(3,15,5), 实际 {tactile_mean.shape}"
    
    # ── 特征1: 三指力总和 ──
    F_per_finger = tactile_mean.sum(axis=(1, 2))   # (3,)
    F_total = float(F_per_finger.sum())
    
    # ── 特征2: 三指力均衡度 (CV = std/mean) ──
    mean_f = F_per_finger.mean()
    balance = float(F_per_finger.std() / (mean_f + 1e-6))
    
    # ── 特征3&4: 合成力心 cx, cy ──
    if F_total < 1e-6:
        cx, cy = 0.0, 0.0
    else:
        cx = float((F_per_finger * finger_positions[:, 0]).sum() / F_total)
        cy = float((F_per_finger * finger_positions[:, 1]).sum() / F_total)
    
    # ── 特征5: 接触面积比 ──
    contact_ratio = float((tactile_mean > threshold).sum() / 225.0)
    
    # ── 特征6: 力分布熵 ──
    flat = tactile_mean.flatten().astype(np.float32)
    flat = np.clip(flat, 0, None) + 1e-6
    p = flat / flat.sum()
    entropy = float(-(p * np.log(p)).sum())
    
    return np.array([F_total, balance, cx, cy, contact_ratio, entropy],
                    dtype=np.float32)


# ============================================================ #
#  视觉数据预处理
# ============================================================ #

def preprocess_vision(rgb, depth, output_size=None,
                       depth_min=None, depth_max=None):
    """
    视觉预处理: resize + 归一化 + 拼成 4 通道
    
    Args:
        rgb: shape=(H, W, 3), uint8, BGR (OpenCV读出的)
        depth: shape=(H, W), uint16 或 float32, 单位 mm
        output_size: int, 输出尺寸 (默认从 config 读)
        depth_min, depth_max: 深度归一化范围 (mm)
        
    Returns:
        shape=(4, S, S), float32, 通道顺序 [R, G, B, D_norm]
        RGB 归一化到 [0, 1]
        Depth 归一化到 [0, 1] (clip 到 [depth_min, depth_max])
    """
    if output_size is None:
        output_size = config.VISION_INPUT_SIZE
    if depth_min is None:
        depth_min = config.DEPTH_MIN_MM
    if depth_max is None:
        depth_max = config.DEPTH_MAX_MM
    
    # RGB: BGR → RGB, resize, 归一化到 [0,1]
    rgb_resized = cv2.resize(rgb, (output_size, output_size),
                              interpolation=cv2.INTER_AREA)
    rgb_rgb = cv2.cvtColor(rgb_resized, cv2.COLOR_BGR2RGB)
    rgb_norm = rgb_rgb.astype(np.float32) / 255.0   # (S,S,3)
    rgb_chw = np.transpose(rgb_norm, (2, 0, 1))     # (3,S,S)
    
    # Depth: resize, clip, 归一化
    depth_resized = cv2.resize(depth.astype(np.float32),
                                (output_size, output_size),
                                interpolation=cv2.INTER_NEAREST)
    # 无效像素 (0 或异常大值) 保留为 0 (深度有效率特征会处理它们)
    depth_clip = np.clip(depth_resized, 0, depth_max)
    depth_norm = np.zeros_like(depth_clip)
    valid = depth_clip >= depth_min
    depth_norm[valid] = (depth_clip[valid] - depth_min) / (depth_max - depth_min)
    depth_chw = depth_norm[np.newaxis, :, :]   # (1,S,S)
    
    # 拼成 4 通道
    rgbd = np.concatenate([rgb_chw, depth_chw], axis=0).astype(np.float32)
    return rgbd


# ============================================================ #
#  视觉物理量 Pv (4维)
# ============================================================ #

def extract_visual_features(rgb, depth,
                             depth_min=None, depth_max=None):
    """
    从原始 RGB + Depth 提取 4 维视觉物理量
    
    Args:
        rgb: shape=(H, W, 3), uint8, BGR
        depth: shape=(H, W), uint16 或 float32, 单位 mm (无效像素=0)
        depth_min, depth_max: 深度有效范围
        
    Returns:
        np.ndarray, shape=(4,), 顺序:
            [edge_strength, depth_variance, depth_validity, brightness]
        所有值已归一化到 [0,1] 量级附近
    """
    if depth_min is None:
        depth_min = config.DEPTH_MIN_MM
    if depth_max is None:
        depth_max = config.DEPTH_MAX_MM
    
    # ── 特征1: 边缘强度 (Canny 边缘像素比例) ──
    # 反映图像清晰度 / mask 清晰度的近似
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)   # 阈值固定
    edge_strength = float((edges > 0).sum() / edges.size)   # ∈ [0,1]
    
    # ── 特征2: 中心 ROI 区域深度方差 ──
    # 物体一般在画面中心, ROI 越复杂(方差大)说明物体形状特征明显
    H, W = depth.shape
    cy, cx = H // 2, W // 2
    half = min(H, W) // 4   # 中心 50% × 50% 区域
    roi = depth[cy-half:cy+half, cx-half:cx+half].astype(np.float32)
    valid_roi = roi[(roi >= depth_min) & (roi <= depth_max)]
    if len(valid_roi) > 10:
        # 归一化方差: std / (depth_max - depth_min)
        depth_variance = float(valid_roi.std() / (depth_max - depth_min))
    else:
        depth_variance = 0.0
    
    # ── 特征3: 深度有效率 (1 - 无效像素比例) ──
    # 间接反映遮挡: 遮挡通常会导致深度无效或异常
    valid_mask = (depth >= depth_min) & (depth <= depth_max)
    depth_validity = float(valid_mask.sum() / depth.size)   # ∈ [0,1]
    
    # ── 特征4: 整体亮度 ──
    # 灰度均值归一化到 [0,1]
    brightness = float(gray.mean() / 255.0)
    
    return np.array([edge_strength, depth_variance, depth_validity, brightness],
                    dtype=np.float32)


# ============================================================ #
#  触觉窗口截取 (辅助函数)
# ============================================================ #

def extract_tactile_window(tactile_seq, tactile_ts, meta,
                            anchor=None, offset_sec=None, duration_sec=None):
    """
    从触觉时序中截取一段窗口 (用 meta 中的事件时间戳定位)
    
    Args:
        tactile_seq: shape=(N, 3, 15, 5), 完整触觉时序
        tactile_ts: shape=(N,), 每帧时间戳
        meta: dict, meta.json 加载结果, 必须含 meta['timestamps'][anchor]
        anchor, offset_sec, duration_sec: 窗口锚点和长度, 默认从 config 读
        
    Returns:
        tactile_window: shape=(T, 3, 15, 5), 截取出的窗口 (T 帧)
        若锚点时刻找不到, 返回 None
    """
    if anchor is None:
        anchor = config.WINDOW_ANCHOR
    if offset_sec is None:
        offset_sec = config.WINDOW_OFFSET_SEC
    if duration_sec is None:
        duration_sec = config.WINDOW_DURATION_SEC
    
    timestamps = meta.get('timestamps', {})
    if anchor not in timestamps:
        return None
    
    t_anchor = float(timestamps[anchor])
    t_start = t_anchor + offset_sec
    t_end = t_start + duration_sec
    
    mask = (tactile_ts >= t_start) & (tactile_ts <= t_end)
    if mask.sum() < 3:
        # 帧数太少, 回退: 锚点之后所有帧最多取 duration 长度
        mask = (tactile_ts >= t_start) & (tactile_ts <= t_start + duration_sec)
        if mask.sum() < 3:
            return None
    
    return tactile_seq[mask]


# ============================================================ #
#  快速自测 (作为 __main__ 运行)
# ============================================================ #

if __name__ == '__main__':
    import os
    
    print("=" * 60)
    print("features.py 自测")
    print("=" * 60)
    
    # ── 1. 触觉校准 + 聚合 + 物理量提取 ──
    print("\n[1] 模拟触觉数据测试")
    np.random.seed(0)
    # 模拟 50 帧, 第一帧是空载, 后面有接触
    fake_seq = np.full((50, 3, 15, 5), 80.0, dtype=np.float32)   # 全部底噪
    fake_seq += np.random.randn(*fake_seq.shape) * 5             # 添加噪声
    # 后 25 帧加上接触力 (中心区域)
    fake_seq[25:, :, 6:9, 1:4] += 100   # 三指中心区域有 ~100 的力
    
    print(f"  原始数据 shape: {fake_seq.shape}, 范围 [{fake_seq.min():.0f}, {fake_seq.max():.0f}]")
    
    calibrated = calibrate_tactile(fake_seq)
    print(f"  校准后范围: [{calibrated.min():.1f}, {calibrated.max():.1f}]")
    
    agg = aggregate_tactile_window(calibrated)
    print(f"  聚合后 shape: {agg.shape} (期望 (9,15,5))")
    
    # 用 mean 通道 (前3指) 提取物理量
    mean_tactile = agg[:3]   # 第一个统计量 mean
    Pt = extract_tactile_features(mean_tactile)
    print(f"  物理量 Pt: {Pt}")
    print(f"    F_total       = {Pt[0]:.2f}")
    print(f"    balance       = {Pt[1]:.4f}")
    print(f"    cx, cy        = ({Pt[2]:.2f}, {Pt[3]:.2f}) mm")
    print(f"    contact_ratio = {Pt[4]:.4f}")
    print(f"    entropy       = {Pt[5]:.4f}")
    
    # ── 2. 用真实样本 (如果存在) ──
    sample_data_path = '/mnt/user-data/uploads/1781517429089_tactile_seq.npy'
    if os.path.exists(sample_data_path):
        print(f"\n[2] 真实触觉样本测试: {sample_data_path}")
        real = np.load(sample_data_path)
        print(f"  原始 shape: {real.shape}")
        
        # 取后 50 帧 (假设是 lifted 后的稳定段)
        window = real[-50:]
        cal = calibrate_tactile(window)
        agg = aggregate_tactile_window(cal)
        Pt = extract_tactile_features(agg[:3])
        print(f"  最后50帧物理量 Pt:")
        print(f"    F_total       = {Pt[0]:.2f}  (校准后总力)")
        print(f"    balance       = {Pt[1]:.4f}")
        print(f"    cx, cy        = ({Pt[2]:.2f}, {Pt[3]:.2f}) mm")
        print(f"    |c|           = {np.sqrt(Pt[2]**2 + Pt[3]**2):.2f} mm")
        print(f"    contact_ratio = {Pt[4]:.4f}  ({int(Pt[4]*225)}/225 个点激活)")
        print(f"    entropy       = {Pt[5]:.4f}")
    
    # ── 3. 视觉预处理 + 物理量 ──
    print("\n[3] 模拟视觉数据测试")
    fake_rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    fake_depth = np.random.randint(400, 600, (400, 640), dtype=np.uint16)
    fake_depth[:50, :] = 0   # 一些无效像素
    
    rgbd = preprocess_vision(fake_rgb, fake_depth)
    print(f"  RGB-D 输出 shape: {rgbd.shape} (期望 (4,224,224))")
    print(f"  RGB 通道范围: [{rgbd[:3].min():.3f}, {rgbd[:3].max():.3f}]")
    print(f"  Depth 通道范围: [{rgbd[3].min():.3f}, {rgbd[3].max():.3f}]")
    
    Pv = extract_visual_features(fake_rgb, fake_depth)
    print(f"  视觉物理量 Pv: {Pv}")
    print(f"    edge_strength  = {Pv[0]:.4f}")
    print(f"    depth_variance = {Pv[1]:.4f}")
    print(f"    depth_validity = {Pv[2]:.4f}")
    print(f"    brightness     = {Pv[3]:.4f}")
    
    print("\n" + "=" * 60)
    print("✅ features.py 自测完成")
    print("=" * 60)
