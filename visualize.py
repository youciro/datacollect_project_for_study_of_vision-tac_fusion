# -*- coding: utf-8 -*-
"""
可视化脚本: 论文 Fig.5 等图表
================================
功能:
  1. 单 sample 综合可视化: RGB / Depth / 触觉热图 / 物理量 / 模型预测 / 置信度
  2. 置信度时序图 (跨多个 sample / 模拟遮挡和正常场景)
  3. 模态权重分布图 (wv vs ct)

运行: python visualize.py --ckpt <path> --sample <sample_dir>
"""

import os
import json
import argparse

import numpy as np
import torch
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import config
import features
from model import GraspFusionModel
from dataset import load_cache

def load_norm_stats():
    """
    加载训练阶段保存的物理量标准化统计量
    """
    if not hasattr(config, 'ENABLE_PHYSICAL_FEATURE_NORMALIZATION'):
        return None
    if not config.ENABLE_PHYSICAL_FEATURE_NORMALIZATION:
        return None

    if not os.path.exists(config.NORM_STATS_PATH):
        print(f"⚠️ 标准化统计量不存在: {config.NORM_STATS_PATH}")
        print("   请先运行一次训练，或关闭 ENABLE_PHYSICAL_FEATURE_NORMALIZATION")
        return None

    stats = np.load(config.NORM_STATS_PATH)
    return {
        'pt_mean': stats['pt_mean'],
        'pt_std':  stats['pt_std'],
        'pv_mean': stats['pv_mean'],
        'pv_std':  stats['pv_std'],
    }


# ============================================================ #
#  单 sample 综合图
# ============================================================ #

def visualize_single_sample(model, device, sample_dir, save_path):
    """
    给定原始 sample 文件夹，画出综合可视化:
       ┌─ RGB ─┬─ Depth ─┬─ 触觉(三指热图) ─┐
       │       │         │                  │
       ├─ 物理量数值 + 预测结果 + 置信度条 ─┤
       └─────────────────────────────────────┘
    """
    # 读 meta + 数据
    with open(os.path.join(sample_dir, 'meta.json'), 'r', encoding='utf-8') as f:
        meta = json.load(f)
    
    timing = config.VISION_TIMING
    rgb = cv2.imread(os.path.join(sample_dir, f'rgb_{timing}.png'))
    rgb_rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    depth = np.load(os.path.join(sample_dir, f'depth_{timing}.npy'))
    tac_seq = np.load(os.path.join(sample_dir, 'tactile_seq.npy'))
    tac_ts = np.load(os.path.join(sample_dir, 'tactile_ts.npy'))
    
    # 处理触觉
    window = features.extract_tactile_window(tac_seq, tac_ts, meta)
    if window is None:
        window = tac_seq[-50:]
    cal = features.calibrate_tactile(window)
    agg = features.aggregate_tactile_window(cal)
    Pt = features.extract_tactile_features(agg[:3])
    
    # 处理视觉
    Pv = features.extract_visual_features(rgb, depth)

    # ---------- 关键：应用训练时的全局标准化 ----------
    norm_stats = load_norm_stats()
    Pt_raw = Pt.copy()
    Pv_raw = Pv.copy()

    if norm_stats is not None:
        Pt = (Pt - norm_stats['pt_mean']) / (norm_stats['pt_std'] + 1e-6)
        Pv = (Pv - norm_stats['pv_mean']) / (norm_stats['pv_std'] + 1e-6)
    # ---------------------------------------------------

    if depth.shape[:2] != rgb.shape[:2]:
        depth_r = cv2.resize(depth.astype(np.float32),
                              (rgb.shape[1], rgb.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
    else:
        depth_r = depth
    rgbd = features.preprocess_vision(rgb, depth_r)
    
    # 模型推理
    batch = {
        'rgbd':    torch.from_numpy(rgbd).unsqueeze(0).float().to(device),
        'tactile': torch.from_numpy(agg).unsqueeze(0).float().to(device),
        'Pv':      torch.from_numpy(Pv).unsqueeze(0).float().to(device),
        'Pt':      torch.from_numpy(Pt).unsqueeze(0).float().to(device),
    }
    with torch.no_grad():
        out = model(batch)
    
    pred_label = out['failure_logits'].argmax(dim=1).item()
    pred_name = config.INT2LABEL[pred_label]
    success_prob = torch.sigmoid(out['success_logit']).item()
    cv_val = out['cv'].item()
    ct_val = out['ct'].item()
    wv_val = out['wv'].item()
    wt_val = out['wt'].item()
    
    gt_name = meta.get('result_label', 'unknown')
    
    # 画图
    fig = plt.figure(figsize=(15, 9))
    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)
    
    # 1. RGB
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(rgb_rgb)
    ax.set_title(f'RGB ({timing})')
    ax.axis('off')
    
    # 2. Depth
    ax = fig.add_subplot(gs[0, 1])
    depth_show = np.where((depth >= config.DEPTH_MIN_MM) & (depth <= config.DEPTH_MAX_MM),
                           depth, 0)
    im = ax.imshow(depth_show, cmap='viridis')
    ax.set_title(f'Depth ({timing})')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # 3-5. 触觉三指热图（用 mean 通道）
    finger_labels = ['Finger 1 (9 o\'clock)', 'Finger 2 (5 o\'clock)', 'Finger 3 (1 o\'clock)']
    vmax = agg[:3].max()
    for fi in range(3):
        ax = fig.add_subplot(gs[0, 2] if fi == 0 else gs[1, fi-1])
        im = ax.imshow(agg[fi], cmap='hot', vmin=0, vmax=vmax)
        ax.set_title(finger_labels[fi])
        ax.set_xlabel('col')
        ax.set_ylabel('row')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    # 6. 触觉物理量
    ax = fig.add_subplot(gs[0, 3])
    ax.axis('off')
    Pt_text = (f"Tactile Features (Pt):\n"
               f"  F_total       = {Pt_raw[0]:.1f} → {Pt[0]:.3f}\n"
               f"  balance       = {Pt_raw[1]:.4f} → {Pt[1]:.3f}\n"
               f"  cx, cy        = ({Pt_raw[2]:.2f}, {Pt_raw[3]:.2f}) → ({Pt[2]:.3f}, {Pt[3]:.3f})\n"
               f"  |c|           = {np.sqrt(Pt_raw[2] ** 2 + Pt_raw[3] ** 2):.2f} → {np.sqrt(Pt[2] ** 2 + Pt[3] ** 2):.3f}\n"
               f"  contact_ratio = {Pt_raw[4]:.4f} → {Pt[4]:.3f}\n"
               f"  entropy       = {Pt_raw[5]:.4f} → {Pt[5]:.3f}\n"
               f"\n(→ = raw → normalized)")
    ax.text(0.05, 0.95, Pt_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lavender'))
    ax.set_title('Tactile Physical Features')
    
    # 7. 视觉物理量
    ax = fig.add_subplot(gs[1, 2])
    ax.axis('off')
    Pv_text = (f"Vision Features (Pv):\n"
               f"  edge_strength  = {Pv_raw[0]:.4f} → {Pv[0]:.3f}\n"
               f"  depth_variance = {Pv_raw[1]:.4f} → {Pv[1]:.3f}\n"
               f"  depth_validity = {Pv_raw[2]:.4f} → {Pv[2]:.3f}\n"
               f"  brightness     = {Pv_raw[3]:.4f} → {Pv[3]:.3f}\n"
               f"\n(→ = raw → normalized)")
    ax.text(0.05, 0.95, Pv_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow'))
    ax.set_title('Visual Physical Features')
    
    # 8. 预测 vs GT
    ax = fig.add_subplot(gs[1, 3])
    ax.axis('off')
    pred_text = (f"Prediction:\n"
                 f"  GT label   = {gt_name}\n"
                 f"  Pred label = {pred_name}\n"
                 f"  Match      = {'✓' if pred_name == gt_name else '✗'}\n\n"
                 f"  success_prob = {success_prob:.3f}\n"
                 f"  (1.0=success, 0.0=failure)")
    color = 'lightgreen' if pred_name == gt_name else 'lightcoral'
    ax.text(0.05, 0.95, pred_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor=color))
    ax.set_title('Prediction')
    
    # 9. 置信度对比条
    ax = fig.add_subplot(gs[2, :2])
    bars_y = ['cv (vision conf)', 'ct (tactile conf)']
    bars_v = [cv_val, ct_val]
    bars = ax.barh(bars_y, bars_v, color=['steelblue', 'orange'])
    for bar, v in zip(bars, bars_v):
        ax.text(v + 0.01, bar.get_y() + bar.get_height()/2,
                f'{v:.3f}', va='center')
    ax.set_xlim(0, 1)
    ax.set_title('Confidence Outputs')
    ax.set_xlabel('Confidence ∈ [0,1]')
    
    # 10. 融合权重
    ax = fig.add_subplot(gs[2, 2:])
    weights_y = ['wv (vision weight)', 'wt (tactile weight)']
    weights_v = [wv_val, wt_val]
    bars = ax.barh(weights_y, weights_v, color=['steelblue', 'orange'])
    for bar, v in zip(bars, weights_v):
        ax.text(v + 0.01, bar.get_y() + bar.get_height()/2,
                f'{v:.3f}', va='center')
    ax.set_xlim(0, 1)
    ax.set_title('Fusion Weights (softmax(cv, ct))')
    ax.set_xlabel('Weight (sum=1)')
    
    fig.suptitle(f'Sample: {os.path.basename(sample_dir)}  |  '
                  f'Object: {meta.get("object_class", "?")}',
                  fontsize=12, y=0.995)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存: {save_path}")


# ============================================================ #
#  跨样本：模态权重分布
# ============================================================ #

def visualize_modality_weights(predictions_npz_path, save_path):
    """
    从 eval.py 输出的 predictions.npz 画 wv / wt 分布
    """
    data = np.load(predictions_npz_path)
    wv = data['wv']
    wt = data['wt']
    labels = data['labels']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 直方图
    ax = axes[0]
    ax.hist(wv, bins=30, alpha=0.6, label='wv (vision)', color='steelblue')
    ax.hist(wt, bins=30, alpha=0.6, label='wt (tactile)', color='orange')
    ax.set_xlabel('Weight')
    ax.set_ylabel('Count')
    ax.set_title('Modality Weight Distribution (whole set)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 按标签分组的箱线图
    ax = axes[1]
    data_box = []
    box_labels = []
    for lbl_int in range(config.NUM_CLASSES):
        mask = labels == lbl_int
        if mask.sum() > 0:
            data_box.append(wt[mask])
            box_labels.append(config.INT2LABEL[lbl_int])
    ax.boxplot(data_box, labels=box_labels)
    ax.set_ylabel('Tactile weight (wt)')
    ax.set_title('Tactile Weight by Class')
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存: {save_path}")


# ============================================================ #
#  主入口
# ============================================================ #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True, help='checkpoint 路径')
    parser.add_argument('--sample', type=str, default=None,
                        help='单 sample 文件夹路径 (生成 Fig.5 风格综合图)')
    parser.add_argument('--predictions', type=str, default=None,
                        help='predictions.npz 路径 (生成模态权重分布图)')
    parser.add_argument('--output_dir', type=str, default='./visualize_output')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[设备] {device}")
    
    # 加载模型
    print(f"[加载模型] {args.ckpt}")
    model = GraspFusionModel().to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    
    # 单样本可视化
    if args.sample:
        print(f"\n[单样本可视化] {args.sample}")
        save_path = os.path.join(args.output_dir,
            f'sample_vis_{os.path.basename(args.sample)}.png')
        visualize_single_sample(model, device, args.sample, save_path)
    
    # 模态权重分布
    if args.predictions:
        print(f"\n[模态权重分布] {args.predictions}")
        save_path = os.path.join(args.output_dir, 'modality_weights.png')
        visualize_modality_weights(args.predictions, save_path)
    
    if not args.sample and not args.predictions:
        print("提示: 提供 --sample <dir> 或 --predictions <npz> 生成对应图")
    
    print("\n✅ 可视化完成")


if __name__ == '__main__':
    main()
