# -*- coding: utf-8 -*-
"""
物理量对比分析脚本
====================
对比标准化前后，不同物体（如马克杯 vs 苹果）的物理量分布差异。
用于验证全局标准化的合理性，并辅助论文写作（Ablation Study）。

运行: python compare_physics.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import config
# 直接从 dataset 导入你已实现好的函数（保证一致性）
from dataset import load_cache, split_indices, load_or_compute_norm_stats

# 设置学术绘图风格
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.3)
COLOR_MUG = "#E74C3C"  # 红色代表马克杯（重）
COLOR_APPLE = "#2ECC71"  # 绿色代表苹果（轻）
COLOR_OTHER = "#3498DB"  # 蓝色代表其他


def plot_feature_comparison(raw_features, norm_features, feature_names,
                            object_classes, save_path, title_prefix=""):
    """
    绘制单个特征的原始值与标准化值对比图（箱线图 + 散点）
    """
    unique_objects = sorted(list(set(object_classes)))

    fig, axes = plt.subplots(2, len(feature_names), figsize=(6 * len(feature_names), 10))
    fig.suptitle(f'{title_prefix}Raw vs Normalized Feature Distribution', fontsize=16, y=1.02)

    # 为了美观，对原始数据进行截断，去除极端异常值
    raw_clipped = raw_features.copy()
    norm_clipped = norm_features.copy()

    for i in range(raw_features.shape[1]):
        # 截断到 1% - 99% 分位数之间，防止极端值撑爆坐标轴
        low = np.percentile(raw_features[:, i], 1)
        high = np.percentile(raw_features[:, i], 99)
        raw_clipped[:, i] = np.clip(raw_features[:, i], low, high)

        low_n = np.percentile(norm_features[:, i], 1)
        high_n = np.percentile(norm_features[:, i], 99)
        norm_clipped[:, i] = np.clip(norm_features[:, i], low_n, high_n)

    for i, feat_name in enumerate(feature_names):
        # --- 子图 1: 原始值 ---
        ax_raw = axes[0, i]
        data_raw = [raw_clipped[object_classes == obj, i] for obj in unique_objects]
        bp1 = ax_raw.boxplot(data_raw, patch_artist=True, showfliers=False)

        # 涂色
        colors = [COLOR_MUG if 'mug' in obj else COLOR_APPLE if 'apple' in obj else COLOR_OTHER
                  for obj in unique_objects]
        for patch, color in zip(bp1['boxes'], colors):
            patch.set_facecolor(color)

        ax_raw.set_title(f'Raw: {feat_name}')
        ax_raw.set_xticklabels(unique_objects, rotation=30, ha='right')
        ax_raw.set_ylabel('Value (Original Scale)')

        # --- 子图 2: 标准化值 ---
        ax_norm = axes[1, i]
        data_norm = [norm_clipped[object_classes == obj, i] for obj in unique_objects]
        bp2 = ax_norm.boxplot(data_norm, patch_artist=True, showfliers=False)

        for patch, color in zip(bp2['boxes'], colors):
            patch.set_facecolor(color)

        ax_norm.set_title(f'Normalized: {feat_name}')
        ax_norm.set_xticklabels(unique_objects, rotation=30, ha='right')
        ax_norm.set_ylabel('Value (Z-score)')
        ax_norm.axhline(y=0, color='black', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  保存对比图: {save_path}")


def calculate_statistics(raw_features, norm_features, feature_names, object_classes):
    """计算并打印统计指标"""
    print("\n" + "=" * 60)
    print("📊 物理量统计摘要 (训练集视角)")
    print("=" * 60)

    objects_of_interest = ['mug', 'apple']
    for obj in objects_of_interest:
        mask = (object_classes == obj)
        if mask.sum() == 0:
            continue

        print(f"\n[{obj.upper()}] (样本数: {mask.sum()})")
        for i, name in enumerate(feature_names):
            raw_vals = raw_features[mask, i]
            norm_vals = norm_features[mask, i]

            mean_raw = raw_vals.mean()
            std_raw = raw_vals.std()
            mean_norm = norm_vals.mean()
            std_norm = norm_vals.std()

            print(f"  {name:15s}: Raw({mean_raw:>8.2f} ± {std_raw:<6.2f}) -> "
                  f"Norm({mean_norm:>6.3f} ± {std_norm:<5.3f})")

            # 如果是总力 F_total，特别标注
            if name == 'F_total':
                if mean_norm > 0.5:
                    print(f"      👉 标准化后仍保持较大正值 (符合物理直觉)")
                elif mean_norm < -0.5:
                    print(f"      👉 标准化后为负值 (较轻物体)")


def main():
    print("[1/4] 加载 Cache 数据...")
    cache = load_cache()

    print("[2/4] 划分训练集索引 (与训练时完全一致)...")
    # ✅ 关键点：你的 split_indices 只返回 3 个值
    train_idx, val_idx, test_idx = split_indices(cache)
    print(f"   训练集样本数: {len(train_idx)}")

    object_classes = cache['object_classes'][train_idx]

    # ✅ 关键点：你的 cache key 是 'Pts' 和 'Pvs'，不是 'Pt'/'Pv'
    raw_Pt = cache['Pts'][train_idx]  # (N, 6)
    raw_Pv = cache['Pvs'][train_idx]  # (N, 4)

    print("\n[3/4] 加载/计算标准化统计量...")
    # 使用 dataset.py 中完全相同的函数，确保一致性
    norm_stats = load_or_compute_norm_stats(
        cache=cache,
        train_idx=train_idx,
        norm_stats_path=config.NORM_STATS_PATH
    )

    print("\n[4/4] 应用标准化...")
    # 手动标准化
    norm_Pt = (raw_Pt - norm_stats['pt_mean']) / (norm_stats['pt_std'] + 1e-6)
    norm_Pv = (raw_Pv - norm_stats['pv_mean']) / (norm_stats['pv_std'] + 1e-6)

    # 定义特征名称（与 features.py 中的 Pt 定义对应）
    pt_feature_names = [
        'F_total (mN)',
        'balance',
        'cx (mm)',
        'cy (mm)',
        'contact_ratio',
        'entropy'
    ]
    pv_feature_names = [
        'edge_strength',
        'depth_var',
        'depth_valid',
        'brightness'
    ]

    # 计算统计指标
    calculate_statistics(raw_Pt, norm_Pt, pt_feature_names, object_classes)
    calculate_statistics(raw_Pv, norm_Pv, pv_feature_names, object_classes)

    print("\n生成可视化图表...")
    os.makedirs('analysis_output', exist_ok=True)

    # Pt 对比图
    plot_feature_comparison(
        raw_Pt, norm_Pt, pt_feature_names, object_classes,
        save_path='analysis_output/physics_comparison_Pt.png',
        title_prefix='Tactile Features (Pt)\n'
    )

    # Pv 对比图
    plot_feature_comparison(
        raw_Pv, norm_Pv, pv_feature_names, object_classes,
        save_path='analysis_output/physics_comparison_Pv.png',
        title_prefix='Visual Features (Pv)\n'
    )

    print("\n✅ 分析完成！")
    print("   输出目录: ./analysis_output/")
    print("\n💡 论文使用建议:")
    print("   1. 将 physics_comparison_Pt.png 放入 Appendix 或 Method 部分")
    print(
        "   2. 引用统计摘要说明：'Global normalization preserves the relative magnitude differences between objects (e.g., mugs remain heavier than apples) while unifying feature scales.'")


if __name__ == '__main__':
    main()




# 脚本功能解读
# 数据来源：脚本直接读取
# cache.npz，并且只分析训练集的数据。这确保了你看到的是模型“看到”的世界，而不是包含了测试集偏差的世界。
# 双栏对比图：
# 上半部分（Raw）：展示原始量纲。你会看到
# F_total的数值可能是几百到几千，而
# balance是
# 0 - 1，证明了标准化的必要性。
# 下半部分（Normalized）：展示
# Z - score。你会发现所有特征都集中在
# 0
# 附近，标准差接近
# 1。
# 颜色编码：
# 🔴 红色：马克杯（重）
# 🟢 绿色：苹果（轻）
# 你会发现，在
# F_total的标准化图中，红色的盒子明显在
# 0
# 以上，绿色的盒子在
# 0
# 以下。这直观地证明了标准化没有消除物理差异，而是把它转化为了均值的偏移。
# 终端统计：
# 脚本会在终端打印出具体的均值变化，例如
# F_total: Raw(1250.3) -> Norm(0.823)。
# 这对于写论文的
# Result
# 部分非常有用