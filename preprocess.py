# -*- coding: utf-8 -*-
"""
数据预处理脚本
================
遍历 data/ 下所有 sample_*** 文件夹，调用 features.py 提取所有特征，
保存到统一的 cache.npz 文件，供 dataset.py 直接读取。

运行: python preprocess.py
"""

import os
import json
import time
from glob import glob

import numpy as np
import cv2
from tqdm import tqdm

import config
import features


def load_whitelist(whitelist_path):
    """
    加载白名单文件。
    支持 JSON 格式: {"whitelist": [{"sample_id": 1}, ...]}
    支持 TXT 格式: 每行一个数字 ID
    """
    if not config.WHITELIST_ENABLED:
        print("白名单过滤已禁用 (WHITELIST_ENABLED=False)")
        return None

    if not os.path.exists(whitelist_path):
        print(f"白名单文件不存在: {whitelist_path}")
        print("   已启用白名单但文件未找到，将退出程序以避免误处理数据。")
        print("   如需处理所有数据，请将 config.WHITELIST_ENABLED 设为 False。")
        exit(1)  # 严格模式：文件不存在直接退出，防止误操作

    whitelist_ids = set()

    # 根据文件后缀判断格式
    if whitelist_path.endswith('.txt'):
        with open(whitelist_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    whitelist_ids.add(int(line))
    elif whitelist_path.endswith('.json'):
        with open(whitelist_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # 兼容两种常见的 JSON 结构
        if isinstance(data, list):  # [1, 2, 3]
            whitelist_ids = set(data)
        elif isinstance(data, dict) and 'whitelist' in data:  # {"whitelist": [{"sample_id": 1}]}
            for item in data['whitelist']:
                # 兼容直接是数字或者字典对象
                if isinstance(item, int):
                    whitelist_ids.add(item)
                elif isinstance(item, dict) and 'sample_id' in item:
                    whitelist_ids.add(item['sample_id'])

    if not whitelist_ids:
        print("白名单为空，没有样本将被处理。")
        return set()  # 返回空集合，后续逻辑会正确处理

    print(f"加载白名单成功: 共 {len(whitelist_ids)} 个样本")
    return whitelist_ids


# ============================================================ #
#  单 sample 处理
# ============================================================ #

def process_single_sample(sample_dir):
    """
    处理单个 sample 文件夹，返回特征字典。
    
    Returns:
        dict 或 None（如果数据有问题）
        包含字段:
            sample_id      : int
            object_class   : str
            label          : int  (0-4)
            rgbd           : (4, 224, 224) float32
            tactile_agg    : (9, 15, 5) float32 (mean/max/std聚合)
            Pt             : (6,) float32 触觉物理量
            Pv             : (4,) float32 视觉物理量
    """
    # --- 1. 读 meta.json ---
    meta_path = os.path.join(sample_dir, 'meta.json')
    if not os.path.exists(meta_path):
        return None, "meta.json 不存在"
    
    try:
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
    except Exception as e:
        return None, f"meta.json 读取失败: {e}"
    
    # 验证必要字段
    if 'result_label' not in meta:
        return None, "缺少 result_label"
    if 'object_class' not in meta:
        return None, "缺少 object_class"
    
    label_str = meta['result_label']
    if label_str not in config.LABEL2INT:
        return None, f"未知 label: {label_str}"
    label = config.LABEL2INT[label_str]
    
    # --- 2. 读触觉时序 + 时间戳 ---
    tac_path = os.path.join(sample_dir, 'tactile_seq.npy')
    ts_path = os.path.join(sample_dir, 'tactile_ts.npy')
    if not (os.path.exists(tac_path) and os.path.exists(ts_path)):
        return None, "触觉文件缺失"
    
    try:
        tactile_seq = np.load(tac_path)
        tactile_ts = np.load(ts_path)
    except Exception as e:
        return None, f"触觉文件读取失败: {e}"
    
    if tactile_seq.ndim != 4 or tactile_seq.shape[1:] != (3, 15, 5):
        return None, f"触觉 shape 异常: {tactile_seq.shape}"
    
    # --- 3. 截取触觉窗口 ---
    window = features.extract_tactile_window(tactile_seq, tactile_ts, meta)
    if window is None or len(window) < 3:
        return None, f"触觉窗口截取失败（锚点={config.WINDOW_ANCHOR}）"
    
    # --- 4. 触觉校准 + 聚合 + 物理量提取 ---
    window_cal = features.calibrate_tactile(window)
    tactile_agg = features.aggregate_tactile_window(window_cal)
    # Pt 从 mean 通道（前 3 个通道）提取
    Pt = features.extract_tactile_features(tactile_agg[:3])
    
    # --- 5. 读 RGB-D ---
    timing = config.VISION_TIMING   # 'before' or 'after'
    rgb_path = os.path.join(sample_dir, f'rgb_{timing}.png')
    depth_path = os.path.join(sample_dir, f'depth_{timing}.npy')
    
    if not (os.path.exists(rgb_path) and os.path.exists(depth_path)):
        return None, f"视觉文件缺失 ({timing})"
    
    rgb = cv2.imread(rgb_path)   # BGR, uint8
    if rgb is None:
        return None, "RGB 读取失败"
    
    try:
        depth = np.load(depth_path)
    except Exception as e:
        return None, f"Depth 读取失败: {e}"
    
    # --- 6. 视觉预处理 + 物理量提取 ---
    # 注：Pv 用原始分辨率算（更准确），RGBD 网络输入用 resize 后的
    Pv = features.extract_visual_features(rgb, depth)
    
    # depth 如果分辨率和 RGB 不一致，先 resize 到 RGB 分辨率
    if depth.shape[:2] != rgb.shape[:2]:
        depth = cv2.resize(depth.astype(np.float32),
                            (rgb.shape[1], rgb.shape[0]),
                            interpolation=cv2.INTER_NEAREST)
    
    rgbd = features.preprocess_vision(rgb, depth)
    
    # --- 7. 组装结果 ---
    return {
        'sample_id':    int(meta.get('sample_id', -1)),
        'object_class': str(meta['object_class']),
        'label':        label,
        'rgbd':         rgbd.astype(np.float32),
        'tactile_agg':  tactile_agg.astype(np.float32),
        'Pt':           Pt.astype(np.float32),
        'Pv':           Pv.astype(np.float32),
    }, None


# ============================================================ #
#  主流程
# ============================================================ #

def main():
    t0 = time.time()

    # 0. 加载白名单
    whitelist_ids = load_whitelist(config.WHITELIST_PATH)

    # 1. 扫描所有 sample 文件夹
    if not os.path.isdir(config.DATA_ROOT):
        print(f"❌ DATA_ROOT 不存在: {config.DATA_ROOT}")
        print(f"   请把采集的数据放到该目录下，结构: data/sample_00001/, sample_00002/, ...")
        return
    
    sample_dirs = sorted(glob(os.path.join(config.DATA_ROOT, 'sample_*')))
    sample_dirs = [d for d in sample_dirs if os.path.isdir(d)]
    
    print(f"\n找到 {len(sample_dirs)} 个 sample 文件夹")
    if len(sample_dirs) == 0:
        print("❌ 没有 sample 数据，预处理终止")
        return

    # 应用白名单过滤
    if whitelist_ids is not None:
        filtered_dirs = []
        skipped_count = 0

        for d in sample_dirs:
            folder_name = os.path.basename(d)
            try:
                # 提取数字部分 (sample_00074 -> 74)
                sample_id = int(folder_name.split('_')[1])
                if sample_id in whitelist_ids:
                    filtered_dirs.append(d)
                else:
                    skipped_count += 1
            except (IndexError, ValueError):
                print(f"⚠️ 文件夹命名格式异常，跳过: {folder_name}")
                skipped_count += 1

        print(f"白名单过滤结果:")
        print(f"   原始数量: {len(sample_dirs)}")
        print(f"   保留数量: {len(filtered_dirs)}")
        print(f"   跳过数量: {skipped_count}")

        sample_dirs = filtered_dirs

    if len(sample_dirs) == 0:
        print("❌ 没有符合条件的样本数据，预处理终止")
        return
    
    # 2. 逐个处理
    print(f"\n开始预处理...")
    print(f"  视觉时间点: {config.VISION_TIMING}")
    print(f"  触觉窗口: {config.WINDOW_ANCHOR} + {config.WINDOW_OFFSET_SEC}s ~ "
          f"{config.WINDOW_OFFSET_SEC + config.WINDOW_DURATION_SEC}s")
    print(f"  触觉聚合: {config.TACTILE_AGG_METHODS}")
    print(f"  激活点阈值: {config.ACTIVATION_THRESHOLD}")
    print()
    
    success_list = []
    skip_log = []
    
    for d in tqdm(sample_dirs, desc='预处理'):
        result, err = process_single_sample(d)
        if result is None:
            skip_log.append((os.path.basename(d), err))
        else:
            success_list.append(result)
    
    print(f"\n预处理完成: ✅成功 {len(success_list)}  ❌跳过 {len(skip_log)}")
    
    # 3. 打印跳过日志
    if skip_log:
        print(f"\n跳过的样本（前20个）:")
        for name, err in skip_log[:20]:
            print(f"  {name}: {err}")
        if len(skip_log) > 20:
            print(f"  ... 共 {len(skip_log)} 个被跳过")
    
    if len(success_list) == 0:
        print("\n❌ 没有有效样本，cache 不生成")
        return
    
    # 4. 统计标签 / 物体分布
    print(f"\n标签分布:")
    label_counts = {}
    object_counts = {}
    for s in success_list:
        lbl_name = config.INT2LABEL[s['label']]
        label_counts[lbl_name] = label_counts.get(lbl_name, 0) + 1
        object_counts[s['object_class']] = object_counts.get(s['object_class'], 0) + 1
    for k, v in sorted(label_counts.items()):
        pct = 100.0 * v / len(success_list)
        print(f"  {k:15s}: {v:4d}  ({pct:5.1f}%)")
    
    print(f"\n物体分布:")
    for k, v in sorted(object_counts.items()):
        in_train = '✓' if k in config.TRAIN_OBJECTS else ' '
        in_test  = '✓' if k in config.TEST_OBJECTS  else ' '
        print(f"  [train:{in_train}|test:{in_test}] {k:20s}: {v:4d}")
    
    # 5. 拼接成大数组
    print(f"\n打包 cache.npz...")
    sample_ids    = np.array([s['sample_id']    for s in success_list], dtype=np.int32)
    object_classes = np.array([s['object_class'] for s in success_list], dtype=object)
    labels        = np.array([s['label']        for s in success_list], dtype=np.int64)
    rgbds         = np.stack([s['rgbd']         for s in success_list])         # (N, 4, 224, 224)
    tactile_aggs  = np.stack([s['tactile_agg']  for s in success_list])         # (N, 9, 15, 5)
    Pts           = np.stack([s['Pt']           for s in success_list])         # (N, 6)
    Pvs           = np.stack([s['Pv']           for s in success_list])         # (N, 4)
    
    # 6. 保存
    os.makedirs(os.path.dirname(config.CACHE_PATH), exist_ok=True)
    np.savez(config.CACHE_PATH,
             sample_ids=sample_ids,
             object_classes=object_classes,
             labels=labels,
             rgbds=rgbds,
             tactile_aggs=tactile_aggs,
             Pts=Pts,
             Pvs=Pvs)
    
    cache_size_mb = os.path.getsize(config.CACHE_PATH) / 1024 / 1024
    print(f"\n✅ cache 已保存: {config.CACHE_PATH}")
    print(f"   样本数 : {len(success_list)}")
    print(f"   文件大小: {cache_size_mb:.1f} MB")
    print(f"   总耗时 : {time.time() - t0:.1f}s")
    
    # 7. 打印特征统计（供 review）
    print(f"\n触觉物理量 Pt 统计（按标签分组）:")
    print(f"  {'label':15s} {'F_total':>10s} {'balance':>10s} {'|c|mm':>10s} "
          f"{'contact_r':>10s} {'entropy':>10s}")
    for lbl_int in range(config.NUM_CLASSES):
        mask = labels == lbl_int
        if mask.sum() == 0:
            continue
        lbl_name = config.INT2LABEL[lbl_int]
        pts = Pts[mask]
        c_norm = np.sqrt(pts[:, 2] ** 2 + pts[:, 3] ** 2)
        print(f"  {lbl_name:15s} "
              f"{pts[:, 0].mean():>10.1f} {pts[:, 1].mean():>10.4f} "
              f"{c_norm.mean():>10.2f} "
              f"{pts[:, 4].mean():>10.4f} {pts[:, 5].mean():>10.4f}")
    
    print()


if __name__ == '__main__':
    main()
