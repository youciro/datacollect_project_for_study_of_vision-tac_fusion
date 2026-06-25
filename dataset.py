# -*- coding: utf-8 -*-
"""
PyTorch Dataset 和 DataLoader 模块
===================================
从 preprocess.py 生成的 cache.npz 加载数据，构造三个 DataLoader:
  - train (TRAIN_OBJECTS, 80%, 增强)
  - val   (TRAIN_OBJECTS, 20%, 无增强)
  - test  (TEST_OBJECTS,  全部, 无增强)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import config


# ============================================================ #
#  全局缓存（避免多次加载 cache）
# ============================================================ #

_CACHE = None


def load_cache(cache_path=None):
    """加载 cache.npz 到内存，全局只加载一次"""
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    
    if cache_path is None:
        cache_path = config.CACHE_PATH
    
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"cache 不存在: {cache_path}\n"
            f"请先运行 `python preprocess.py` 生成 cache"
        )
    
    data = np.load(cache_path, allow_pickle=True)
    _CACHE = {
        'sample_ids':     data['sample_ids'],
        'object_classes': data['object_classes'],
        'labels':         data['labels'],
        'rgbds':          data['rgbds'],          # (N, 4, 224, 224) float32
        'tactile_aggs':   data['tactile_aggs'],   # (N, 9, 15, 5) float32
        'Pts':            data['Pts'],            # (N, 6) float32
        'Pvs':            data['Pvs'],            # (N, 4) float32
    }
    return _CACHE


def load_or_compute_norm_stats(cache, train_idx, norm_stats_path):
    """
    加载或计算物理量（Pt/Pv）的标准化统计量（仅用训练集，无数据泄露）

    Args:
        cache: 加载的cache字典
        train_idx: 训练集索引（仅用这部分数据计算统计量）
        norm_stats_path: 统计量保存路径

    Returns:
        dict: 包含 pt_mean, pt_std, pv_mean, pv_std 的字典
    """
    # 如果已有保存的统计量，直接加载
    if os.path.exists(norm_stats_path):
        print(f"📊 加载已有标准化统计量: {os.path.basename(norm_stats_path)}")
        stats = np.load(norm_stats_path)
        return {
            'pt_mean': stats['pt_mean'],
            'pt_std': stats['pt_std'],
            'pv_mean': stats['pv_mean'],
            'pv_std': stats['pv_std'],
        }

    # 否则用训练集计算统计量
    print(f"📊 计算训练集物理量标准化统计量（基于 {len(train_idx)} 个训练样本）...")
    pt_train = cache['Pt'][train_idx]  # (n_train, 6)，仅训练集的触觉物理量
    pv_train = cache['Pv'][train_idx]  # (n_train, 4)，仅训练集的视觉物理量

    # 计算均值和标准差，给标准差加eps防止除零
    pt_mean = pt_train.mean(axis=0)
    pt_std = np.clip(pt_train.std(axis=0), 1e-6, None)
    pv_mean = pv_train.mean(axis=0)
    pv_std = np.clip(pv_train.std(axis=0), 1e-6, None)

    # 保存统计量
    os.makedirs(os.path.dirname(norm_stats_path), exist_ok=True)
    np.savez(norm_stats_path,
             pt_mean=pt_mean, pt_std=pt_std,
             pv_mean=pv_mean, pv_std=pv_std)
    print(f"   已保存统计量到: {norm_stats_path}")
    print(f"   Pt均值: {pt_mean.round(4)}")
    print(f"   Pt标准差: {pt_std.round(4)}")
    print(f"   Pv均值: {pv_mean.round(4)}")
    print(f"   Pv标准差: {pv_std.round(4)}")

    return {'pt_mean': pt_mean, 'pt_std': pt_std,
            'pv_mean': pv_mean, 'pv_std': pv_std}

# ============================================================ #
#  数据集划分
# ============================================================ #

def split_indices(cache):
    """
    根据物体名按 TRAIN_OBJECTS / TEST_OBJECTS 划分；
    再把 TRAIN_OBJECTS 内部的样本按 VAL_RATIO 随机分 train/val。
    
    Returns:
        (train_idx, val_idx, test_idx): 三个 np.ndarray
    """
    objs = cache['object_classes']
    
    train_mask = np.array([o in config.TRAIN_OBJECTS for o in objs])
    test_mask  = np.array([o in config.TEST_OBJECTS  for o in objs])
    
    train_pool = np.where(train_mask)[0]
    test_idx   = np.where(test_mask)[0]
    
    # 物体不在 TRAIN_OBJECTS 也不在 TEST_OBJECTS → 警告
    unclassified = np.where(~train_mask & ~test_mask)[0]
    if len(unclassified) > 0:
        unknown_objs = sorted(set(objs[unclassified]))
        print(f"⚠️ 有 {len(unclassified)} 个样本的物体类别既不在 TRAIN_OBJECTS 也不在 TEST_OBJECTS:")
        print(f"   {unknown_objs}")
        print(f"   这些样本将被忽略。请检查 config.py 中的列表。")
    
    # 训练池随机分 train/val
    rng = np.random.RandomState(config.SPLIT_SEED)
    perm = rng.permutation(train_pool)
    n_val = int(len(train_pool) * config.VAL_RATIO)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    
    return train_idx, val_idx, test_idx


# ============================================================ #
#  Dataset 类
# ============================================================ #

class GraspDataset(Dataset):
    """
    单次 __getitem__ 返回:
      'rgbd'      : (4, 224, 224) float32 tensor
      'tactile'   : (9, 15, 5)    float32 tensor
      'Pt'        : (6,)          float32 tensor
      'Pv'        : (4,)          float32 tensor
      'label'     : ()            int64   tensor
      'label_bin' : ()            float32 tensor (1.0=失败, 0.0=成功)
      'meta'      : dict (sample_id, object_class)  # 仅用于调试，DataLoader 默认不会聚合 dict
    """

    def __init__(self, indices, cache=None, augment=False, norm_stats=None):
        self.indices = np.asarray(indices)
        if cache is None:
            cache = load_cache()
        self.cache = cache
        self.augment = augment
        self.norm_stats = norm_stats  # 新增：标准化统计量

        # 数据增强（保持不变）
        if augment and config.AUG_ENABLED:
            self.color_jitter = transforms.ColorJitter(
                brightness=config.AUG_BRIGHTNESS,
                contrast=config.AUG_CONTRAST,
                saturation=config.AUG_SATURATION,
            )
        else:
            self.color_jitter = None

    def __getitem__(self, idx):
        i = self.indices[idx]
        c = self.cache

        rgbd = c['rgbds'][i].copy()
        tactile = c['tactile_aggs'][i].copy()
        Pt = c['Pt'][i].copy()  # (6,) 原始触觉物理量
        Pv = c['Pvs'][i].copy()  # (4,) 原始视觉物理量
        label = int(c['labels'][i])

        # -------------------------- #
        # 新增：物理量全局标准化
        # -------------------------- #
        if config.ENABLE_PHYSICAL_FEATURE_NORMALIZATION and self.norm_stats is not None:
            # Pt标准化：(原始值 - 训练集均值) / 训练集标准差
            Pt = (Pt - self.norm_stats['pt_mean']) / self.norm_stats['pt_std']
            # Pv标准化：同理
            Pv = (Pv - self.norm_stats['pv_mean']) / self.norm_stats['pv_std']

        # 数据增强（保持不变）
        if self.augment and config.AUG_ENABLED:
            rgbd = self._augment_rgbd(rgbd)

        # 转Tensor（保持不变）
        rgbd_t = torch.from_numpy(rgbd).float()
        tactile_t = torch.from_numpy(tactile).float()
        Pt_t = torch.from_numpy(Pt).float()
        Pv_t = torch.from_numpy(Pv).float()
        label_t = torch.tensor(label, dtype=torch.long)
        label_bin = torch.tensor(0.0 if label == 0 else 1.0, dtype=torch.float32)

        return {
            'rgbd': rgbd_t,
            'tactile': tactile_t,
            'Pt': Pt_t,
            'Pv': Pv_t,
            'label': label_t,
            'label_bin': label_bin,
        }
    
    def _augment_rgbd(self, rgbd):
        """
        对 4 通道 RGB-D 做增强:
          - RGB 通道做色彩抖动（用 torchvision）
          - 整体（含 depth）做随机旋转
        """
        # 拆 RGB 和 Depth
        rgb = rgbd[:3]      # (3, H, W) float32 [0,1]
        depth = rgbd[3:]    # (1, H, W) float32 [0,1]
        
        # --- 色彩抖动（只作用于 RGB） ---
        if self.color_jitter is not None:
            rgb_tensor = torch.from_numpy(rgb)
            rgb_tensor = self.color_jitter(rgb_tensor)
            rgb = rgb_tensor.numpy()
            rgb = np.clip(rgb, 0, 1)
        
        # --- 随机旋转（整体一起转） ---
        if config.AUG_ROTATION_DEG > 0:
            angle = (np.random.rand() * 2 - 1) * config.AUG_ROTATION_DEG
            full = np.concatenate([rgb, depth], axis=0)  # (4, H, W)
            full_t = torch.from_numpy(full).unsqueeze(0)  # (1, 4, H, W)
            full_t = transforms.functional.rotate(full_t, angle, fill=0.0)
            full = full_t.squeeze(0).numpy()
            rgb = full[:3]
            depth = full[3:]
        
        return np.concatenate([rgb, depth], axis=0)


# ============================================================ #
#  对外接口
# ============================================================ #

def build_dataloaders(batch_size=None, num_workers=None):
    """
    构造 train/val/test 三个 DataLoader。
    
    Returns:
        train_loader, val_loader, test_loader, info_dict
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    if num_workers is None:
        num_workers = config.NUM_WORKERS

    cache = load_cache()
    train_idx, val_idx, test_idx = split_indices(cache)

    # -------------------------- #
    # 新增：加载/计算标准化统计量
    # -------------------------- #
    norm_stats = None
    if config.ENABLE_PHYSICAL_FEATURE_NORMALIZATION:
        norm_stats = load_or_compute_norm_stats(
            cache=cache,
            train_idx=train_idx,
            norm_stats_path=config.NORM_STATS_PATH
        )

    # 把统计量传给所有Dataset实例（train/val/test都用同一套训练集统计量）
    train_set = GraspDataset(train_idx, cache=cache, augment=True, norm_stats=norm_stats)
    val_set = GraspDataset(val_idx, cache=cache, augment=False, norm_stats=norm_stats)
    test_set = GraspDataset(test_idx, cache=cache, augment=False, norm_stats=norm_stats)
    
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=config.PIN_MEMORY, drop_last=False)
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=config.PIN_MEMORY, drop_last=False)
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=config.PIN_MEMORY, drop_last=False) \
        if len(test_idx) > 0 else None
    
    # 标签分布统计（供 train.py 算 class_weights）
    labels_train = cache['labels'][train_idx]
    class_counts = np.bincount(labels_train, minlength=config.NUM_CLASSES)
    
    info = {
        'n_train':      len(train_idx),
        'n_val':        len(val_idx),
        'n_test':       len(test_idx),
        'class_counts': class_counts.tolist(),
    }
    
    return train_loader, val_loader, test_loader, info


# ============================================================ #
#  自测
# ============================================================ #

if __name__ == '__main__':
    print("=" * 60)
    print("dataset.py 自测")
    print("=" * 60)
    
    if not os.path.exists(config.CACHE_PATH):
        print(f"❌ cache 不存在: {config.CACHE_PATH}")
        print("请先运行 `python preprocess.py`")
        exit(0)
    
    train_loader, val_loader, test_loader, info = build_dataloaders()
    print(f"\n数据集统计:")
    print(f"  Train: {info['n_train']} 个 sample")
    print(f"  Val  : {info['n_val']} 个 sample")
    print(f"  Test : {info['n_test']} 个 sample")
    print(f"  各类样本数 (train): {info['class_counts']}")
    print(f"    " + ", ".join([f"{config.INT2LABEL[i]}={c}" 
                                for i, c in enumerate(info['class_counts'])]))
    
    # 取一个 batch 看 shape
    print(f"\n取一个 train batch 验证 shape:")
    for batch in train_loader:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k:12s}: shape={tuple(v.shape)}  dtype={v.dtype}")
        break
    
    print("\n✅ dataset.py 自测完成")

    # 标准化效果验证
    if config.ENABLE_PHYSICAL_FEATURE_NORMALIZATION:
        print("\n[标准化验证] 训练集前10个样本标准化后统计量:")
        sample_pt, sample_pv = [], []
        for i in range(min(10, len(train_set))):
            batch = train_set[i]
            sample_pt.append(batch['Pt'].numpy())
            sample_pv.append(batch['Pv'].numpy())
        sample_pt = np.stack(sample_pt)
        sample_pv = np.stack(sample_pv)
        print(f"   Pt均值: {sample_pt.mean(axis=0).round(4)}（理想：接近0）")
        print(f"   Pt标准差: {sample_pt.std(axis=0).round(4)}（理想：接近1）")
        print(f"   Pv均值: {sample_pv.mean(axis=0).round(4)}（理想：接近0）")
        print(f"   Pv标准差: {sample_pv.std(axis=0).round(4)}（理想：接近1）")
