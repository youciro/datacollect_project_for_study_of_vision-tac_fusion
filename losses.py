# -*- coding: utf-8 -*-
"""
多任务损失函数
================
L = w_succ × BCE(success_logit, label_bin)
  + w_fail × CE_weighted(failure_logits, label)

- success 头: 二分类（0=成功, 1=失败），BCE with logits
- failure 头: 5分类（含 success 类），CE，可加 class_weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import config


def compute_class_weights(class_counts, num_classes=None, smooth=1.0):
    """
    根据各类样本数计算 class_weights（用于 CrossEntropyLoss）。
    
    公式: w_i = (N_total / num_classes) / (n_i + smooth)
    
    Args:
        class_counts: list or array, 每类样本数, len=num_classes
        smooth: 平滑项，避免 n_i=0 时除零
        
    Returns:
        torch.Tensor shape=(num_classes,)
    """
    if num_classes is None:
        num_classes = config.NUM_CLASSES
    
    counts = np.asarray(class_counts, dtype=np.float64)
    N = counts.sum()
    if N == 0:
        return torch.ones(num_classes, dtype=torch.float32)
    
    weights = (N / num_classes) / (counts + smooth)
    return torch.tensor(weights, dtype=torch.float32)


class MultiTaskLoss(nn.Module):
    """
    多任务损失:
        L = w_s · BCE(success_logit, label_bin)
          + w_f · CE_weighted(failure_logits, label)
    
    Args:
        class_weights: (num_classes,) torch.Tensor 或 None
        w_success: success 头权重
        w_failure: failure 头权重
    """
    
    def __init__(self, class_weights=None,
                  w_success=None, w_failure=None):
        super().__init__()
        if w_success is None:
            w_success = config.LOSS_WEIGHT_SUCCESS
        if w_failure is None:
            w_failure = config.LOSS_WEIGHT_FAILURE
        
        self.w_success = w_success
        self.w_failure = w_failure
        
        # BCE: success 头
        self.bce = nn.BCEWithLogitsLoss()
        
        # CE: failure 头 (可加类别权重)
        if class_weights is not None:
            self.ce = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.ce = nn.CrossEntropyLoss()
    
    def forward(self, output, batch):
        """
        Args:
            output: dict from model
                - 'success_logit': (B,)
                - 'failure_logits': (B, num_classes)
            batch: dict from dataloader
                - 'label': (B,) long
                - 'label_bin': (B,) float
        Returns:
            dict:
                'loss': total loss (scalar)
                'loss_success': scalar
                'loss_failure': scalar
        """
        loss_success = self.bce(output['success_logit'], batch['label_bin'])
        loss_failure = self.ce(output['failure_logits'], batch['label'])
        
        total = self.w_success * loss_success + self.w_failure * loss_failure
        
        return {
            'loss':         total,
            'loss_success': loss_success.detach(),
            'loss_failure': loss_failure.detach(),
        }


# ============================================================ #
#  自测
# ============================================================ #

if __name__ == '__main__':
    print("=" * 60)
    print("losses.py 自测")
    print("=" * 60)
    
    # 模拟标签分布
    class_counts = [180, 48, 6, 6, 27]
    print(f"\n模拟 class_counts: {class_counts}")
    cw = compute_class_weights(class_counts)
    print(f"class_weights: {cw.tolist()}")
    print(f"  注: 小类 weight 大, 大类 weight 小")
    
    # 模拟一个 batch 的 forward
    B = 8
    num_classes = config.NUM_CLASSES
    output = {
        'success_logit':  torch.randn(B),
        'failure_logits': torch.randn(B, num_classes),
    }
    batch = {
        'label':     torch.randint(0, num_classes, (B,)),
        'label_bin': (torch.randint(0, num_classes, (B,)) > 0).float(),
    }
    
    criterion = MultiTaskLoss(class_weights=cw)
    losses = criterion(output, batch)
    print(f"\nloss 计算结果:")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")
    
    print("\n✅ losses.py 自测完成")
