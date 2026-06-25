# -*- coding: utf-8 -*-
"""
网络模型: 可解释置信度门控视触觉融合网络
============================================
论文创新点 1+2 的核心实现:

  视觉: RGB-D (4×224×224) → ResNet18(预训练, 4通道适配) → Fv(256)
                          → 视觉物理量 Pv(4)
  触觉: (9×15×5)          → 小 CNN                      → Ft(128)
                          → 触觉物理量 Pt(6)

  cv = MLP_v(Pv) ∈ [0,1]    (视觉置信度, 物理量驱动)
  ct = MLP_t(Pt) ∈ [0,1]    (触觉置信度, 物理量驱动)
  wv, wt = softmax(cv, ct)  (融合权重)

  Fv_aligned = Linear(Fv, FUSION_DIM)
  Ft_aligned = Linear(Ft, FUSION_DIM)
  F_fused = wv·Fv_aligned + wt·Ft_aligned

  输出头:
    success_head: F_fused → 64 → 1     (sigmoid, BCE 二分类)
    failure_head: F_fused → 64 → 5     (CE 5分类)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

import config


# ============================================================ #
#  视觉分支
# ============================================================ #

class VisualBranch(nn.Module):
    """
    ResNet18 主干（4通道输入适配）+ 物理量驱动的置信度 MLP
    
    输入:
        rgbd: (B, 4, 224, 224)
        Pv:   (B, 4)
    输出:
        Fv: (B, VISUAL_FEAT_DIM)
        cv: (B, 1)  ∈ [0,1]
    """
    
    def __init__(self,
                  feat_dim=None,
                  pretrained=None):
        super().__init__()
        if feat_dim is None:
            feat_dim = config.VISUAL_FEAT_DIM
        if pretrained is None:
            pretrained = config.USE_PRETRAINED_RESNET
        
        # --- ResNet18 主干 ---
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        
        # 替换第一层 conv：3通道 → 4通道
        # 前3通道复用预训练权重，第4通道（depth）初始化为前3通道的均值
        old_conv1 = backbone.conv1   # (64, 3, 7, 7)
        new_conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            if pretrained:
                new_conv1.weight[:, :3] = old_conv1.weight
                # 第4通道用前3通道权重的均值（让 depth 通道有合理初始化）
                new_conv1.weight[:, 3:4] = old_conv1.weight.mean(dim=1, keepdim=True)
            else:
                nn.init.kaiming_normal_(new_conv1.weight, mode='fan_out', nonlinearity='relu')
        backbone.conv1 = new_conv1
        
        # 移除最后的 fc 层，保留到 avgpool 之后的 512 维
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        # 投影到 feat_dim
        self.fc_feat = nn.Linear(512, feat_dim)
        
        # --- 视觉置信度 MLP ---
        self.conf_mlp = ConfidenceMLP(in_dim=config.PV_DIM)
    
    def forward(self, rgbd, Pv):
        # 提取视觉特征
        feat = self.backbone(rgbd)            # (B, 512, 1, 1)
        feat = feat.flatten(1)                # (B, 512)
        Fv = self.fc_feat(feat)               # (B, feat_dim)
        # 视觉置信度
        cv = self.conf_mlp(Pv)                # (B, 1)
        return Fv, cv


# ============================================================ #
#  触觉分支
# ============================================================ #

class TactileBranch(nn.Module):
    """
    小 CNN: 9 通道 → 128 维 + 物理量驱动的置信度 MLP
    
    输入尺寸 (15, 5) 较小，用 3 层 3×3 卷积 + 全局池化
    
    输入:
        tactile: (B, 9, 15, 5)
        Pt:      (B, 6)
    输出:
        Ft: (B, TACTILE_FEAT_DIM)
        ct: (B, 1)  ∈ [0,1]
    """
    
    def __init__(self,
                  in_channels=None,
                  feat_dim=None):
        super().__init__()
        if in_channels is None:
            in_channels = config.TACTILE_INPUT_CHANNELS
        if feat_dim is None:
            feat_dim = config.TACTILE_FEAT_DIM
        
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc_feat = nn.Linear(64, feat_dim)
        
        # --- 触觉置信度 MLP ---
        self.conf_mlp = ConfidenceMLP(in_dim=config.PT_DIM)
    
    def forward(self, tactile, Pt):
        feat = self.cnn(tactile)              # (B, 64, 1, 1)
        feat = feat.flatten(1)                # (B, 64)
        Ft = self.fc_feat(feat)               # (B, feat_dim)
        ct = self.conf_mlp(Pt)                # (B, 1)
        return Ft, ct


# ============================================================ #
#  置信度 MLP（视觉/触觉共用结构）
# ============================================================ #

class ConfidenceMLP(nn.Module):
    """
    物理量 → 标量置信度 ∈ [0,1]
    
    结构: in_dim → 16 → 8 → 1 + Sigmoid
    """
    def __init__(self, in_dim, hidden1=16, hidden2=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden1), nn.ReLU(inplace=True),
            nn.Linear(hidden1, hidden2), nn.ReLU(inplace=True),
            nn.Linear(hidden2, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        return self.net(x)


# ============================================================ #
#  融合 + 输出头
# ============================================================ #

class FusionHead(nn.Module):
    """
    置信度门控融合 + 多任务输出
    
    输入:
        Fv (B, visual_dim), Ft (B, tactile_dim), cv (B,1), ct (B,1)
    输出 dict:
        'success': (B, 1)    sigmoid
        'failure': (B, 5)    logits（不带 softmax）
        'wv':      (B,)      视觉权重
        'wt':      (B,)      触觉权重
        'F_fused': (B, FUSION_DIM)
    """
    
    def __init__(self,
                  visual_dim=None,
                  tactile_dim=None,
                  fusion_dim=None,
                  num_classes=None):
        super().__init__()
        if visual_dim is None:
            visual_dim = config.VISUAL_FEAT_DIM
        if tactile_dim is None:
            tactile_dim = config.TACTILE_FEAT_DIM
        if fusion_dim is None:
            fusion_dim = config.FUSION_DIM
        if num_classes is None:
            num_classes = config.NUM_CLASSES
        
        # 将视觉和触觉特征对齐到同一维度
        self.proj_v = nn.Linear(visual_dim, fusion_dim)
        self.proj_t = nn.Linear(tactile_dim, fusion_dim)
        
        # 输出头
        self.head_success = nn.Sequential(
            nn.Linear(fusion_dim, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
        self.head_failure = nn.Sequential(
            nn.Linear(fusion_dim, 64), nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )
    
    def forward(self, Fv, Ft, cv, ct):
        # softmax 门控
        w = F.softmax(torch.cat([cv, ct], dim=1), dim=1)  # (B, 2)
        wv = w[:, 0:1]    # (B, 1)
        wt = w[:, 1:2]    # (B, 1)
        
        # 融合
        Fv_aligned = self.proj_v(Fv)
        Ft_aligned = self.proj_t(Ft)
        F_fused = wv * Fv_aligned + wt * Ft_aligned   # (B, fusion_dim)
        
        # 输出头（logit）
        success_logit = self.head_success(F_fused).squeeze(1)   # (B,)
        failure_logits = self.head_failure(F_fused)             # (B, num_classes)
        
        return {
            'success_logit':  success_logit,
            'failure_logits': failure_logits,
            'wv':             wv.squeeze(1),
            'wt':             wt.squeeze(1),
            'cv':             cv.squeeze(1),
            'ct':             ct.squeeze(1),
            'F_fused':        F_fused,
        }


# ============================================================ #
#  顶层组合模型
# ============================================================ #

class GraspFusionModel(nn.Module):
    """
    完整融合模型
    """
    def __init__(self):
        super().__init__()
        self.visual_branch = VisualBranch()
        self.tactile_branch = TactileBranch()
        self.fusion_head = FusionHead()
    
    def forward(self, batch):
        """
        Args:
            batch: dict from DataLoader
                'rgbd'    : (B, 4, 224, 224)
                'tactile' : (B, 9, 15, 5)
                'Pv'      : (B, 4)
                'Pt'      : (B, 6)
        Returns:
            dict (来自 FusionHead.forward)
        """
        rgbd    = batch['rgbd']
        tactile = batch['tactile']
        Pv      = batch['Pv']
        Pt      = batch['Pt']
        
        Fv, cv = self.visual_branch(rgbd, Pv)
        Ft, ct = self.tactile_branch(tactile, Pt)
        out = self.fusion_head(Fv, Ft, cv, ct)
        return out


# ============================================================ #
#  自测
# ============================================================ #

if __name__ == '__main__':
    print("=" * 60)
    print("model.py 自测")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")
    
    model = GraspFusionModel().to(device)
    
    # 模拟一个 batch
    B = 4
    batch = {
        'rgbd':    torch.randn(B, 4, 224, 224, device=device),
        'tactile': torch.randn(B, 9, 15, 5, device=device),
        'Pv':      torch.randn(B, 4, device=device),
        'Pt':      torch.randn(B, 6, device=device),
    }
    
    with torch.no_grad():
        out = model(batch)
    
    print("\n输出 shapes:")
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:18s}: {tuple(v.shape)}  dtype={v.dtype}")
    
    # 验证 softmax 权重和为 1
    w_sum = (out['wv'] + out['wt']).mean().item()
    print(f"\n  权重和 (应为 1.0): {w_sum:.6f}")
    assert abs(w_sum - 1.0) < 1e-5
    
    # 参数量统计
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n参数量: 总 {n_params/1e6:.2f}M (可训练 {n_trainable/1e6:.2f}M)")
    
    # 各分支参数量
    for name, mod in [('视觉分支', model.visual_branch),
                       ('触觉分支', model.tactile_branch),
                       ('融合+输出', model.fusion_head)]:
        n = sum(p.numel() for p in mod.parameters())
        print(f"  {name:10s}: {n/1e6:.2f}M")
    
    print("\n✅ model.py 自测完成")
