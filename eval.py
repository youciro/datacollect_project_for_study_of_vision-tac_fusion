# -*- coding: utf-8 -*-
"""
评测脚本
================
运行: python eval.py --ckpt <path_to_checkpoint> [--split val|test]

输出:
  - 整体 acc / per-class P/R/F1
  - 5×5 混淆矩阵 (PNG + numpy)
  - ROC 曲线 (success vs failure)
  - 评测报告 (json)
"""

import os
import json
import argparse
from datetime import datetime

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_recall_fscore_support, roc_curve, auc,
)

import config
from dataset import build_dataloaders
from model import GraspFusionModel


# ============================================================ #
#  评测核心
# ============================================================ #

@torch.no_grad()
def run_inference(model, loader, device):
    """跑一遍模型，收集所有预测和标签"""
    model.eval()
    all_failure_logits = []
    all_success_probs = []
    all_labels = []
    all_label_bins = []
    all_wv = []
    all_wt = []
    all_cv = []
    all_ct = []
    
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                 for k, v in batch.items()}
        output = model(batch)
        
        all_failure_logits.append(output['failure_logits'].cpu().numpy())
        all_success_probs.append(torch.sigmoid(output['success_logit']).cpu().numpy())
        all_labels.append(batch['label'].cpu().numpy())
        all_label_bins.append(batch['label_bin'].cpu().numpy())
        all_wv.append(output['wv'].cpu().numpy())
        all_wt.append(output['wt'].cpu().numpy())
        all_cv.append(output['cv'].cpu().numpy())
        all_ct.append(output['ct'].cpu().numpy())
    
    return {
        'failure_logits': np.concatenate(all_failure_logits),
        'success_probs':  np.concatenate(all_success_probs),
        'labels':         np.concatenate(all_labels),
        'label_bins':     np.concatenate(all_label_bins),
        'wv':             np.concatenate(all_wv),
        'wt':             np.concatenate(all_wt),
        'cv':             np.concatenate(all_cv),
        'ct':             np.concatenate(all_ct),
    }


def plot_confusion_matrix(cm, class_names, save_path, title='Confusion Matrix'):
    """画混淆矩阵热图"""
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    
    # 在格子上写数字
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = 'white' if cm[i, j] > thresh else 'black'
            ax.text(j, i, str(int(cm[i, j])), ha='center', va='center',
                    color=color, fontsize=11)
    
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_roc(label_bins, success_probs, save_path):
    """画 ROC 曲线 (success vs failure 二分类)"""
    # 核心修正：label_bins里1=成功（正类），0=失败（负类）
    # success_probs是成功的概率，越高越可能是正类
    y_true = label_bins  # 直接用原始标签，1就是我们要的正类
    y_score = success_probs  # 正类（成功）的预测概率

    # 显式指定pos_label=1，告诉sklearn“标签为1的是正类”，彻底避免歧义
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC: Success vs Failure')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return roc_auc


# ============================================================ #
#  主流程
# ============================================================ #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True, help='checkpoint 路径')
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test'],
                        help='评测哪个 split')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录，默认为 ckpt 同级 eval_{timestamp}/')
    args = parser.parse_args()
    
    # 1. 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[设备] {device}")
    
    # 2. 数据
    print(f"[加载数据] split = {args.split}")
    train_loader, val_loader, test_loader, info = build_dataloaders()
    if args.split == 'val':
        loader = val_loader
        n = info['n_val']
    else:
        loader = test_loader
        n = info['n_test']
    
    if loader is None or n == 0:
        print(f"❌ {args.split} 集为空")
        return
    print(f"  样本数: {n}")
    
    # 3. 加载模型
    print(f"[加载模型] {args.ckpt}")
    model = GraspFusionModel().to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    print(f"  来自 epoch {ckpt.get('epoch', '?')}")
    
    # 4. 推理
    print(f"[推理]")
    result = run_inference(model, loader, device)
    
    # 5. 计算指标
    labels = result['labels']
    preds = result['failure_logits'].argmax(axis=1)
    
    overall_acc = (preds == labels).mean()
    print(f"\n  整体 acc: {overall_acc:.4f}")
    
    # success/failure 二分类
    succ_pred_bin = (result['success_probs'] >= 0.5).astype(np.int32)
    succ_acc = (succ_pred_bin == result['label_bins'].astype(np.int32)).mean()
    print(f"  二分类 succ_acc: {succ_acc:.4f}")
    
    # 5分类详细指标
    class_names = [config.INT2LABEL[i] for i in range(config.NUM_CLASSES)]
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, labels=list(range(config.NUM_CLASSES)),
        average=None, zero_division=0)
    
    print(f"\n  Per-class metrics:")
    print(f"  {'class':15s} {'precision':>10s} {'recall':>10s} {'f1':>10s} {'support':>8s}")
    for i, name in enumerate(class_names):
        print(f"  {name:15s} {precision[i]:>10.4f} {recall[i]:>10.4f} "
              f"{f1[i]:>10.4f} {int(support[i]):>8d}")
    
    macro_f1 = f1.mean()
    print(f"\n  Macro F1: {macro_f1:.4f}")
    
    # 6. 混淆矩阵
    cm = confusion_matrix(labels, preds, labels=list(range(config.NUM_CLASSES)))
    
    # 7. 输出目录
    if args.output_dir is None:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = os.path.join(os.path.dirname(args.ckpt),
                                         f'eval_{args.split}_{ts}')
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\n[输出目录] {args.output_dir}")
    
    # 8. 保存混淆矩阵
    cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(cm, class_names, cm_path,
                            title=f'Confusion Matrix ({args.split})')
    np.save(os.path.join(args.output_dir, 'confusion_matrix.npy'), cm)
    print(f"  保存混淆矩阵: {cm_path}")
    
    # 9. 保存 ROC
    roc_path = os.path.join(args.output_dir, 'roc_success_vs_failure.png')
    roc_auc = plot_roc(result['label_bins'], result['success_probs'], roc_path)
    print(f"  保存 ROC 曲线: {roc_path}  AUC={roc_auc:.4f}")
    
    # 10. 保存评测报告
    report = {
        'ckpt':             args.ckpt,
        'split':            args.split,
        'n_samples':        int(n),
        'overall_acc':      float(overall_acc),
        'succ_acc':         float(succ_acc),
        'macro_f1':         float(macro_f1),
        'roc_auc':          float(roc_auc),
        'per_class': {
            name: {
                'precision': float(precision[i]),
                'recall':    float(recall[i]),
                'f1':        float(f1[i]),
                'support':   int(support[i]),
            }
            for i, name in enumerate(class_names)
        },
        'confusion_matrix': cm.tolist(),
        'modality_weights': {
            'wv_mean':    float(result['wv'].mean()),
            'wv_std':     float(result['wv'].std()),
            'wt_mean':    float(result['wt'].mean()),
            'wt_std':     float(result['wt'].std()),
            'cv_mean':    float(result['cv'].mean()),
            'ct_mean':    float(result['ct'].mean()),
        }
    }
    report_path = os.path.join(args.output_dir, 'eval_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  保存评测报告: {report_path}")
    
    # 11. 保存预测明细
    np.savez(os.path.join(args.output_dir, 'predictions.npz'),
             labels=labels,
             preds=preds,
             failure_logits=result['failure_logits'],
             success_probs=result['success_probs'],
             label_bins=result['label_bins'],
             wv=result['wv'],
             wt=result['wt'],
             cv=result['cv'],
             ct=result['ct'])
    
    print(f"\n✅ 评测完成")
    print(f"   模态权重平均: wv={result['wv'].mean():.3f}, wt={result['wt'].mean():.3f}")


if __name__ == '__main__':
    main()
