# -*- coding: utf-8 -*-
"""
训练主脚本
================
运行: python train.py

功能:
  - 加载 cache.npz 构造 dataloader
  - 实例化 GraspFusionModel
  - 多任务训练: success(BCE) + failure(CE_weighted)
  - 早停 + CosineAnnealingLR
  - TensorBoard 日志
  - Checkpoint: best top-K + last
"""

import os
import json
import time
import random
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import config
from dataset import build_dataloaders
from model import GraspFusionModel
from losses import MultiTaskLoss, compute_class_weights


# ============================================================ #
#  辅助函数
# ============================================================ #

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_run_dir(root):
    """创建本次训练的子目录: train/runs/YYYYMMDD_HHMMSS/"""
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(root, 'runs', ts)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_checkpoint(model, optimizer, epoch, val_metrics, path):
    """保存 checkpoint"""
    ckpt = {
        'epoch':       epoch,
        'model_state': model.state_dict(),
        'optim_state': optimizer.state_dict(),
        'val_metrics': val_metrics,
    }
    torch.save(ckpt, path)


def cleanup_old_checkpoints(ckpt_dir, keep_n):
    """保留 best_*.pth 中 val_acc 最高的 keep_n 个"""
    ckpts = [f for f in os.listdir(ckpt_dir) if f.startswith('best_') and f.endswith('.pth')]
    if len(ckpts) <= keep_n:
        return
    # 文件名格式: best_epoch{E}_acc{A:.4f}.pth
    def get_acc(name):
        try:
            return float(name.split('_acc')[-1].replace('.pth', ''))
        except Exception:
            return 0.0
    ckpts_sorted = sorted(ckpts, key=get_acc, reverse=True)
    for old in ckpts_sorted[keep_n:]:
        os.remove(os.path.join(ckpt_dir, old))


# ============================================================ #
#  Epoch 训练 / 验证
# ============================================================ #

def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    
    losses = []
    losses_succ = []
    losses_fail = []
    n_correct = 0
    n_total = 0
    
    pbar = tqdm(loader, desc=f'Epoch {epoch} [train]', leave=False)
    for batch in pbar:
        batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                 for k, v in batch.items()}
        
        optimizer.zero_grad()
        output = model(batch)
        loss_dict = criterion(output, batch)
        loss = loss_dict['loss']
        loss.backward()
        # 梯度裁剪，防止小数据集训练初期梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        losses.append(loss.item())
        losses_succ.append(loss_dict['loss_success'].item())
        losses_fail.append(loss_dict['loss_failure'].item())
        
        # 5 分类准确率
        preds = output['failure_logits'].argmax(dim=1)
        n_correct += (preds == batch['label']).sum().item()
        n_total += batch['label'].size(0)
        
        pbar.set_postfix(loss=f'{np.mean(losses):.4f}',
                          acc=f'{n_correct/max(n_total,1):.4f}')
    
    return {
        'loss':         float(np.mean(losses)),
        'loss_success': float(np.mean(losses_succ)),
        'loss_failure': float(np.mean(losses_fail)),
        'acc':          n_correct / max(n_total, 1),
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device, desc='val'):
    model.eval()
    
    losses = []
    all_preds = []
    all_labels = []
    all_success_probs = []
    all_label_bins = []
    
    for batch in tqdm(loader, desc=desc, leave=False):
        batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                 for k, v in batch.items()}
        
        output = model(batch)
        loss_dict = criterion(output, batch)
        losses.append(loss_dict['loss'].item())
        
        preds = output['failure_logits'].argmax(dim=1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(batch['label'].cpu().numpy())
        all_success_probs.append(torch.sigmoid(output['success_logit']).cpu().numpy())
        all_label_bins.append(batch['label_bin'].cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_success_probs = np.concatenate(all_success_probs)
    all_label_bins = np.concatenate(all_label_bins)
    
    acc = (all_preds == all_labels).mean()
    
    # success 二分类 acc（基于 sigmoid 阈值 0.5）
    succ_pred = (all_success_probs >= 0.5).astype(np.int32)
    succ_acc = (succ_pred == all_label_bins.astype(np.int32)).mean()
    
    return {
        'loss':     float(np.mean(losses)),
        'acc':      float(acc),
        'succ_acc': float(succ_acc),
        'preds':    all_preds,
        'labels':   all_labels,
    }


# ============================================================ #
#  主流程
# ============================================================ #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=config.EPOCHS)
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE)
    parser.add_argument('--resume', type=str, default=None,
                        help='指定 checkpoint 路径以恢复训练')
    args = parser.parse_args()
    
    # 1. 随机种子
    set_seed(config.GLOBAL_SEED)
    
    # 2. 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[设备] {device}")
    if device == 'cuda':
        print(f"[GPU] {torch.cuda.get_device_name(0)}")
    
    # 3. 数据
    print("\n[加载数据]")
    train_loader, val_loader, test_loader, info = build_dataloaders(
        batch_size=args.batch_size)
    print(f"  Train: {info['n_train']}  Val: {info['n_val']}  Test: {info['n_test']}")
    print(f"  类别样本数: {info['class_counts']}")
    
    # 4. 模型
    print("\n[构建模型]")
    model = GraspFusionModel().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {n_params/1e6:.2f}M")
    
    # 5. 损失 + class weights
    if config.USE_CLASS_WEIGHTS:
        class_weights = compute_class_weights(info['class_counts']).to(device)
        print(f"  使用 class_weights: {class_weights.tolist()}")
    else:
        class_weights = None
    criterion = MultiTaskLoss(class_weights=class_weights).to(device)
    
    # 6. 优化器 + scheduler
    optimizer = torch.optim.AdamW(model.parameters(),
                                    lr=args.lr,
                                    weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.LR_SCHEDULER_T_MAX)
    
    # 7. 输出目录
    run_dir = make_run_dir(config.TRAIN_OUTPUT_ROOT)
    ckpt_dir = os.path.join(run_dir, 'checkpoints')
    tb_dir = os.path.join(run_dir, 'tensorboard')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir)
    print(f"\n[输出目录] {run_dir}")
    
    # 保存训练配置
    with open(os.path.join(run_dir, 'train_config.json'), 'w') as f:
        cfg = {k: getattr(config, k) for k in dir(config) if k.isupper()}
        json.dump({k: (v if isinstance(v, (str, int, float, bool, list)) else str(v))
                   for k, v in cfg.items()}, f, indent=2, ensure_ascii=False)
    
    # 8. 恢复
    start_epoch = 1
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optim_state'])
        start_epoch = ckpt['epoch'] + 1
        print(f"\n[恢复] 从 {args.resume} 恢复，起始 epoch={start_epoch}")
    
    # 9. 训练循环
    print(f"\n[训练] 共 {args.epochs} epoch, 早停 patience={config.EARLY_STOP_PATIENCE}\n")
    
    best_val_loss = float('inf')
    best_val_acc = 0.0
    early_stop_counter = 0
    
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        
        train_metrics = train_one_epoch(model, train_loader, criterion,
                                          optimizer, device, epoch)
        val_metrics = evaluate(model, val_loader, criterion, device, desc=f'Epoch {epoch} [val]')
        scheduler.step()
        
        # 打印
        lr = optimizer.param_groups[0]['lr']
        dt = time.time() - t0
        print(f"Epoch {epoch:3d}/{args.epochs} ({dt:.1f}s)  "
              f"lr={lr:.2e}  "
              f"train: loss={train_metrics['loss']:.4f} acc={train_metrics['acc']:.4f}  "
              f"val: loss={val_metrics['loss']:.4f} acc={val_metrics['acc']:.4f} "
              f"succ_acc={val_metrics['succ_acc']:.4f}")
        
        # TensorBoard
        writer.add_scalar('train/loss',         train_metrics['loss'], epoch)
        writer.add_scalar('train/loss_success', train_metrics['loss_success'], epoch)
        writer.add_scalar('train/loss_failure', train_metrics['loss_failure'], epoch)
        writer.add_scalar('train/acc',          train_metrics['acc'], epoch)
        writer.add_scalar('val/loss',           val_metrics['loss'], epoch)
        writer.add_scalar('val/acc',            val_metrics['acc'], epoch)
        writer.add_scalar('val/succ_acc',       val_metrics['succ_acc'], epoch)
        writer.add_scalar('lr',                 lr, epoch)
        
        # Checkpoint: best
        if val_metrics['acc'] > best_val_acc:
            best_val_acc = val_metrics['acc']
            best_ckpt_path = os.path.join(ckpt_dir,
                f'best_epoch{epoch:03d}_acc{best_val_acc:.4f}.pth')
            save_checkpoint(model, optimizer, epoch, val_metrics, best_ckpt_path)
            cleanup_old_checkpoints(ckpt_dir, config.KEEP_BEST_N)
        
        # Checkpoint: 周期性 + last
        if epoch % config.SAVE_EVERY_N_EPOCHS == 0:
            periodic_path = os.path.join(ckpt_dir, f'epoch{epoch:03d}.pth')
            save_checkpoint(model, optimizer, epoch, val_metrics, periodic_path)
        if config.SAVE_LAST:
            last_path = os.path.join(ckpt_dir, 'last.pth')
            save_checkpoint(model, optimizer, epoch, val_metrics, last_path)
        
        # 早停
        if config.EARLY_STOP_ENABLED:
            if val_metrics['loss'] < best_val_loss - config.EARLY_STOP_MIN_DELTA:
                best_val_loss = val_metrics['loss']
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= config.EARLY_STOP_PATIENCE:
                    print(f"\n[早停] val loss 连续 {config.EARLY_STOP_PATIENCE} 个 epoch 未改善，停止训练")
                    break
    
    writer.close()
    
    # 10. 训练结束 - 在测试集上评测一次
    if test_loader is not None and len(test_loader) > 0:
        print(f"\n[测试集评测]")
        # 用 best checkpoint
        best_files = sorted([f for f in os.listdir(ckpt_dir) if f.startswith('best_')],
                             key=lambda x: float(x.split('_acc')[-1].replace('.pth', '')),
                             reverse=True)
        if best_files:
            best_path = os.path.join(ckpt_dir, best_files[0])
            print(f"  加载 best: {best_files[0]}")
            ckpt = torch.load(best_path, map_location=device)
            model.load_state_dict(ckpt['model_state'])
        
        test_metrics = evaluate(model, test_loader, criterion, device, desc='test')
        print(f"  Test: loss={test_metrics['loss']:.4f}  acc={test_metrics['acc']:.4f}  "
              f"succ_acc={test_metrics['succ_acc']:.4f}")
    
    print(f"\n[完成] 最优 val acc = {best_val_acc:.4f}")
    print(f"[输出目录] {run_dir}")


if __name__ == '__main__':
    main()
