# 训练端代码总览

## 一、文件清单

| 文件 | 作用 |
|---|---|
| `config.py` | 全局配置（路径、超参、阈值） |
| `features.py` | 触觉/视觉物理量计算函数 |
| `preprocess.py` | 遍历 sample 文件夹 → 生成 cache.npz |
| `dataset.py` | PyTorch Dataset + DataLoader（含数据增强） |
| `model.py` | 网络结构（视觉+触觉+置信度门控融合+输出头） |
| `losses.py` | 多任务损失（BCE + 加权CE） |
| `train.py` | 训练主脚本（早停、CosineLR、TensorBoard） |
| `eval.py` | 评测脚本（混淆矩阵、ROC、per-class metrics） |
| `visualize.py` | 可视化脚本（论文 Fig.5 风格） |
| `verify_env.py` | 环境验证脚本 |
| `requirements.txt` | 依赖列表 |
| `可调参数清单.md` | 所有可能要回头调的参数说明 |

## 二、目录结构

```
/home/chenyou/Aubo/
├── config.py
├── features.py
├── preprocess.py
├── dataset.py
├── model.py
├── losses.py
├── train.py
├── eval.py
├── visualize.py
├── verify_env.py
├── requirements.txt
├── 可调参数清单.md
├── data/
│   ├── sample_00001/         # ← 采集端传过来
│   │   ├── meta.json
│   │   ├── rgb_before.png
│   │   ├── rgb_after.png
│   │   ├── depth_before.npy
│   │   ├── depth_after.npy
│   │   ├── tactile_seq.npy
│   │   └── tactile_ts.npy
│   ├── sample_00002/
│   ├── ...
│   └── cache.npz             # ← preprocess.py 生成
└── train/
    └── runs/
        └── 20260615_140000/  # ← train.py 自动生成
            ├── checkpoints/
            │   ├── best_epoch020_acc0.8500.pth
            │   ├── epoch010.pth
            │   └── last.pth
            ├── tensorboard/
            └── train_config.json
```

## 三、完整运行流程

### 步骤 0：环境验证（首次运行前）
```bash
conda activate aubo_train
cd /home/chenyou/Aubo
python verify_env.py
```

### 步骤 1：把采集的 sample 文件夹放到 `data/` 下
（手动操作，从采集机传过来）

### 步骤 2：预处理 → 生成 cache.npz
```bash
python preprocess.py
```
输出会打印：
- 处理成功 / 跳过的样本数
- 标签分布、物体分布
- 各类物理量 Pt 的统计（对照 Pilot 数据看是否合理）

### 步骤 3：训练
```bash
python train.py
```
可选参数：
```bash
python train.py --epochs 100 --batch_size 64 --lr 5e-4
python train.py --resume train/runs/20260615_140000/checkpoints/last.pth
```

实时监控 TensorBoard：
```bash
tensorboard --logdir train/runs/
# 浏览器访问 http://localhost:6006
```

### 步骤 4：评测
```bash
python eval.py --ckpt train/runs/20260615_140000/checkpoints/best_epoch020_acc0.8500.pth --split val
python eval.py --ckpt train/runs/20260615_140000/checkpoints/best_epoch020_acc0.8500.pth --split test
```
会输出：
- `eval_report.json` —— 所有指标
- `confusion_matrix.png` —— 5×5 混淆矩阵
- `roc_success_vs_failure.png` —— ROC 曲线
- `predictions.npz` —— 所有预测明细（供后续可视化用）

### 步骤 5：可视化（论文 Fig.5）
```bash
# 单样本综合图
python visualize.py --ckpt <ckpt_path> --sample data/sample_00050

# 模态权重分布图（先跑完 eval.py）
python visualize.py --ckpt <ckpt_path> --predictions <eval_dir>/predictions.npz
```

## 四、训练监控关注什么

### TensorBoard 重点观察
- `train/loss` 和 `val/loss` —— 是否还在下降，是否过拟合（train↓ val↑）
- `val/acc` —— 主要指标
- `val/succ_acc` —— 二分类（成功/失败）准确率
- `train/loss_success` 和 `train/loss_failure` —— 两个头哪个更难收敛

### 训练终止条件
1. 达到最大 epoch (50)
2. 早停触发（val loss 连续 10 个 epoch 不下降）

### Checkpoint 保留
- **best** —— 验证集 acc 最高的前 3 个
- **periodic** —— 每 5 epoch 一个
- **last** —— 始终是最后一个 epoch

## 五、常见调整场景

详见《可调参数清单.md》，几个常用：

| 现象 | 改 config.py 哪个变量 |
|---|---|
| 训练 loss 不下降 / 震荡 | `LEARNING_RATE` 降到 `5e-4` 或 `1e-4` |
| 显存不够 | `BATCH_SIZE` 降到 16 或 8 |
| 小类（too_shallow 等）召回低 | 调整 `LOSS_WEIGHT_FAILURE` 或合并类别 |
| 触觉物理量区分度低 | 调 `ACTIVATION_THRESHOLD`（默认 50，可降到 20-30）|
| 想换视觉时间点 | `VISION_TIMING = 'after'`（需重跑 preprocess） |

## 六、与采集端的对接

| 字段 | 类型 | 训练端用途 |
|---|---|---|
| `meta.json -> result_label` | str (5个之一) | 主标签 |
| `meta.json -> object_class` | str (英文) | 数据集划分（必须在 TRAIN_OBJECTS 或 TEST_OBJECTS） |
| `meta.json -> timestamps.lifted` | float (s) | 触觉窗口锚点 |
| `tactile_seq.npy` | (N, 3, 15, 5) | 触觉时序 |
| `tactile_ts.npy` | (N,) | 每帧时间戳 |
| `rgb_before.png` / `depth_before.npy` | RGB + 深度图 mm | 视觉输入 |

⚠️ **如果 `meta.json` 字段名或类型变化，需要同步修改 `preprocess.py`**
