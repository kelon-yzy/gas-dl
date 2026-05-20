# 项目开发指南

## 构建与运行

```powershell
# 安装依赖
pip install -r requirements.txt

# 深度训练（带 CLI 进度）
python src/pipeline/train_deep.py --config configs/deep/<配置文件>.yaml

# 深度训练（无 UI，适合后台运行）
python src/pipeline/train_deep.py --config configs/deep/<配置文件>.yaml --no-ui

# 训练曲线绘制
python src/pipeline/plot_deep_training_curves.py
```

## 代码结构（DL 链路）

```
src/dl/
├── models/
│   ├── registry.py            # 模型注册与构建
│   └── cnn1d_tcn_fusion.py    # CNN1D-TCN 多模态融合模型 (242K params)
├── training/
│   ├── orchestrator.py        # 训练主编排逻辑
│   ├── losses.py              # 损失函数（WeightedMSE/L1, UncertaintyWeightedLoss, SumConstraint）
│   ├── metrics.py             # 评估指标（RMSE, MAE, MAPE, R², sum偏差）
│   ├── runtime.py             # 训练/评估的单 epoch 逻辑
│   ├── checkpoints.py         # 检查点保存/恢复
│   ├── early_stopping.py      # 早停
│   ├── data_setup.py          # 数据加载与配置
│   └── seed.py                # 随机种子管理
└── data/
    └── split_utils.py         # 数据集划分逻辑
```

## Loss 函数说明

### 1. WeightedMSELoss / WeightedL1Loss

按列加权的 MSE/MAE，权重归一化到 mean=1。配合 `label_balanced_loss: true` 使用时，
权重自动计算为 `1/std²`（各组分标准差的倒数平方）。

**已知问题**：`1/std²` 加权导致 CH₄（std=10.19, weight=0.286）被严重低估，
因为 CH₄ 的权重仅为 CO₂（weight=2.332）的 1/8。

### 2. UncertaintyWeightedLoss（推荐）

基于 Kendall et al. (CVPR 2018) 的可学习不确定性加权。

**原理**：每个组分学习一个 `log_σ` 参数，loss = Σ [0.5·exp(-2·log_σ_i)·MSE_i + log_σ_i]

**配置**：
```yaml
training:
  label_balanced_loss: false  # 必须关闭旧加权
  uncertainty_weighted:
    num_tasks: 4              # 输出组分数
    init_log_sigma: 0.0       # 初始 σ=1，各组分等权
    lr_multiplier: 10.0       # σ参数学习率倍率
```

**特点**：
- 自动平衡各组分权重，无需手动调
- 正则项 `log_σ_i` 防止某组分被完全忽略
- σ 参数独立于模型参数，使用更高学习率快速收敛
- 训练日志中记录 `uw_sigmas` 和 `uw_weights` 监控权重变化
- Checkpoint 中保存 loss_fn 状态，支持断点续训

### 3. SumConstraintLoss

在 base_loss 基础上加 sum=100 惩罚项。可与上述任意 base_loss 组合使用。

## 实验配置说明

| 配置文件 | 说明 |
|---------|------|
| `slow_only_cnn1d_tcn_fusion_multimodal_formal.yaml` | 基线（1/std² 加权） |
| `uncertainty_weighted_cnn1d_tcn_fusion.yaml` | 不确定性加权（修复CH₄偏差） |

## 关键超参数

| 参数 | 基线值 | UW实验值 | 说明 |
|------|--------|---------|------|
| learning_rate | 0.0003 | 0.0002 | 降低以配合更稳定的 loss landscape |
| batch_size | 32 | 64 | 增大以减少梯度噪声 |
| warmup_epochs | 10 | 15 | 加长 warmup 让 σ 参数稳定 |
| eta_min | 3e-5 | 1e-5 | 降低最终学习率 |
| early_stopping_patience | 25 | 30 | 放宽容忍度 |
| sum_constraint.weight | 0.1 | 0.2 | 增强 sum 约束 |

## 文献参考

- **Uncertainty Weighting**: Kendall, A., Gal, Y., & Cipolla, R. (2018). Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics. CVPR.
- **GradNorm**: Chen, Z. et al. (2018). Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks. ICML.
- **DWA**: Liu, S. et al. (2019). End-To-End Multi-Task Learning With Attention. CVPR.
- **Deep Imbalanced Regression**: Yang, Y. et al. (2021). Delving into Deep Imbalanced Regression. ICML.
