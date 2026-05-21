# 项目开发指南

## 快速开始

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

完整命令参考见 [docs/服务器命令大全.md](docs/服务器命令大全.md)。

## 代码结构（DL 链路）

```
src/dl/
├── models/
│   ├── registry.py            # 模型注册与构建
│   ├── cnn1d_tcn_fusion.py    # CNN1D-TCN 多模态融合模型 (242K params)
│   └── cnn1d_tcn_fusion_slow_branch.py  # slow_branch 改造版 + bounded_simplex 参数化
├── training/
│   ├── orchestrator.py        # 训练主编排逻辑
│   ├── losses.py              # 损失函数（见下文 Loss 函数说明）
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

### 2. UncertaintyWeightedLoss

基于 Kendall et al. (CVPR 2018) 的可学习不确定性加权。

**原理**：每个组分学习一个 `log_σ` 参数，loss = Σ [0.5·exp(-2·log_σ_i)·MSE_i + log_σ_i]

**配置**：
```yaml
training:
  label_balanced_loss: false  # 必须关闭旧加权
  uncertainty_weighted:
    num_tasks: 4              # 输出组分数（3-task 时设为 3）
    init_log_sigma: 0.0       # 初始 σ=1，各组分等权
    lr_multiplier: 2.0        # σ参数学习率倍率（建议 2.0，旧值 10.0 过大）
    sigma_clamp: [-1.0, 1.5]  # 限制 log_sigma 范围，防止权重崩塌
```

**特点**：
- 自动平衡各组分权重，无需手动调
- 正则项 `log_σ_i` 防止权重塌缩（σ→∞ 使某组分loss为0）
- sigma_clamp 限制最大权重比 ≤ e^(2*1.5+2*1.0) = e^5 ≈ 148
- 训练日志中记录 `uw_sigmas` 和 `uw_weights` 监控权重变化
- Checkpoint 中保存 loss_fn 状态，支持断点续训

**已知问题**：4-task UW 下 CH₄/N₂ 权重崩塌（CO₂:N₂ = 111:1），建议配合
`loss_columns: 3` 使用 3-task UW。

### 3. SlicedLoss (3-task loss)

只对前 N 列计算 loss，忽略剩余列（派生组分不参与梯度计算）。

```yaml
training:
  loss_columns: 3  # 只对 H2/CH4/CO2 计算 loss，N2 无梯度
```

**设计动机**：`derive_last: true` 下 N2 = 100 - H2 - CH4 - CO2，
N2 loss 梯度会反传到前三项，导致 CH4-N2 误差高度耦合 (corr=-0.93)。
去掉 N2 loss 后，三个自由头独立优化，N2 精度由前三项隐式决定。

### 4. SumConstraintLoss

在 base_loss 基础上加 sum=100 惩罚项。可与上述任意 base_loss 组合使用。
使用 `bounded_simplex` 参数化时不需要此项（已硬保证 sum=100）。

## 模型输出参数化

### derive_last + bounded_simplex

```yaml
model:
  derive_last: true
  derive_last_mode: bounded_simplex
  output_prior: [9.288469, 75.755157, 4.994778, 9.961745]  # 训练集标签均值
```

- 模型预测 H2/CH4/CO2 的比例（softmax）和总量（sigmoid），N2 = 100 - sum(三者)
- 保证所有组分非负且 sum 严格 = 100
- `output_prior` 用于初始化输出偏置，使初始预测接近训练集均值

## 实验配置说明

### 当前消融实验矩阵（bounded_simplex 系列）

| 配置文件 | 架构 | Loss | 关键参数 |
|---------|------|------|---------|
| `bounded_simplex_3task_loss_slow_branch.yaml` | TCN | 3-task 等权 MSE | `loss_columns: 3` |
| `bounded_simplex_3task_uw_slow_branch.yaml` | TCN | 3-task UW | `loss_columns: 3`, UW `num_tasks: 3` |
| `bounded_simplex_equal_mse_slow_branch.yaml` | TCN | 4-task 等权 MSE | — |
| `bounded_simplex_equal_mse_slow_branch_lstm.yaml` | LSTM | 3-task 等权 MSE | `loss_columns: 3` |
| `bounded_simplex_3task_uw_slow_branch_lstm.yaml` | LSTM | 3-task UW | `loss_columns: 3`, UW `num_tasks: 3` |

### 历史实验配置

| 配置文件 | 说明 |
|---------|------|
| `slow_only_cnn1d_tcn_fusion_multimodal_formal.yaml` | 基线（1/std² 加权） |
| `uncertainty_weighted_cnn1d_tcn_fusion.yaml` | UW 加权（无 slow_branch） |
| `uw_slow_branch_cnn1d_tcn_fusion.yaml` | UW + slow_branch（N2 崩塌） |
| `uw_clamped_slow_branch_cnn1d_tcn_fusion.yaml` | UW + sigma_clamp + bounded_simplex |
| `slow_branch_cnn1d_tcn_fusion.yaml` | slow_branch 等权 MSE 基线 |

### 已有实验结果对比

| 实验 | macro RMSE | CH4 R² | N2 R² | 备注 |
|------|-----------|--------|-------|------|
| UW 4-task 无截断 | 3.030 | 0.689 | -0.000 | N2 完全崩塌 |
| UW 4-task + sigma_clamp + bounded | 2.212 | 0.866 | 0.587 | 修复大部分崩塌 |
| slow_branch 等权 MSE | 2.082 | 0.861 | 0.632 | 等权基线 |
| bounded 4-task 等权 MSE | 2.167 | 0.874 | 0.615 | bounded_simplex 持平 |
| tcn_slow | 1.112 | 0.983 | 0.932 | 当前最优之一 |
| tcn_multimodal | 0.979 | 0.987 | 0.955 | 当前最优 |

## 关键超参数

| 参数 | 当前推荐值 | 说明 |
|------|----------|------|
| learning_rate | 0.0002 | |
| batch_size | 128 | |
| warmup_epochs | 15 | cosine_warmup 预热 |
| eta_min | 1e-5 | cosine 最终学习率 |
| early_stopping_patience | 25-30 | 等权用 30，UW 用 25 |
| grad_clip_norm | 1.0 | 梯度裁剪 |

## 已知问题

### torch.compile 与 bounded_simplex 不兼容

**状态**：未解决，所有 bounded_simplex 配置必须设 `compile: false`。

**现象**：启用 `torch.compile(mode="default")` 后，bounded_simplex 参数化（softmax + sigmoid）
的模型梯度完全失效——train_loss 在整个训练过程中不下降（锁死在初始值），所有 R²=0。

**根因**：Inductor 后端在编译 bounded_simplex 的 `F.softmax` + `torch.sigmoid` + 乘法链时
梯度传播断裂。可能与 `@torch.compiler.disable` 图断点 + AMP autocast 的交互有关。

**临时方案**：所有使用 `derive_last_mode: bounded_simplex` 的配置中加 `compile: false`。
对训练速度影响约 10-15%（单 epoch 慢 1-2 秒），但保证梯度正确。

### UW + warmup 的 sigma 膨胀

**状态**：已修复（orchestrator.py 中自动冻结 sigma 参数至 warmup 结束）。

**机制**：warmup 期间模型 lr 极低（~1e-5），sigma 参数以更高 lr 膨胀，
`exp(-2·log_σ)` 衰减压制模型梯度信号，导致模型卡在初始化状态。
修复后 warmup 期间 sigma 冻结，等模型建立基础表征后再解冻。

## 文献参考

- **Uncertainty Weighting**: Kendall, A., Gal, Y., & Cipolla, R. (2018). Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics. CVPR.
- **GradNorm**: Chen, Z. et al. (2018). Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks. ICML.
- **DWA**: Liu, S. et al. (2019). End-To-End Multi-Task Learning With Attention. CVPR.
- **Deep Imbalanced Regression**: Yang, Y. et al. (2021). Delving into Deep Imbalanced Regression. ICML.
