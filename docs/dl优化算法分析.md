# 深度学习优化算法分析与改进方案

> 基于 d2l 第十一章「优化算法」理论与项目代码现状分析
> 创建日期：2026-05-16

## 1. 当前优化配置现状

### 1.1 优化器

所有模型统一使用 `torch.optim.Adam`，硬编码在 `train.py:611-614`：

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=float(config["training"].get("learning_rate", 1e-3)),
    weight_decay=float(config["training"].get("weight_decay", 0.0)),
)
```

| 配置项           | 默认值  | 各模型实际值                                            |
| ------------- | ---- | ------------------------------------------------- |
| optimizer     | Adam | Adam（固定）                                          |
| learning_rate | 1e-3 | 0.001（全部）                                         |
| weight_decay  | 0.0  | 纯慢变量 0.0001，多模态 0.0001，MultimodalFusionV3+AMP 0.0 |

### 1.2 学习率调度

**项目当前无任何学习率调度器**。`train.py` 中不含 `scheduler`、`lr_scheduler`、`ReduceLROnPlateau`、`CosineAnnealing` 等任何相关代码。学习率在整个训练过程中恒为 0.001。

### 1.3 Early Stopping

使用固定 patience=25 的 EarlyStopping，监控 val_macro_RMSE，无最小变化量阈值（min_delta）。

### 1.4 损失函数

全部使用 MSE，可选附加 sum_constraint 正则化（默认关闭）。

## 2. 基于已跑实验的问题诊断

### 2.1 CNN1D_multimodal 训练曲线分析

从 `v3_cnn1d_multimodal_seed42` 的 train_log 可以看到典型问题：

| 阶段   | Epoch | val_RMSE   | 现象                   |
| ---- | ----- | ---------- | -------------------- |
| 快速下降 | 1→11  | 13.29→2.84 | 正常                   |
| 震荡回升 | 11→21 | 2.84→4.60  | 验证 RMSE 反弹           |
| 再次下降 | 21→38 | 4.60→1.17  | 缓慢收敛                 |
| 停滞发散 | 38→63 | 1.17→3.65  | best 后 25 epoch 持续恶化 |

**典型问题**：

- epoch 38 之后 val_RMSE 从 1.17 反弹到 3.65，说明学习率在收敛区间仍为 0.001 没有衰减，导致在最优解附近振荡甚至发散
- d2l 11.11 指出：「持续高学习率会在最小值附近震荡；理想的衰减应比 O(t^{-1/2}) 更慢」
- epoch 60-63 出现 val_loss=18.78 的极端值，说明存在梯度不稳定（d2l 11.1 中的"梯度消失/爆炸"端）

### 2.2 与已有基线对比

| 模型                        | macro_RMSE | epochs | 注意点          |
| ------------------------- | ---------- | ------ | ------------ |
| MultimodalFusionV3 (纯多模态) | 0.640      | 109    | 专用波形模型，端到端设计 |
| MultimodalFusionV3+AMP    | 0.670      | 145    | AMP 模式       |
| TCN (纯慢变量)                | 0.686      | 96     | 纯 slow 8 维输入 |
| LSTM (纯慢变量)               | 0.743      | 103    | 纯 slow 8 维输入 |
| **CNN1D_multimodal**      | **1.163**  | **63** | 收敛不充分        |

CNN1D_multimodal 的 RMSE=1.163 明显高于同数据集上其他模型（0.64-0.74），而且仅跑了 63 epoch 就 early stop 了。可能的改善空间很大。

## 3. d2l 第十一章的优化建议与分析

### 3.1 学习率调度器（d2l 11.11）—— 最高优先级

**核心问题**：当前学习率在整个训练过程中恒为 0.001，这是导致训练震荡和 early stopping 过早触发的主因。

d2l 11.11 明确指出：

- 「在训练期间逐步降低学习率可以提高准确性，并且减少模型的过拟合」
- 「进展趋于稳定时降低学习率很有效，可确保收敛并减小参数固有方差」
- 余弦调度器「在计算机视觉的背景下可能产生改进的结果」

**推荐方案**：MultiStepLR（分段常数衰减）

理由：

1. 多因子调度器是「训练深度网络的常见策略之一」
2. 实现最简，与现有 early stopping 机制兼容
3. 可控性强：在已知收敛放缓的时间点降学习率
4. 不需要修改训练循环主体结构

具体参数：

```python
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 140], gamma=0.1)
# epoch 80: lr 0.001 → 0.0001
# epoch 140: lr 0.0001 → 0.00001
```

### 3.2 动量法改进（d2l 11.6）—— 中优先级

当前 Adam 已内置动量（β₁=0.9），等效于「过去约 10 个梯度的加权平均方向」。d2l 11.6 的分析说明：

- β₁=0.9 使有效样本数 = 1/(1-0.9) = 10
- 动量法防止「条件不佳」（病态曲率）问题，即一个方向梯度远大于另一个方向

当前 Adam 的默认 β₁=0.9 是合理的，不需要调整。但值得注意 CNN1D_multimodal 训练曲线的震荡可能部分源于条件不佳问题——卷积层的梯度在不同通道间差异大。

### 3.3 Adam 的 weight_decay（d2l 11.10 + 11.7）—— 中优先级

当前配置中：

- 纯慢变量模型和 multimodal 模型使用 weight_decay=0.0001
- MultimodalFusionV3+AMP 使用 weight_decay=0.0

d2l 11.7 的 AdaGrad 分析指出：「稀疏特征和学习率」需要不同处理。在 Adam 中，weight_decay 与自适应学习率的交互值得注意：

- L2 正则化被 Adam 的自适应学习率缩放，导致正则化效果不均匀
- 解耦权重衰减（AdamW）将 weight_decay 与梯度更新分离，是更稳定的选择

**推荐方案**：将 `torch.optim.Adam` 替换为 `torch.optim.AdamW`，保持 weight_decay 参数不变。AdamW 是 PyTorch 标准实现，无需额外依赖。

### 3.4 梯度裁剪—— 低优先级但推荐

CNN1D_multimodal 训练中 epoch 62 出现 val_loss=18.78 的极端值（normal 水平约 2-5），这是典型梯度爆炸。

d2l 11.1 第 4 点指出「梯度消失可导致优化停滞」，而梯度爆炸同样危险。梯度裁剪（gradient clipping）是深度学习中的标准防御手段：

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 3.5 Warmup（d2l 11.11）—— 低优先级

d2l 指出：「使用预热期，在此期间学习率将增加至初始最大值」。对于从头训练的小模型（参数量 < 150K），随机初始化到稳定训练的过渡期很短（1-3 epoch），warmup 收益有限。如果引入余弦调度器，可考虑加 5 epoch warmup。

## 4. 改进方案优先级

| 优先级    | 改进项                    | d2l 对应     | 预期收益                       | 改动范围                                | 风险  |
| ------ | ---------------------- | ---------- | -------------------------- | ----------------------------------- | --- |
| **P0** | 引入学习率调度器 (MultiStepLR) | 11.11      | 训练稳定性大幅提升，RMSE 预计下降 20-40% | `train.py` 加 scheduler 逻辑 + 配置 YAML | 低   |
| **P1** | Adam → AdamW           | 11.7/11.10 | 正则化效果更均匀                   | `train.py` 改一行                      | 极低  |
| **P2** | 梯度裁剪 (clip_grad_norm)  | 11.1       | 防止梯度爆炸导致训练崩溃               | `train.py` 加一行                      | 极低  |
| **P3** | 余弦退火调度器 + Warmup       | 11.11      | 可能进一步收敛，但需要更多调参            | `train.py` + 新配置                    | 中   |

## 5. P0 实施细节：学习率调度器

### 5.1 配置设计

在 `training` 配置段增加 `lr_scheduler` 子段：

```yaml
training:
  epochs: 200
  batch_size: 8
  learning_rate: 0.001
  weight_decay: 0.0001
  lr_scheduler:
    type: multistep          # multistep | cosine | plateau
    milestones: [80, 140]    # multistep 专用
    gamma: 0.1               # 衰减系数
  early_stopping_patience: 25
```

### 5.2 代码改动

`train.py` 中在 optimizer 创建后增加：

```python
# 学习率调度器
scheduler_cfg = config["training"].get("lr_scheduler", {})
if scheduler_cfg:
    scheduler_type = scheduler_cfg.get("type", "multistep")
    if scheduler_type == "multistep":
        milestones = scheduler_cfg.get("milestones", [80, 140])
        gamma = float(scheduler_cfg.get("gamma", 0.1))
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    elif scheduler_type == "cosine":
        T_max = total_epochs
        eta_min = float(scheduler_cfg.get("eta_min", 1e-6))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    elif scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=float(scheduler_cfg.get("gamma", 0.1)), patience=int(scheduler_cfg.get("patience", 10)))
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
else:
    scheduler = None

# 训练循环中每 epoch 结束后:
if scheduler is not None:
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(monitor_value)
    else:
        scheduler.step()
```

### 5.3 兼容性

- `lr_scheduler` 配置为可选，未配置时等价于恒定学习率（当前行为），完全向后兼容
- 6 个已跑的模型结果不受影响（不会重新训练）
- checkpoint 恢复时需要保存/恢复 scheduler 状态（需在 `_checkpoint_payload` 中增加 `scheduler_state_dict` 字段）

## 6. 预期效果

基于 d2l 第十一章的理论分析和 CNN1D_multimodal 的训练曲线：

| 指标         | 当前 (无 scheduler)        | 预期 (MultiStepLR) |
| ---------- | ----------------------- | ---------------- |
| macro_RMSE | 1.163                   | 0.7-0.9          |
| 有效训练 epoch | 38 (best) / 63 (stop)   | 100-150          |
| 训练稳定性      | 震荡严重，val_loss 极端值 18.78 | 平滑收敛             |
| 与纯慢变量基线差距  | 1.163 vs 0.69-0.74      | 缩小到 0.05 以内      |

核心论据（来自 d2l）：

1. 11.11：「在训练期间逐步降低学习率可以提高准确性，并且减少模型的过拟合」
2. 11.6 动量分析：Adam 的 β₁=0.9 已提供动量，但恒定学习率使后期在最优解附近大幅震荡
3. 11.1：「梯度消失可导致优化停滞」——后期 val_loss 极端值表明优化缺乏稳定性保障