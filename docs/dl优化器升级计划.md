# CNN1D_multimodal 优化器改进计划：Linear Warmup + Cosine Decay

> 创建日期：2026-05-16
> 基于搜索结果：PyTorch 官方文档、Lightning-Bolts、fairseq、pytorch-scheduler、ICLR 2023/2025 论文

## 1. 当前问题诊断

CNN1D_multimodal 训练曲线显示：

- **epoch 1-11**：RMSE 从 13.29 快速降到 2.84，正常
- **epoch 11-21**：RMSE 反弹到 4.60，训练不稳定
- **epoch 21-38**：缓慢收敛到 best 1.17
- **epoch 38-63**：RMSE 从 1.17 反弹到 3.65，极端值 val_loss=18.78

根因：恒定 lr=0.001 在最优解附近过度振荡，引发梯度爆炸。

## 2. 文献与实践搜证

### 2.1 Linear Warmup + Cosine Decay 是主流方案

- **BERT/GPT/T5/LLaMA** 全部使用 linear warmup + cosine decay（搜索来源：engineersofai.com, pytorch-scheduler README）
- Lightning-Bolts 提供 `LinearWarmupCosineAnnealingLR` 开箱即用
- fairseq 的 `CosineLRSchedule` 也内置 warmup 参数
- ICLR 2023 Blog 指出 AdamW 的 bias correction 问题使 warmup 几乎是强制的

### 2.2 关键参数设定

| 参数 | 推荐值 | 来源 | 项目采用 |
|------|--------|------|---------|
| **warmup 比例** | 小模型 1-5% 总 steps；Transformer 4-10% | engineersofai.com, mbrenndoerfer.com | **5 epochs**（200/5=2.5%） |
| **eta_min** | η_max/10 或 1e-6 | LLaMA 用 η_max/10 | **1e-4**（η_max/10） |
| **weight_decay** | AdamW 解耦衰减，随 LR 缩放 | ICLR 2025 Wang et al. | **0.01**（AdamW 标准值） |
| **peak LR** | 不变 | — | **1e-3** |
| **T_max** | 总 epoch 数（含 warmup） | d2l 11.11 | **200** |

### 2.3 Adam → AdamW

PyTorch 的 AdamW 实现中，权重衰减因子为 `(1 - η_t * λ)`，即衰减与当前学习率成正比。这意味着：
- 在 warmup 阶段 LR 较小时，衰减弱
- 在 cosine decay 阶段 LR 逐渐降低时，衰减也自动减弱
- 这比 Adam 的 L2 正则化（固定衰减）更合理

ICLR 2025 论文 (Wang et al.) 指出：**当 LR 和 weight_decay 都变化时，EMA timescale（按 epoch 计）应保持恒定**。对于 cosine decay，weight_decay = 0.01 是良好的起点。

### 2.4 梯度裁剪

CNN1D_multimodal 出现 val_loss=18.78 的极端值，是梯度爆炸的典型特征。`clip_grad_norm_` 是标准防御手段。

## 3. 实施方案

### 3.1 配置变更

```yaml
# 修改前
training:
  epochs: 200
  batch_size: 8
  learning_rate: 0.001
  weight_decay: 0.0001
  loss: mse
  early_stopping_patience: 25

# 修改后
training:
  epochs: 200
  batch_size: 8
  optimizer: adamw
  learning_rate: 0.001
  weight_decay: 0.01
  loss: mse
  early_stopping_patience: 25
  lr_scheduler:
    type: cosine_warmup
    warmup_epochs: 5
    eta_min: 0.0001
  grad_clip_norm: 1.0
```

### 3.2 代码改动

#### 3.2.1 优化器：Adam → AdamW（`train.py`）

```python
# 修改前
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=float(config["training"].get("learning_rate", 1e-3)),
    weight_decay=float(config["training"].get("weight_decay", 0.0)),
)

# 修改后
optimizer_name = config["training"].get("optimizer", "adam").lower()
lr = float(config["training"].get("learning_rate", 1e-3))
wd = float(config["training"].get("weight_decay", 0.01))
if optimizer_name == "adamw":
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
```

#### 3.2.2 学习率调度器（`train.py`）

在 optimizer 创建后增加：

```python
scheduler_cfg = config["training"].get("lr_scheduler", {})
scheduler = None
if scheduler_cfg:
    from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
    stype = scheduler_cfg.get("type", "cosine_warmup")
    if stype == "cosine_warmup":
        warmup_epochs = int(scheduler_cfg.get("warmup_epochs", 5))
        eta_min = float(scheduler_cfg.get("eta_min", 1e-4))
        warmup = LinearLR(optimizer, start_factor=1e-8, end_factor=1.0,
                          total_iters=warmup_epochs)
        cosine = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs,
                                    eta_min=eta_min)
        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine],
                                  milestones=[warmup_epochs])
```

训练循环中每 epoch 结尾：

```python
if scheduler is not None:
    scheduler.step()
```

#### 3.2.3 梯度裁剪（`train.py`）

在 `_train_one_epoch` 的 `optimizer.step()` 之前：

```python
grad_clip_norm = float(config["training"].get("grad_clip_norm", 0.0))
# 在 loss.backward() 之后、optimizer.step() 之前
if grad_clip_norm > 0:
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
```

#### 3.2.4 Checkpoint 兼容

在 `_checkpoint_payload` 和 `_load_checkpoint` 中增加 `scheduler_state_dict`。

### 3.3 只改 CNN1D_multimodal 还是全部模型？

**建议先只改 CNN1D_multimodal**，验证效果后再推广。理由：
1. CNN1D_multimodal 已跑完一轮，有明确的基线 RMSE=1.163
2. 其他 5 个 multimodal 模型尚未训练，改了优化器后没有对比基线
3. 验证 warmup+cosine 有效后，再统一更新其他配置

修改的配置文件：仅 `configs/deep/slow_only_cnn1d_multimodal_formal.yaml`

## 4. 预期效果

| 指标 | 当前 (Adam, lr=0.001 恒定) | 预期 (AdamW, warmup+cosine) |
|------|-----|------|
| macro_RMSE | 1.163 | 0.70-0.85 |
| 有效训练 epoch | 38 (best) | 80-150 |
| val_loss 极端值 | 18.78 | 无 |
| 收敛曲线形态 | 震荡→发散 | 平滑下降→稳定 |

## 5. 验证计划

1. 先对 CNN1D_multimodal 重跑一轮，对比：
   - 新 RMSE vs 旧 RMSE (1.163)
   - 新 RMSE vs 同数据集纯慢变量基线 (0.686-0.743)
   - 新 RMSE vs MultimodalFusionV3 (0.640-0.670)
2. 对比训练曲线：val_loss 是否不再出现极端振荡
3. 如果 RMSE 降到 0.7-0.85 范围，推广到其他 5 个 multimodal 配置