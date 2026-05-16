# CNN1D_multimodal 优化器改进：Linear Warmup + Cosine Decay 详细实现文档

> 创建日期：2026-05-16
> 范围：仅修改 CNN1D_multimodal 配置和训练代码，验证后推广到其他 5 个模型

## 1. 变更总览

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| `src/dl/training/train.py` | 修改 | 增加 AdamW 支持、学习率调度器、梯度裁剪、checkpoint 恢复调度器 |
| `configs/deep/slow_only_cnn1d_multimodal_formal.yaml` | 修改 | optimizer=adamw, weight_decay=0.01, 新增 lr_scheduler 和 grad_clip_norm |
| `tests/test_deep_checkpoint_resume.py` | 修改 | 增加 scheduler 恢复的测试用例 |

## 2. 代码改动详述

### 2.1 优化器创建（train.py L611-614）

**位置**：`train_config()` 函数内，`model = build_model(...)` 之后。

**修改前**：
```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=float(config["training"].get("learning_rate", 1e-3)),
    weight_decay=float(config["training"].get("weight_decay", 0.0)),
)
```

**修改后**：
```python
lr = float(config["training"].get("learning_rate", 1e-3))
wd = float(config["training"].get("weight_decay", 0.01))
optimizer_name = config["training"].get("optimizer", "adam").lower()
if optimizer_name == "adamw":
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
```

**要点**：
- `optimizer` 配置键默认为 `"adam"`，向后兼容——所有现有配置没有此键时走 Adam 分支
- `weight_decay` 默认值从 `0.0` 改为 `0.01`——但这只影响新配置显式覆盖的场景，现有配置多数已写明 `weight_decay: 0.0001`

### 2.2 学习率调度器创建（train.py L616 后新增）

**位置**：`stopper = EarlyStopping(...)` 之后、`total_epochs = ...` 之后。

**新增代码**：
```python
total_epochs = int(config["training"].get("epochs", 200))
grad_clip_norm = float(config["training"].get("grad_clip_norm", 0.0))

scheduler_cfg = config["training"].get("lr_scheduler", None)
scheduler = None
if scheduler_cfg is not None:
    from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
    stype = scheduler_cfg.get("type", "cosine_warmup")
    if stype == "cosine_warmup":
        warmup_epochs = int(scheduler_cfg.get("warmup_epochs", 5))
        eta_min = float(scheduler_cfg.get("eta_min", lr * 0.1))
        warmup = LinearLR(
            optimizer,
            start_factor=1e-8,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=total_epochs - warmup_epochs,
            eta_min=eta_min,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )
    elif stype == "cosine":
        eta_min = float(scheduler_cfg.get("eta_min", lr * 0.1))
        scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=eta_min)
    elif stype == "multistep":
        milestones = [int(m) for m in scheduler_cfg.get("milestones", [80, 140])]
        gamma = float(scheduler_cfg.get("gamma", 0.1))
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    elif stype == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min",
            factor=float(scheduler_cfg.get("gamma", 0.1)),
            patience=int(scheduler_cfg.get("patience", 10)),
        )
    else:
        raise ValueError(f"Unknown lr_scheduler type: {stype}")
```

**要点**：
- `lr_scheduler` 配置键不存在时 `scheduler = None`，等价于恒定 LR——完全向后兼容
- 支持 4 种调度策略：`cosine_warmup`（新增推荐）、`cosine`、`multistep`、`plateau`
- `eta_min` 默认为 `lr * 0.1`（即 0.001 → 0.0001），与 LLaMA 实践一致

### 2.3 梯度裁剪（train.py `_train_one_epoch` 内）

**位置**：`_train_one_epoch()` 函数签名增加 `grad_clip_norm` 参数。

**修改前**：
```python
def _train_one_epoch(
    model, loader, loss_fn, optimizer, device, dataset,
    env_aug_sigma: float, amp_enabled: bool = False,
    scaler: torch.amp.GradScaler | None = None,
):
```

**修改后**：
```python
def _train_one_epoch(
    model, loader, loss_fn, optimizer, device, dataset,
    env_aug_sigma: float, amp_enabled: bool = False,
    scaler: torch.amp.GradScaler | None = None,
    grad_clip_norm: float = 0.0,
):
```

**位置**：`loss.backward()` 之后、`optimizer.step()` 之前插入：

**AMP 分支**（`if amp_enabled`）：
```python
if amp_enabled:
    scaler.scale(loss).backward()
    if grad_clip_norm > 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
    scaler.step(optimizer)
    scaler.update()
```

**非 AMP 分支**（`else`）：
```python
else:
    loss.backward()
    if grad_clip_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
    optimizer.step()
```

**调用处**（训练循环中）：
```python
train_loss = _train_one_epoch(
    model, loaders["train"], loss_fn, optimizer, device, datasets["train"],
    float(config["training"].get("environment_augmentation_sigma", 0.0)),
    amp_enabled=amp_enabled, scaler=scaler,
    grad_clip_norm=grad_clip_norm,
)
```

### 2.4 调度器步进（train.py 训练循环内）

**位置**：每个 epoch 结尾，`early_stopping.should_stop` 判断之后、`checkpoint_every` 保存之前。

在现有 `_save_checkpoint(last_ckpt_path, ...)` 行之后增加：

```python
if scheduler is not None:
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(monitor_value)
    else:
        scheduler.step()
```

**要点**：
- `ReduceLROnPlateau` 需要 `val_loss` 或 `macro_RMSE` 作为参数，其他调度器不需要
- scheduler.step() 必须在 optimizer.step() 之后调用（每个 epoch 一次）

### 2.5 Checkpoint 保存/恢复调度器状态

**`_checkpoint_payload` 增加 `scheduler_state_dict` 键**：

在现有 `"amp_scaler_state_dict": scaler.state_dict()` 之后增加：
```python
"scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
```

**`_load_checkpoint` 恢复调度器**：

在 `_restore_early_stopping` 调用之后增加：
```python
scheduler_state = ckpt.get("scheduler_state_dict")
if scheduler_state is not None and scheduler is not None:
    scheduler.load_state_dict(scheduler_state)
```

**`_validate_checkpoint_compat` 增加调度器兼容检查**：

在现有 `model_keys` 循环之后增加：
```python
# 调度器类型变更不阻塞恢复，但记录差异
ckpt_scheduler_type = ckpt_config.get("training", {}).get("lr_scheduler", {}).get("type")
cur_scheduler_type = config.get("training", {}).get("lr_scheduler", {}).get("type")
if ckpt_scheduler_type != cur_scheduler_type:
    mismatches.append(f"lr_scheduler.type: checkpoint={ckpt_scheduler_type}, config={cur_scheduler_type}")
```

### 2.6 `_train_one_epoch` 函数签名变更的影响

`_train_one_epoch` 在以下位置被调用：
1. `train_config()` 主训练循环 — **需更新调用处**

检查其他调用点：
- 不存在其他调用处（`evaluate_with_predictions` 和 `predict` 不调用 `_train_one_epoch`）

## 3. 配置变更

**`configs/deep/slow_only_cnn1d_multimodal_formal.yaml` 完整修改**：

```yaml
run:
  name: v3_cnn1d_multimodal_seed42
  seed: 42
  output_dir: outputs/exp02_deep_e2e/v3_cnn1d_multimodal_seed42

data:
  dataset_type: waveform_v3
  npz_path: ../../data/waveform_v3
  index_path: ../../data/waveform_v3/sequence_index.csv
  split_dir: ../../data/waveform_v3/splits
  scaler_path: ../../data/waveform_v3/scalers/scaler_slow_sequence.json
  split_strategy: existing_or_group_mixture
  time_window: all

model:
  name: cnn1d_multimodal
  slow_dim: 8
  hidden_channels: [32, 64, 64]
  kernel_size: 5
  dropout: 0.1
  out_dim: 4

training:
  epochs: 200
  batch_size: 8
  device: auto
  optimizer: adamw
  learning_rate: 0.001
  weight_decay: 0.01
  loss: mse
  early_stopping_patience: 25
  num_workers: 2
  eval_num_workers: 1
  grad_clip_norm: 1.0
  lr_scheduler:
    type: cosine_warmup
    warmup_epochs: 5
    eta_min: 0.0001
```

**变更对比**：

| 键 | 旧值 | 新值 | 理由 |
|----|------|------|------|
| `optimizer` | （无，默认 adam） | `adamw` | 解耦 weight decay |
| `weight_decay` | `0.0001` | `0.01` | AdamW 标准值（ICLR 2025 Wang et al.） |
| `grad_clip_norm` | （无，默认 0） | `1.0` | 防止 val_loss 极端值 18.78 |
| `lr_scheduler.type` | （无） | `cosine_warmup` | d2l 11.11 + LLaMA 实践 |
| `lr_scheduler.warmup_epochs` | （无） | `5` | 200 epoch × 2.5% = 5 epoch |
| `lr_scheduler.eta_min` | （无） | `0.0001` | η_max/10 = 0.001/10 |

## 4. 向后兼容性

| 场景 | 行为 |
|------|------|
| 旧配置（无 `optimizer`、`lr_scheduler`、`grad_clip_norm` 键） | `optimizer` 默认 `"adam"` → Adam；`scheduler = None` → 恒定 LR；`grad_clip_norm = 0.0` → 不裁剪。**行为与改动前完全一致** |
| 旧 checkpoint 恢复 | `scheduler_state_dict` 不存在时跳过恢复，不影响 Adam 恢复 |
| 旧测试 | `test_deep_checkpoint_resume.py` 中的恢复测试不需要修改（scheduler_state_dict 为 None 时不恢复） |

## 5. 测试计划

### 5.1 新增单元测试

**文件**：`tests/test_lr_scheduler.py`

```python
class LRSchedulerTests(unittest.TestCase):
    def test_cosine_warmup_lr_schedule_shape(self):
        """warmup 阶段 LR 线性增长，之后 cosine 递减到 eta_min"""
        # 创建简单模型和配置，验证 LR 曲线形状

    def test_cosine_warmup_no_scheduler_config_is_identity(self):
        """无 lr_scheduler 配置时 LR 恒定，与旧行为一致"""

    def test_adamw_default_weight_decay(self):
        """AdamW 默认 weight_decay=0.01"""

    def test_adam_default_weight_decay(self):
        """Adam 默认 weight_decay=0.0（向后兼容）"""

    def test_grad_clip_norm_zero_is_noop(self):
        """grad_clip_norm=0 时不裁剪梯度"""

    def test_grad_clip_norm_one_clips_gradient(self):
        """grad_clip_norm=1 时梯度范数被裁剪到 1.0"""

    def test_scheduler_checkpoint_round_trip(self):
        """scheduler state_dict 保存和恢复一致性"""

    def test_config_without_scheduler_backward_compatible(self):
        """旧配置（无 lr_scheduler 键）创建 scheduler=None"""
```

### 5.2 扩展现有测试

**`test_deep_checkpoint_resume.py`** 增加一个测试：

```python
def test_checkpoint_with_scheduler_can_resume(self):
    """带 scheduler 的 checkpoint 可以恢复并继续训练"""
```

### 5.3 回归测试

运行 `python -m pytest tests` 确保全部通过。

## 6. 验证计划

### 6.1 重跑 CNN1D_multimodal

```powershell
python src/pipeline/train_deep.py --config configs/deep/slow_only_cnn1d_multimodal_formal.yaml --no-ui
```

### 6.2 对比指标

| 指标 | 旧模型 (Adam, 恒定 lr) | 目标 (AdamW, warmup+cosine) |
|------|----------------------|--------------------------|
| macro_RMSE | 1.163 | ≤ 0.85 |
| 训练曲线形态 | 震荡、极端值 val_loss=18.78 | 平滑收敛 |
| 有效 epoch | 38 (best) / 63 (stop) | 80-150 |
| val_loss 极端值 | 18.78 | < 5.0 |

### 6.3 推广决策

若 RMSE 降到 0.70-0.85 范围且训练曲线平稳：
- 将 `optimizer: adamw`、`grad_clip_norm: 1.0`、`lr_scheduler` 配置推广到其余 5 个 multimodal 配置
- 保留各 backbone 的超参差异（hidden_size、kernel_size 等），只统一优化策略

若 RMSE 改善不明显（>0.90）：
- 检查数据加载是否成为瓶颈（P0 num_workers=2 是否生效）
- 尝试增大 warmup_epochs 到 10（5% 总 steps）
- 尝试 eta_min=1e-6（更激进的衰减）

## 7. LR 曲线可视化（预期）

```
LR
|  0.001 ─────╱‾‾‾‾‾‾╲
|           ╱          ╲
|          ╱            ╲
|         ╱              ╲
|        ╱                ╲
|       ╱                  ╲
|      ╱                    ╲
|     ╱                      ╲
|0.000╱────────────────────────╲─→ 0.0001
|     |5|                    200
|   warmup              cosine decay
```

- Epoch 1-5：LR 从 1e-11 线性增长到 1e-3（warmup）
- Epoch 5-200：LR 从 1e-3 余弦衰减到 1e-4（cosine decay）
- Early stopping patience=25 仍在最佳 epoch 后 25 epoch 停止

## 8. 依赖变更

无新增依赖。所有使用的类（`AdamW`、`SequentialLR`、`LinearLR`、`CosineAnnealingLR`、`clip_grad_norm_`）均为 PyTorch 标准库，当前项目 PyTorch 版本即可支持。