# 多模态 formal 一键训练脚本

本文档说明新增的 3 个多模态 formal 单模型训练入口。三份脚本都在项目根目录，默认读取对应的 `configs/deep/*_multimodal_formal.yaml`，并优先请求显示 CLI UI；若宿主终端不兼容，则自动退回普通日志。

## 脚本列表

- `run_gru_multimodal_formal.ps1`
- `run_lstm_multimodal_formal.ps1`
- `run_tcn_multimodal_formal.ps1`

## 默认对应关系

| 脚本                               | 配置文件                                                 | 默认 run.name                 |
| -------------------------------- | ---------------------------------------------------- | --------------------------- |
| `run_gru_multimodal_formal.ps1`  | `configs/deep/slow_only_gru_multimodal_formal.yaml`  | `v3_gru_multimodal_seed42`  |
| `run_lstm_multimodal_formal.ps1` | `configs/deep/slow_only_lstm_multimodal_formal.yaml` | `v3_lstm_multimodal_seed42` |
| `run_tcn_multimodal_formal.ps1`  | `configs/deep/slow_only_tcn_multimodal_formal.yaml`  | `v3_tcn_multimodal_seed42`  |

## 基本用法

在项目根目录 PowerShell 中执行：

```powershell
.\run_gru_multimodal_formal.ps1
.\run_lstm_multimodal_formal.ps1
.\run_tcn_multimodal_formal.ps1
```

如需显式关闭 UI：

```powershell
.\run_gru_multimodal_formal.ps1 --no-ui
.\run_lstm_multimodal_formal.ps1 --no-ui
.\run_tcn_multimodal_formal.ps1 --no-ui
```

## 透传参数

脚本会把额外参数原样透传给 `python src/pipeline/train_deep.py`，因此可以直接补充恢复训练、修改 epoch 数、覆盖输出根目录等参数。

示例：

```powershell
.\run_gru_multimodal_formal.ps1 --epochs 20
.\run_lstm_multimodal_formal.ps1 --output-root outputs/exp02_single_runs
.\run_tcn_multimodal_formal.ps1 --resume outputs/exp02_deep_e2e/v3_tcn_multimodal_seed42/last_checkpoint.pt
```

## 当前训练口径

这 3 份 formal 配置已经与 `slow_only_cnn1d_multimodal_formal.yaml` 对齐到同一训练流程，仅保留 backbone 参数差异。当前共同训练参数为：

- `epochs: 120`
- `batch_size: 32`
- `amp: false`
- `optimizer: adamw`
- `learning_rate: 0.0003`
- `weight_decay: 0.01`
- `label_balanced_loss: true`
- `sum_constraint.weight: 0.1`
- `early_stopping_patience: 25`
- `grad_clip_norm: 1.0`
- `lr_scheduler.type: cosine_warmup`
- `lr_scheduler.warmup_epochs: 10`
- `lr_scheduler.eta_min: 0.00003`

## 备注

- 三个脚本都会切回项目根目录后再启动训练，避免从其他目录运行时找不到相对路径。
- 不同 backbone 使用独立 `run.name` 和 `output_dir`，不会与 `cnn1d_multimodal` 训练结果混写。
- 当使用“右键使用 PowerShell 运行”等不兼容 ANSI/VT 的宿主时，脚本会自动退回普通日志，不需要手动改回 `--no-ui`。

## 训练曲线导出

训练完成后，可直接用 `src/pipeline/plot_deep_training_curves.py` 从对应 run 目录导出训练曲线。对单个多模态 run，推荐把该 run 目录直接作为 `--root`，这样只会处理该模型自身的 `train_log.csv`。

示例：

```powershell
python src/pipeline/plot_deep_training_curves.py --root outputs/exp02_deep_e2e/v3_tcn_multimodal_seed42 --output-dir outputs/deep_training_curves/v3_tcn_multimodal_seed42
```

默认会输出：

- `v3_tcn_multimodal_seed42_training_curves.png`
- `v3_tcn_multimodal_seed42_training_curves.svg`
- `all_runs_val_macro_RMSE.png`
- `all_runs_val_macro_RMSE.svg`
