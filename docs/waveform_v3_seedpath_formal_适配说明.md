# `waveform_v3_seedpath_formal` 训练入口适配说明

> 更新日期：2026-05-19
> 范围：DL 端 `cnn1d_tcn_fusion`、ML 端全套传统模型
> 数据源：`data/waveform_v3_seedpath_formal/`
> 配套报告：[`waveform_v3_seedpath_formal_数据分析报告.md`](./waveform_v3_seedpath_formal_数据分析报告.md)

## 1. 背景

`waveform_v3_seedpath_formal` 是在 V3.1 30000 序列数据包基础上，重新生成 split 与 scaler 的正式版本：

- split 策略：`stratified_group_by_mixture_id_with_extrapolation_holdout`
- 四路 split：`train` 17400 / `val` 4050 / `test` 4050 / `extrapolation` 4500
- scaler 仅在新 train split 上重新拟合（z-score）
- 旧数据集 `data/waveform_v3` 仅有 10000 序列，特征产物 `outputs/exp01_traditional/` 也只有 40000 样本，保留作历史对照，不删除

本文档记录把现有 ML / DL 训练入口切到新数据集时已经动过的内容。

## 2. DL 端：`cnn1d_tcn_fusion` 三套配置

改动文件：

| 配置文件 | 改动 |
|---|---|
| `configs/deep/slow_only_cnn1d_tcn_fusion_multimodal_formal.yaml` | data 段 4 个路径切到新数据集 |
| `configs/deep/slow_only_cnn1d_tcn_fusion_multimodal_formal_run_b.yaml` | 同上 |
| `configs/deep/slow_only_cnn1d_tcn_fusion_multimodal_formal_run_c.yaml` | 同上 |

具体改动（main 配置示例）：

```yaml
data:
  dataset_type: waveform_v3
  npz_path: ../../data/waveform_v3_seedpath_formal
  index_path: ../../data/waveform_v3_seedpath_formal/sequence_index.csv
  split_dir: ../../data/waveform_v3_seedpath_formal/splits
  scaler_path: ../../data/waveform_v3_seedpath_formal/scalers/scaler_slow_sequence.json
  split_strategy: existing_or_group_mixture
  time_window: all
```

`split_strategy: existing_or_group_mixture` 字段保持不变——`load_existing_splits` 已经能自动加载新数据集 `splits/` 下的 4 路 split 文件，包括 `extrapolation_sequence_ids.csv`。

### 2.1 代码无需改动

新数据集与旧数据集在以下层面完全一致：

- 目录结构（`sequences/`、`labels/`、`metadata/`、`splits/`、`scalers/`、`quality/`）
- 文件命名（`ultrasonic_int16.npy` / `fiber_mic_int16.npy` / `slow.npy` 等）
- `metadata/waveform_v3_spec.json` 字段
- `splits/*.csv` 列（`sequence_id`, `mixture_id`）
- `condition_grid_sequence.csv` 列

所以 `src/dl/data/dataset_waveform.py`、`split_utils.py`、`data_setup.py` 不动。`_build_waveform_datasets` 会把 4 路 split 全部构造成 dataset 实例。

### 2.2 已知未做的事

- `orchestrator._prepare_loader_resources` 目前只为 `train/val/test` 创建 loader，`extrapolation` dataset 会被构造出来但不会评估。如果要在 summary.json 里单独报告 extrapolation 指标，需要单独改 `orchestrator.py`。
- 三个 `output_dir`（`outputs/exp02_deep_e2e_tuned/run_{b,c,d}/...`）未改名，重新训练会覆盖旧的 `best_model.pt` / `train_log.csv` / `summary.json`。需要保留旧结果时请训练前手动备份。

### 2.3 验证结果

直接调用 `build_datasets` dry-run：

```
train:         dataset_len=17400, split_csv_rows=17400
val:           dataset_len=4050,  split_csv_rows=4050
test:          dataset_len=4050,  split_csv_rows=4050
extrapolation: dataset_len=4500,  split_csv_rows=4500
sample shapes: ultrasonic=(120,1000), fiber_mic=(120,2000), slow=(120,8), target=(4,)
sample_id:     Q000073（与 train_sequence_ids.csv 第二行 sequence_id 一致）
```

## 3. ML 端：传统模型流水线

### 3.1 特征表重新生成

特征提取脚本 `src/sim/scripts/extract_dual_waveform_features.py` 是自包含的，只通过 `--source-dir` 取数据，按 `--output-dir` 落盘。新数据集的特征产物已落到独立目录：

```
outputs/exp01_traditional_seedpath/
├── condition_grid_v1.csv            23M (120000 行)
├── labels/labels.csv                9.4M
├── features/feature_table*.csv      各 90M
└── training/train_{acoustic,optical,thermal}{,_env}.csv
```

样本数从旧的 40000（10000 sequence × 4 timestep）变为 120000（30000 sequence × 4 timestep）。

重产命令：

```powershell
python src/pipeline/feature_extraction.py `
  --source-dir data/waveform_v3_seedpath_formal `
  --output-dir outputs/exp01_traditional_seedpath
```

机器实测耗时约 4 分钟。

### 3.2 让 ML 复用 DL 的 split

三个脚本新增 `--split-dir` / `--split-include-val-in-train` 参数：

| 文件 | 新增内容 |
|---|---|
| `src/ml/scripts/train_patent_model.py` | 加 CLI；新增 `_load_split_mixture_ids` 和 `_split_dataset_by_mixture`；`prepare_training_data` 命中 `--split-dir` 时改用按 mixture_id 切分；`summary.json` 记录 `split_dir` / `split_include_val_in_train` 字段 |
| `src/ml/scripts/run_four_component_model_grid.py` | 加同名 CLI，透传到 `train_args` |
| `src/pipeline/train_traditional.py` | 加同名 CLI，透传到下游 |

切分语义：

- `train` = DL `train` mixtures + DL `val` mixtures（默认；ML pipeline 没有独立 val 槽位）
- `test` = DL `test` mixtures（与 DL 评估对象完全相同）
- `extrapolation` mixtures 既不进 train 也不进 test（与 DL 行为一致）

想保留 val 作为 holdout 不参与训练：加 `--no-split-include-val-in-train`。

向后兼容：不传 `--split-dir` 时仍走原有 `grouped_train_test_split`，旧脚本和旧 CI 不受影响。

### 3.3 验证结果

- dry-run（无 filter）：`train=85800 / test=16200`，train mixtures=4627 / test mixtures=563，与 DL `4050+577=4627` / `563` 完全对齐
- 端到端 smoke（svr_ridge，train_limit=4000，test_limit=800）：8 秒跑通，`fused_macro_RMSE_pp=6.79`，summary.json 中 `split_dir` 字段写入正确

### 3.4 推荐运行命令

新数据集全套传统 ML 重跑：

```powershell
python src/pipeline/train_traditional.py `
  --data-dir outputs/exp01_traditional_seedpath `
  --output-root outputs/exp01_traditional_seedpath `
  --tag formal_seedpath `
  --split-dir data/waveform_v3_seedpath_formal/splits `
  --profiles v3_raw_no_env v3_raw_tph `
  --max-workers 4
```

### 3.5 已知未做的事

- ML pipeline 当前只评估 test split。如要在 ML 端单独报告 extrapolation 指标，需要再扩展 `_split_dataset_by_mixture` 返回三路集合并在 `train_patent_model.run_training` 中加一段对 extrapolation 的评估。
- 环境补偿系列脚本 `run_environment_compensation_*.py` 同样调用 `prepare_training_data`，但 `--split-dir` 没透传到这些入口；如需对齐再补。

## 4. mixture_id 一致性核验

ML 的 `sample_id`（`W00001` 风格）与 DL 的 `sequence_id`（`Q000001` 风格）不在同一命名空间，但 `mixture_id`（`MD00001` 风格）是共享的。已核验：

- ML 特征表 `condition_grid_v1.csv` 中的 mixture_id 集合大小 = 5427
- DL 四路 split 的 mixture_id 并集 = 5427
- 双向差集为 0

所以 ML 按 mixture_id 复用 DL split 时不会有样本溢出或缺失。

## 5. 没有改动的事项

- DL 其他模型（`cnn1d_multimodal`、`gru_multimodal`、`lstm_multimodal`、`tcn_multimodal`、`early_fusion_film` 等）的配置仍指向旧 `data/waveform_v3`。后续如需切换，沿用本文 §2 的相同 4 字段改动。
- 旧 `outputs/exp01_traditional/` 与旧 cnn1d_tcn_fusion 训练产物保留，不覆盖、不删除。
- DL 训练流程未接入 extrapolation 评估（见 §2.2）；ML pipeline 同样未接入 extrapolation 评估（见 §3.5）。

## 6. 参考链接

- 数据分析报告：[`waveform_v3_seedpath_formal_数据分析报告.md`](./waveform_v3_seedpath_formal_数据分析报告.md)
- 数据集生成设计：[`数据集生成设计.md`](./数据集生成设计.md)
- 数据集改造 PLAN：[`数据集改造PLAN.md`](./数据集改造PLAN.md)
- 数据卡片：[`data_card.md`](./data_card.md)
