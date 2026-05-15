# 代码审查报告

**项目**: 多模态掺氢天然气浓度检测实验系统  
**审查日期**: 2026-05-15  
**审查范围**: `src/dl/` `src/ml/` `src/sim/` `src/pipeline/` `experiments/` `configs/` `tests/`

---

## 目录

- [验证修订说明](#验证修订说明)
- [P0 严重问题（2个，已修复）](#p0-严重问题)
- [P1 一般问题（21个）](#p1-一般问题)
- [P2 改进建议（18个）](#p2-改进建议)
- [架构层面共性问题](#架构层面共性问题)
- [优点总结](#优点总结)

---

## 验证修订说明

本报告已按 2026-05-15 的代码核查结果修订：

- 原 `P0-4`、`P0-6` 保留为 P0：前者会造成 `PatentDataset.provenance/filter_report` 隐性丢失，后者在组数不足时会用 in-sample 预测训练元学习器。
- 原 `P0-1`、`P0-2`、`P0-3`、`P0-5` 降级为 P1：问题真实存在或值得改进，但当前证据不足以判定为立即阻断级缺陷。
- 原 `P0-7` 降级为 P2：属于明显重复和维护成本问题，不是当前运行正确性的严重缺陷。
- 原 `P1-12` 删除：当前代码使用 `inclusive="both"`，这是 pandas 现行合法写法，不应改为 `closed="both"`。
- 原 `P1-15` 改写并降级为 P2：实验 profile 名称通过 `environment_compensation_common.py` 显式映射，不是运行错误；剩余问题是标签/展示命名体系统一性。
- 原 `P2-16` 改写：PowerShell 脚本已有 `$ErrorActionPreference = "Stop"`，问题应表述为缺少更完整的日志与收尾处理。
- 第一轮文档验证：`python -m pytest tests` 通过，结果为 4 passed。
- 2026-05-15 已按本报告优先路线修复核心 bug 项，并新增 `tests/test_code_review_fixes.py` 回归测试。
- 修复后验证：`python -m pytest tests` 通过，结果为 12 passed。

已修复条目：

- `P0-1`: `PatentDataset` 重构后保留 `provenance` 和 `filter_report`。
- `P0-2`: OOF 退化路径输出 `WARNING`，避免用户无感知使用退化结果。
- `P1-D1`: TCN 输入格式改为模型通过 `input_format = "NCT"` 声明，训练逻辑不再依赖类名字符串。
- `P1-D2`: `V2SequenceDataset` / `WaveformSequenceDataset` 支持 `preloaded_data`，训练构建阶段不再重复加载同一数据包。
- `P1-D3`: 缺省 `scaler_path` 时自动写入训练输出目录，避免只存在内存 scaler。
- `P1-D4`: 四组分模式显式校验 `condition_grid_v1.csv` 中的 `x_N2`。
- `P1-5`: `set_seed()` 补齐 cuDNN deterministic / benchmark 设置。
- `P1-13`: `train_patent_model.py` 复用 `scripts._cli_utils` 中的 CLI 工具函数。

---

## P0 严重问题

> **建议优先修复。** 这些问题会导致静默数据错误、推理不可用或模块间不同步。

### P0-1: PatentDataset 重构时丢失字段

**位置**: `src/ml/patent_model/fault_labels.py:175-188`
**状态**: 已修复。`inject_faults()`、`add_environment_noise()`、`add_profile_environment_noise()`、`select_pressure_slice()` 构造新 `PatentDataset` 时会复制 `provenance` 和 `filter_report`。

```python
return PatentDataset(
    sample_ids=dataset.sample_ids.copy(),
    ...
    # provenance= 缺失
    # filter_report= 缺失
)
```

`PatentDataset` 定义为 `@dataclass(frozen=True)`，`inject_faults()` 通过构造新实例返回结果，但遗漏了 `provenance` 和 `filter_report` 字段。

同样的问题出现在：

- `src/ml/patent_model/robustness.py:105-118`（`add_environment_noise`）
- `src/ml/patent_model/robustness.py:231-244`（`select_pressure_slice`）

**影响**: 下游分析如果依赖 `provenance` 或 `filter_report`，会拿到空 dict 且不报错，属于隐性数据丢失。

**建议**: 所有 `PatentDataset(...)` 构造处补上遗漏字段，或改用 `dataclasses.replace()` 统一处理。

---

### P0-2: OOF 退化时产生信息泄漏且无日志标识

**位置**: `src/ml/patent_model/modeling.py:388-395`
**状态**: 已部分修复。退化路径现在会输出 `WARNING` 标识 in-sample fallback；仍保留可执行 fallback，未改为强制失败。

当 `n_splits < 2` 时退化为全量拟合，OOF 预测变为 in-sample 预测，动态权重基于 in-sample 预测计算，元学习器输入存在严重信息泄漏。虽然注释提到了"退化"但无日志标记，用户无法感知模型质量已退化。

**建议**: 退化时至少打印 WARNING 日志，考虑在 `ModelConfig` 中增加 `min_groups` 参数。

---

## P1 一般问题

> 不会直接导致错误，但影响代码可维护性、可复现性或运行时行为。

### P1-D1: 类名字符串做模型调度，脆弱且易出错

**位置**: `src/dl/training/train.py:108`
**状态**: 已修复。`TCNRegressor` 声明 `input_format = "NCT"`，训练前向逻辑读取该属性决定是否转置。

```python
if model.__class__.__name__ == "TCNRegressor":
    slow_input = slow_input.transpose(1, 2)
```

用硬编码类名字符串决定输入格式转换。如果 `TCNRegressor` 被重命名、子类化，或新增需要 NCT 格式的模型，会导致输入格式转换与模型声明脱节。当前问题真实存在，但只影响模型扩展和重构路径，降级为 P1。

**建议**: 在模型基类或协议上添加 `input_format` 属性，让模型自己声明期望格式。

---

### P1-D2: 数据被加载两次，内存和加载时间增加

**位置**: `src/dl/training/train.py` + `src/dl/data/dataset_v2.py` + `src/dl/data/dataset_waveform.py`
**状态**: 已修复。两个 Dataset 均支持 `preloaded_data`，`_build_v2_datasets()` / `_build_waveform_datasets()` 会把已加载数据传入 Dataset。

`_build_v2_datasets` 先调用 `load_v2_npz` 获取完整数据做 split 和 scaler 计算，然后 `V2SequenceDataset._ensure_loaded()` 再调用 `load_v2_npz`。波形数据同理。该问题主要影响内存和加载时间，不是直接的数据正确性缺陷。

**建议**: 让 Dataset 接受预加载的数据，或让 `_build_*_datasets` 返回已加载数据供 Dataset 复用。

---

### P1-D3: scaler_path 缺省时 scaler 不会持久化

**位置**: `src/dl/training/train.py`
**状态**: 已修复。训练入口在 `data.scaler_path` 缺省时自动设置为 `output_dir/scaler_sequence.json` 或 `output_dir/scaler_slow_sequence.json`。

`_build_waveform_datasets` 中 `load_or_fit_scaler` 的 `scaler_path` 如果为 None，scaler 只存在于内存中。当前正式配置已提供 `scaler_path`，因此不是已验证的 P0；但入口层仍应避免静默使用不可复用的内存 scaler。

**建议**: 在配置校验中要求正式训练必须提供 `scaler_path`，或将缺省 scaler 保存到 `output_dir` 并写入 `config.json`。

---

### P1-D4: 四组分 target 来源不够直观

**位置**: `src/ml/patent_model/data_loader.py:415-475`
**状态**: 已修复。四组分 profile 会在读取阶段显式要求 `condition_grid_v1.csv` 包含 `x_N2`，缺列时抛出明确 `ValueError`。

原结论“`labels.csv` 缺少 `x_N2` 会导致四组分模式列校验不完整”不成立。当前四组分 `x_N2` 是从 `condition_grid_v1.csv` 合入 metadata 后补到 `targets_frame`，不是从 `labels.csv` 读取。

真实问题是：`labels.csv` 只校验三组分列，而四组分 target 的第 4 列来自另一张表，数据来源不够显式，后续维护者容易误判。建议在四组分分支增加注释或显式校验 `condition_grid_v1.csv` 中 `x_N2` 的存在与非空。

---

### P1-1: LSTM 和 GRU 模型近乎完全相同

**位置**: `src/dl/models/lstm.py` + `src/dl/models/gru.py`

两个文件仅 3 处差异：`nn.LSTM` vs `nn.GRU`、`self.lstm` vs `self.gru`、隐藏状态解包方式。`forward` 完全相同。

**建议**: 提取基类 `RNNRegressorBase`，子类只需指定 RNN 类型和隐藏状态解包方式。

---

### P1-2: TIME_WINDOWS 有重复键

**位置**: `src/dl/data/channel_groups.py:32-33`

```python
"baseline_exposure": list(range(0, 70)),
"baseline+exposure": list(range(0, 70)),
```

两个不同键映射到相同值。如果是有意支持不同命名风格应加注释，否则是冗余。

---

### P1-3: CHANNEL_GROUPS 重复定义

**位置**: `src/dl/models/branch_fusion.py:11-16`

`DEFAULT_CHANNEL_GROUPS` 与 `channel_groups.py` 中的 `CHANNEL_GROUPS` 完全相同。如某处被修改，另一处不会同步。

**建议**: 直接从 `channel_groups.py` 导入。

---

### P1-4: early_stopping 缺少 min_delta

**位置**: `src/dl/training/early_stopping.py`

任何微小改善（如 1e-10）都被认为是有效改善。浮点噪声可能被误判为改善，导致 early stopping 无法正确触发。

**建议**: 添加 `min_delta` 参数，仅当改善量超过阈值才算有效改善。

---

### P1-5: seed 设置不完整，GPU 上不可完全复现

**位置**: `src/dl/training/seed.py`
**状态**: 已修复。`set_seed()` 已设置 `torch.backends.cudnn.deterministic = True` 和 `torch.backends.cudnn.benchmark = False`。

缺少以下设置：

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

GPU 训练时不同运行间结果可能不一致。

---

### P1-6: train_config 函数过长（140行）

**位置**: `src/dl/training/train.py:357-493`

该函数承担了数据加载、模型创建、优化器配置、训练循环、评估、结果保存——违反单一职责原则，难以测试和复用。

**建议**: 至少拆分为 `create_trainer_components()`、`run_training_loop()`、`save_results()`。

---

### P1-7: persistent_workers 在 Windows 上可能有问题

**位置**: `src/dl/training/train.py:67`

```python
persistent_workers=num_workers > 0,
```

Windows 上 `persistent_workers=True` 可能导致进程无法正常退出。

**建议**: 仅在 Linux 上启用，或添加平台判断。

---

### P1-8: 波形数据回退时加载整个数据集

**位置**: `src/dl/data/dataset_waveform.py:42`

```python
data = load_waveform_package(path)
return _to_str_list(data.get("sequence_ids", np.arange(len(data["ultrasonic"]))))
```

当 `sequence_ids.npy` 不存在时，会加载整个波形数据包（可能数 GB）仅为了获取 sequence_ids。

**建议**: 在数据预处理阶段确保 `sequence_ids.npy` 存在，或在此处添加 warning。

---

### P1-9: R² 计算阈值可能过严

**位置**: `src/dl/training/metrics.py:31`

```python
denom = float(np.sum((y_true[:, i] - np.mean(y_true[:, i])) ** 2))
r2 = float(1.0 - np.sum(err**2) / denom) if denom > 1e-12 else float("nan")
```

对极小目标值（如 x_H2 在 0-5% 范围），方差可能接近 1e-12。

**建议**: 改为 `1e-8` 或基于数据范围的相对阈值。

---

### P1-10: 硬编码维度 12

**位置**: `src/dl/data/dataset_v2.py:108`

```python
if X.shape[2] != 12:
    raise ValueError(f"Expected X.shape[2] == 12, got {X.shape[2]}")
```

添加声学特征后通道数可能变化。

**建议**: 改为动态验证或与 `EXPECTED_CHANNEL_NAMES` 长度对齐。

---

### P1-11: matplotlib.use("Agg") 在模块导入时执行

**位置**: `src/ml/patent_model/robustness.py:10-11`

模块导入时立即执行 `matplotlib.use("Agg")`，会全局影响所有后续导入 matplotlib 的模块的 backend 设置。Jupyter 交互式环境导入此模块后绘图会失效。

**建议**: 将 `matplotlib.use("Agg")` 移到 `if __name__ == "__main__"` 块或脚本入口中。

---

### P1-13: CLI 工具函数重复定义

**位置**: `src/ml/scripts/train_patent_model.py:36-50` + `src/ml/scripts/_cli_utils.py:12-26`
**状态**: 已修复。`train_patent_model.py` 已从 `scripts._cli_utils` 导入 `positive_int` 和 `limit_dataset`，不再保留重复定义。

`positive_int` 和 `limit_dataset` 在两处完全相同。

**建议**: `train_patent_model.py` 从 `_cli_utils` 导入。

---

### P1-14: _concat_datasets 不校验 parts 非空

**位置**: `src/ml/patent_model/environment_augmentation.py:36-52`

`_concat_datasets(parts)` 直接取 `parts[0]`，如果 `parts` 为空会抛 `IndexError`。虽然当前调用路径保证非空，但作为公共函数缺少防御。

---

### P1-16: sim_v2 从 V1 脚本导入私有函数

**位置**: `src/sim/sim_v2/dynamics.py`

```python
from scripts.generate_v1_dataset import (
    PROCESSING_PARAMS,
    _fmt,
    _generate_main_features,
)
```

`_generate_main_features`（约70行传感器模型）是 V1 的私有函数，V2/V3 都依赖它：

- 造成循环依赖风险
- 私有函数 `_` 前缀跨包导入破坏封装
- V1 的实现变更会静默影响 V2/V3

**建议**: 将 `_generate_main_features` 和 `PROCESSING_PARAMS` 提取到 `sim_common/sensor_models.py`。

---

### P1-17: 波形条件窗口检查用纯 N2 声速

**位置**: `src/sim/scripts/generate_waveform_dataset.py` 中 `_condition_fits_waveform_window`

```python
c_sound = _hidden_sound_speed_v2(0.0, 0.0, 0.0, 100.0, float(condition["T_C_base"]))
```

用纯 N2 (x_H2=0) 声速做窗口检查，实际混合气（含 H2）声速更高，peak_index 可能超出窗口。

**建议**: 用混合气声速检查，或至少用最大可能声速做保守估计。

---

### P1-18: pipeline 层大量 sys.path.insert hack

**位置**: `src/pipeline/` 下 5 个入口脚本

```python
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
```

每个入口脚本重复相同的路径 hack，`parents[2]` 硬编码层级脆弱。

**建议**: 添加 `pyproject.toml`，用 `pip install -e .` 安装项目，消除所有 `sys.path` hack。

---

### P1-19: 域合并算法无最大迭代保护

**位置**: `src/pipeline/domain_split.py` 中 `_merge_sparse_domains`

```python
while True:
    stats = _domain_stats(merged)
    sparse = stats.loc[stats["sample_count"] < min_domain_samples]
    if sparse.empty or len(stats) <= 1:
        return merged
    ...
```

如果合并不收敛，会无限循环。

**建议**: 添加最大迭代次数保护（如 `max_iterations=100`）。

---

## P2 改进建议

> 不会导致功能问题，但提升代码质量和可维护性。

### P2-D1: V1 仿真脚本与 sim_common 大量代码重复

**位置**: `src/sim/scripts/generate_v1_dataset.py` 与 `src/sim/sim_common/v1_helpers.py`

两个文件中约 10 个函数完全重复，包括 `_condition_rows`、`_condition_row`、`_sample_condition_base`、`_build_condition_row`、`_normalize_components`、`_stage_for_index`、`_sample_hydrogen_percent`、`_pressure_for_stage`、`_path_length_for_stage`、`fmt` 和部分常量。该问题会增加维护成本，但不是当前运行正确性的 P0 缺陷。

**建议**: `generate_v1_dataset.py` 应从 `sim_common.v1_helpers` 导入这些共享函数，只保留 V1 特有的逻辑。

---

### P2-D2: profile 展示命名体系不统一

**位置**: `src/ml/scripts/environment_compensation_common.py` + `src/ml/patent_model/plotting_style.py`

`environment_compensation_common.py` 将实验层 profile（如 `v3_raw_no_env`）映射到实际 feature profile（如 `raw_no_env_four`），该映射是显式存在的，因此不是运行错误。剩余问题是不同模块的展示标签命名体系不完全统一，绘图或报告中可能回退显示原始 key。

**建议**: 统一实验 profile、feature profile 和展示 label 的映射表，避免脚本各自维护名称。

---

### P2-1: 模型配置模式不一致

**位置**: `src/dl/models/`

- `CNN1DRegressor`、`TCNRegressor`：使用显式构造函数参数
- `LSTMRegressor`、`GRURegressor`、`TransformerRegressor` 等：使用 `merge_model_kwargs` + dict

两种模式混用增加认知负担。建议统一为 dict 模式（与 registry 的 `build_model(config)` 接口一致）。

---

### P2-2: 应同时保存 best_model 和 final_model

**位置**: `src/dl/training/train.py`

当前只保存 best model。若 early stopping 的 patience 设置不当，best model 可能来自早期 epoch，最终模型（可能更稳定）被丢弃。

**建议**: 同时保存 `best_model.pt` 和 `final_model.pt`。

---

### P2-3: 逐 batch 构建 DataFrame 效率低

**位置**: `src/dl/training/train.py:119-126`

每个 batch 创建一个 DataFrame，然后用 `pd.concat` 合并。大数据集下产生大量小 DataFrame 对象。

**建议**: 累积 list of dicts，最后一次性构建 DataFrame。

---

### P2-4: PatentDataset 字段过多

**位置**: `src/ml/patent_model/dataset.py`

`PatentDataset` 有 13 个字段，多处手动构造新实例时需逐字段复制（P0-4 的根因）。

**建议**: 添加 `with_replaced(**kwargs)` 方法或使用 `dataclasses.replace()` 统一处理。

---

### P2-5: modeling.py 中 XGBoost 参数重复

**位置**: `src/ml/patent_model/modeling.py:152-195`

`_make_pipeline` 和 `_make_meta_model` 两处 XGBoost 参数完全相同。

**建议**: 提取为 `_xgb_params(config)` 辅助函数。

---

### P2-6: 脚本间 CLI 参数大量重复

**位置**: `src/ml/scripts/` 下多个脚本

`train_patent_model.py`、`run_robustness_analysis.py`、`run_environment_compensation_robustness.py` 等各自定义了几乎相同的参数。

**建议**: 统一到共享模块。

---

### P2-7: generate_v1_dataset.py 过长（1022行）

**位置**: `src/sim/scripts/generate_v1_dataset.py`

包含条件采样、传感器仿真、特征提取、CSV 写入、scaler 拟合、CLI 入口等全部逻辑。

**建议**: 拆分为 `conditions.py` / `sensor_models.py` / `feature_extraction.py` / `io.py` / `cli.py`。

---

### P2-8: acoustic_quality_summary 函数过长（220行）

**位置**: `src/sim/sim_v2/acoustic_derived.py`

该函数遍历所有序列计算多种统计量。建议拆分为 `_compute_amp_stats()`、`_compute_alpha_stats()`、`_compute_fit_stats()` 等。

---

### P2-9: sim_v2 薄壳 re-export 模块无新增逻辑

**位置**: `src/sim/sim_v2/conditions.py`, `io.py`, `scalers.py`, `splits.py`

这些文件是纯 re-export：

```python
from sim_common.conditions import (
    build_synthetic_condition_rows as sequence_condition_rows,
    ...
)
```

增加了间接层但无新增逻辑，调用方可直接从 `sim_common` 导入。

---

### P2-10: YAML 配置文件路径不一致

`configs/deep/fusion_formal.yaml` 使用 `../../data/waveform_v3`，`waveform_only_formal.yaml` 使用 `data/waveform_v3`（无 `../../` 前缀）。不同配置在不同工作目录下行为不一致。

---

### P2-11: 硬编码采样率

**位置**: `src/sim/scripts/generate_traditional_from_waveform_v3.py`

```python
sample_rate_hz = 200000.0
```

应从 `waveform_v3_spec.json` 读取。

---

### P2-12: 条件采样无最大迭代保护

**位置**: `src/sim/sim_common/conditions.py` 中 `build_synthetic_condition_rows`

```python
while len(synthetic_rows) < sequence_count:
    ...
    requested = int(requested * 1.25) + 1
```

若 `_condition_rows` 有 bug 导致永远不产生有效行，会无限循环。

**建议**: 添加最大迭代次数保护。

---

### P2-13: aggregate.py 实验路径硬编码

**位置**: `src/pipeline/aggregate.py`

新増实验时需手动修改 glob 模式列表。

**建议**: 改为从 `outputs/STATUS.tsv` 自动发现。

---

### P2-14: status.py 文件缺失时无友好提示

**位置**: `src/pipeline/status.py`

`STATUS.tsv` 不存在时直接抛异常。作为诊断工具应给出友好提示。

---

### P2-15: 测试覆盖严重不足

当前仅 4 个测试文件，覆盖范围极窄。缺失的关键测试：

- `sim_common` 的条件采样（四组分约束、归一化）
- `sim_v2/dynamics.py` 的通道值计算
- `sim_v2/matrix_builder.py` 的多路径模式
- `domain_split.py` 的域合并逻辑
- `generate_waveform_dataset.py` 的端到端小规模生成
- DL 模型的前向传递形状验证

---

### P2-16: PowerShell 脚本缺少完整日志与收尾处理

6 个 `.ps1` 脚本已设置 `$ErrorActionPreference = "Stop"`，基础失败中断逻辑存在。当前不足是缺少统一的 `try/catch/finally`、时间戳日志和失败上下文记录，长时间实验失败时不利于复盘。

**建议**: 在实验入口添加 `Start-Transcript` 时间戳日志，并用 `try/catch/finally` 记录失败命令、退出码和输出目录。

---

## 架构层面共性问题

### 1. sys.path hack 泛滥

`pipeline/`（5个脚本）、`dl/experiments/`、`ml/scripts/` 等多处使用 `sys.path.insert`。项目缺少正式的包安装机制。

**影响**: 入口脚本路径变更即失效，IDE 无法正确解析导入。

**建议**: 根目录添加 `pyproject.toml`，用 `pip install -e .` 安装。

### 2. 私有函数跨包导入

`sim_v2/dynamics.py` 和 `generate_waveform_dataset.py` 从 V1 脚本导入 `_generate_main_features` 等私有函数。

**影响**: V1 的实现变更会静默改变 V2/V3 的输出。

**建议**: 将共享的传感器模型提取到 `sim_common/sensor_models.py`。

### 3. 仿真核心代码未独立抽取

`_generate_main_features`（约70行传感器模型）作为 V1 脚本中的私有函数，却是 V2/V3 的共同依赖。这是 P2-D1 和 P1-16 的共同根因。

---

## 优点总结

### 数据防泄漏严格

- 按 `mixture_id` 分组划分 train/val/test
- `validate_group_splits` 验证无跨 set 重叠
- 同一配气条件的序列不会同时出现在训练集和测试集

### 内存效率设计到位

- 延迟加载模式：Dataset 只在首次访问时载入数据
- memmap 支持：大数组使用 `mmap_mode="r"` 避免全量载入
- pickle 兼容：数据集可安全序列化（多进程 DataLoader）
- int16 压缩存储：波形数据 int16 + scale factor 还原

### 数据可追溯性强

- ML 链路每步筛选写入 `filter_report`（metadata → physical_range → label_closure → duplicate）
- 每个数据包生成 `quality_summary.json` 和 `README.md`
- 通过 `provenance` 字段追踪数据来源

### 实验设计规范

- Feature profile 机制灵活：字典驱动，新增实验仅需添加条目
- OOF 堆叠实现正确：`GroupKFold` 保证组隔离，支持 `joblib` 并行
- CLS 进度系统健壮：ANSI 渲染 + 线程锁 + 非交互环境自动降级
- 实验脚本幂等：`exp06_reproducibility.ps1` 通过 `Test-Path` 跳过已有结果

### 模块职责清晰

- `dl/` 链路：data / models / training 三层分离
- `ml/` 链路：config / dataset / data_loader / modeling / robustness 各司其职
- `sim/` 链路：sim_common → sim_v2 → scripts 依赖方向正确

### 文档注释充分

- 每个传感器模型有详细物理意义注释（NDIR、TCS、超声、光纤麦克风）
- 时间常数、噪声比例等参数有物理解释
- README 和设计文档完整覆盖实验目标和执行状态

---

## 优先修复路线图

| 状态    | 优先级    | 编号    | 模块  | 修复难度 | 建议                                                                      |
| ----- | ------ | ----- | --- | ---- | ----------------------------------------------------------------------- |
| 已修复   | **P0** | P0-1  | ML  | 低    | 所有 `PatentDataset(...)` 构造处补上 `provenance` 和 `filter_report`            |
| 已部分修复 | **P0** | P0-2  | ML  | 中    | OOF 退化时打印 WARNING；是否增加 `min_groups` 配置仍待决策                              |
| 已修复   | **P1** | P1-D1 | DL  | 中    | 模型添加 `input_format` 类属性替代类名字符串                                          |
| 已修复   | **P1** | P1-D2 | DL  | 中    | Dataset 支持 `preloaded_data` 参数，消除重复加载                                   |
| 已修复   | **P1** | P1-D3 | DL  | 中    | 缺省 scaler 保存到 `output_dir`                                              |
| 已修复   | **P1** | P1-D4 | ML  | 低    | 明确四组分 `x_N2` 来自 condition metadata，并校验 `condition_grid_v1.csv` 中该列存在且非空 |
| 已修复   | **P1** | P1-13 | ML  | 低    | `train_patent_model.py` 从 `_cli_utils` 导入                               |
| 未修复   | **P1** | P1-11 | ML  | 低    | 将 `matplotlib.use("Agg")` 移到脚本入口                                        |
| 已修复   | **P1** | P1-5  | DL  | 低    | seed.py 添加 cudnn 确定性设置                                                  |
| 未修复   | **P1** | P1-16 | Sim | 高    | 提取 `_generate_main_features` 到 `sim_common/sensor_models.py`            |
| 未修复   | **P2** | P2-D1 | Sim | 高    | 重构 `generate_v1_dataset.py`，导入 `v1_helpers` 共享函数                        |
