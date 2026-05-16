# 传统模型代码审查报告

**项目**: 多模态掺氢天然气浓度检测实验系统
**审查日期**: 2026-05-15
**审查范围**: `src/ml/patent_model/`（11 文件）、`src/ml/scripts/`（10 文件）、`src/pipeline/train_traditional.py`、`src/pipeline/feature_extraction.py`、`experiments/exp01_traditional.ps1`、`tests/test_patent_*.py` 等 5 个相关测试
**关联报告**: `CODE_REVIEW.md`（覆盖 sim/dl/ml/pipeline 通用层面），本报告只列入未在原报告中处理或针对传统模型新增的发现

---

## 审查环境

| 项       | 值                                                              |
| ------- | -------------------------------------------------------------- |
| sklearn | 1.8.0（已支持 `GroupKFold(shuffle=True, random_state=...)`，无兼容性问题） |
| numpy   | >=2.0（来自 `src/ml/requirements.txt`）                            |
| xgboost | >=2.0（已支持 `device="cuda"`）                                     |
| pytest  | 已通过 `pytest tests/` 12 项（基于原报告记录，本次未重跑）                        |
| 测试覆盖    | 传统模型核心建模 + grid summary schema + duplicate filter，共约 600 行测试代码 |

---

## 目录

- [P0 严重问题](#p0-严重问题)
- [P1 一般问题](#p1-一般问题)
- [P2 改进建议](#p2-改进建议)
- [测试覆盖缺口](#测试覆盖缺口)
- [优点](#优点)

---

## P0 严重问题

### T-P0-1: V3 嵌入式环境 profile 的鲁棒性评估存在方法学失真

**位置**: `src/ml/patent_model/robustness.py:158-211`

`_uses_embedded_derived_environment(profile)` 当前只识别 `{"derived_env", "derived_env_mc_aug", "derived_env_four"}`：

```python
def _uses_embedded_derived_environment(profile: str) -> bool:
    return profile in {"derived_env", "derived_env_mc_aug", "derived_env_four"}
```

但 V3 体系下，`v3_waveform_dual_channel_env_four`（见 `feature_profiles.py:158-170`）已经把 `T_C / P_MPa / H_RH / T_K / P_kPa / p_H2O_kPa / x_H2O / AH_g_m3 / P_dry_kPa` 嵌入到 acoustic / optical / thermal 三个特征矩阵中（部分还嵌入到 acoustic 的 sound_speed/attenuation/c_sound/c_T_norm 等派生列）。

当调用 `add_profile_environment_noise(dataset, profile="v3_waveform_dual_channel_env_four", ...)` 时，会进入"不嵌入派生环境"分支：

```python
if not _uses_embedded_derived_environment(profile):
    # 只更新 environment 矩阵，不更新 acoustic/optical/thermal 中已嵌入的环境列
    return PatentDataset(..., environment=noisy.environment, ...)
```

结果：环境噪声只加到 `dataset.environment`，但 V3 env profile 的模型输入实际是 acoustic/optical/thermal 矩阵中嵌入的副本，不会感知到噪声。**这条 profile 下的鲁棒性数字会被系统性低估。**

**影响范围**:

- `evaluate_profile_environment_noise()`（鲁棒性曲线核心入口）
- `run_environment_compensation_robustness.py`（环境补偿对比实验）
- 上述函数当前主流程 exp01 只跑 `v3_raw_no_env / v3_raw_tph`，不直接触发；但 `environment_compensation_common.py:PROFILES` 已经把 `v3_env` 列入，一旦 environment compensation 系列脚本启用，结论会失真。

**建议**:

- 把 `_uses_embedded_derived_environment` 改为按 profile 的 `embedded_env_columns` 字段判断（或检测 acoustic_columns/optical_columns/thermal_columns 是否包含环境列名集合的交集），不再用硬编码白名单
- 或者在 `feature_profiles.py` 给每个 profile 显式声明 `env_embedding: bool`，让 robustness 模块直接读取

### T-P0-2: `mc_env_samples` 在 V3 env profile 下接口契约不一致

**位置**: `src/ml/scripts/train_patent_model.py:144-150` 与 `src/ml/scripts/run_four_component_model_grid.py:133`

```python
# train_patent_model.py
def _validate_args(args):
    if args.mc_env_samples > 0 and args.feature_profile not in {"derived_env", "derived_env_four"}:
        raise ValueError("--mc-env-samples is only supported with derived_env profiles.")

# run_four_component_model_grid.py
train_args.mc_env_samples = args.mc_env_samples if profile == "v3_env" else 0
```

grid 脚本的意图：`v3_env` profile 启用 MC 增强；但传到 `train_patent_model.py` 时 `feature_profile=v3_waveform_dual_channel_env_four`，不在 `{derived_env, derived_env_four}` 白名单中，会被 `_validate_args` 拒绝。

**触发条件**: 默认 `mc_env_samples=0` 时不触发；如果用户显式传 `--mc-env-samples 4` 且 `--profiles v3_env`，整个 grid 跑到第一个 v3_env item 时崩溃。

**建议**:

- 在 `_validate_args` 的白名单中加入 `v3_waveform_dual_channel_env_four`
- 或在 grid 端不再对 v3_env 启用 MC 增强（删除 `if profile == "v3_env"`）
- 二选一，并在 `feature_profile` docstring 中显式说明哪些 profile 支持 MC 增强

---

## P1 一般问题

### T-P1-1: `derived_env_mc_aug` 在 V3 主流程下是死代码

**位置**:

- `src/ml/scripts/run_environment_compensation_experiment.py:72-83`
- `src/ml/scripts/run_environment_compensation_robustness.py:87-96`

```python
# environment_compensation_common.py
PROFILES = ("v3_raw_no_env", "v3_raw_tph", "v3_env")

# run_environment_compensation_experiment.py:72
if profile == "derived_env_mc_aug":
    argv.extend(["--mc-env-samples", str(args.mc_env_samples), ...])
```

`PROFILES` 中没有 `derived_env_mc_aug`，循环 `for profile in PROFILES` 时永远不会进入该分支。整组 `--mc-env-samples` / `--mc-env-sigma-*` 参数在 V3 体系下永远不生效。

**建议**:

- 如果保留 V1 兼容，明确写注释说明该分支只在 V1 数据集上有效
- 或者删除 V1 的 mc_aug 分支，把 MC 增强统一迁移到 V3 env profile 上（与 T-P0-2 一并处理）

### T-P1-2: OOF 退化分支仍然继续训练元学习器（信息泄漏）

**位置**: `src/ml/patent_model/modeling.py:371-383`

原 `CODE_REVIEW.md` 的 P0-2 已经把 WARNING 加上，但 fallback 仍然返回 in-sample 预测作为元学习器输入：

```python
if n_splits < 2:
    logger.warning("OOF fallback: only %d unique group(s) ...")
    fitted = {name: SingleModalityMultiOutputModel(...).fit(inputs[name], targets) for ...}
    base_predictions = np.stack([fitted[name].predict(inputs[name]) for name in BRANCH_NAMES], axis=2)
    return base_predictions, _compute_modal_dynamic_weights(...)
```

退化后元学习器仍然在训练，且预测会被当成"OOF 预测"使用——这是隐性信息泄漏。当前 CLI 没有任何阻止这条路径的开关。

**建议**:

- 在 `ModelConfig` 中加 `min_groups_for_stacking: int = 2`，n_groups 不足时直接 `raise ValueError`，让用户显式选择"放弃融合，只保留单模态"或"降低 stacking_folds"
- 或在 fallback 路径中跳过元学习器训练，融合输出回退为单模态等权平均，让 evaluate 阶段也能感知到模型退化

### T-P1-3: `fault_labels.inject_faults` 的光学故障算子可能产生负值

**位置**: `src/ml/patent_model/fault_labels.py:152-154`

```python
if case in {"optical_fail", "mixed_fail"}:
    optical_scale = _column_scale(optical)
    optical += rng.normal(0.0, optical_scale * scale, size=optical.shape)
    optical *= rng.normal(1.0 - scale * 0.25, scale * 0.08, size=optical.shape)
```

`rng.normal(mean, std)` 的乘子分布在 `severity=severe`（scale=0.70）时是 `N(0.825, 0.056)`，左尾约 1e-30 概率出现负值；但 `mild`（scale=0.15）时 `N(0.9625, 0.012)` 很稳定。
即便不取负，`optical *= negative` 也会改变 NDIR 电压信号的符号，物理上不可解释。

**建议**:

- 对乘子做 `np.clip(multiplier, 0.05, 2.0)` 截断
- 或改为 `optical *= (1.0 + rng.normal(...))` 的相对扰动，配合 `np.clip`

### T-P1-4: `_profile_columns(None)` 把训练表所有非 `sample_id` 列吸入特征矩阵

**位置**: `src/ml/patent_model/data_loader.py:65-68`

```python
def _profile_columns(frame, configured):
    if configured is None:
        return [column for column in frame.columns if column != "sample_id"]
    return list(configured)
```

`derived_env` 系列 profile 的 `acoustic_columns / optical_columns / thermal_columns` 都是 `None`，依赖训练表自身列约定。如果上游误把 metadata 列（`mixture_id / pressure_stage / x_H2 ...`）混入训练表 CSV，会被作为特征参与建模，可能造成目标泄漏。

**建议**:

- 增加 deny-list：`mixture_id / stage_id / x_H2 / x_CH4 / x_CO2 / x_N2 / target_sum / sample_id` 等绝不进入特征矩阵
- 或者在 profile 中要求显式声明特征列，移除 `None` 兜底

### T-P1-5: `physical_range_filter=strict` 缺列时延迟报错

**位置**: `src/ml/patent_model/data_loader.py:240-251`

```python
for column, (lower, upper) in PHYSICAL_RANGE_LIMITS.items():
    values = pd.to_numeric(stats_frame[column], errors="coerce")  # 缺列直接抛 KeyError
```

`PHYSICAL_RANGE_LIMITS = {"sound_speed": ..., "attenuation_alpha": ...}` 都来自 `feature_table.csv`。当前 `_require_columns(feature_table, [...], ...)` 没把这两列纳入校验，缺列时只在执行 strict 过滤时才报错。

**建议**: 在 `load_patent_dataset` 中按 `physical_range_filter == "strict"` 提前校验 feature_table 列；非 strict 模式不需要时则放宽。

### T-P1-6: `grouped_train_test_split` 没保证 train_groups >= stacking_folds

**位置**: `src/ml/patent_model/data_loader.py:520-535`

```python
n_test_groups = min(len(groups) - 1, max(1, int(round(len(groups) * test_ratio))))
```

极端 case：`len(groups)=3`, `test_ratio=0.2`, `n_test_groups=1`, train_groups=2；但 `stacking_folds` 默认 5，OOF 拆分会退化到 `n_splits=2`，再加上 T-P1-2 的 in-sample 泄漏路径，结果可信度低。

**建议**: 把 `stacking_folds` 与 `len(groups)` 的耦合关系挪到 `prepare_training_data` 里做前置校验；不满足时报错或自动收敛 `stacking_folds`。

### T-P1-7: `select_pressure_slice` 自动容差兜底可能选回全集

**位置**: `src/ml/patent_model/robustness.py:228-232`

```python
else:
    nearest_pressure = pressure[order[0]]
    tolerance = max(0.005, abs(nearest_pressure - target_pressure_mpa) + 1e-12)
    order = np.flatnonzero(np.abs(pressure - target_pressure_mpa) <= tolerance)
```

当数据集中最接近 target 的样本本身就离 target 较远时，`tolerance` 被放大到 `|nearest - target| + 1e-12`，反而把"目标压力附近"扩成"所有比 nearest 更近的样本"，与 1 atm 邻域意图不符。

**建议**:

- 固定 `tolerance=0.005 MPa`（约 5 kPa），找不到任何样本时显式报错或返回空 + WARNING
- 或要求调用方必须显式传 `max_samples`，移除自动兜底

### T-P1-8: `run_four_component_model_grid.py` 反射式 train_args 构造易腐化

**位置**: `src/ml/scripts/run_four_component_model_grid.py:111-143`

```python
train_args = build_train_parser().parse_args([])  # 拿默认 namespace
train_args.data_dir = ...                          # 逐字段覆盖
train_args.feature_profile = ...
```

如果 `train_patent_model.build_parser()` 新增必填参数，`parse_args([])` 会触发 argparse 默认值（不抛错），但 grid 端不会感知到。后续 train_main 行为可能与单独调用 `train_patent_model.py` 不一致。

**建议**:

- 把 `TrainArgs` 抽成 `@dataclass(frozen=True)`，grid 和 train 入口都从同一份 dataclass 构造 Namespace
- 或为 grid 写一个 `_build_train_namespace(...)` 显式构造，新增字段时编译期就报错

### T-P1-9: grid 的 ProcessPoolExecutor 分支无进度回传 + Windows spawn 风险

**位置**: `src/ml/scripts/run_four_component_model_grid.py:339-344`

```python
with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
    futures = [executor.submit(_run_plan_item, args, output_root, plan_item) for plan_item in plan]
```

- `progress` 对象不能跨进程共享，Windows spawn 模式下子进程根本不会构造 progress
- `args` 中可能含不可序列化对象（当前看是 argparse Namespace，应该 OK，但容易被改坏）
- 任何子进程异常都会以 `future.result()` 抛出，但当前没有 try/except 处理

**建议**: 子进程通过 stdout 写 JSON 行，主进程聚合更新 progress；或显式说明并行模式无 UI。

### T-P1-10: `environment_compensation_common.py` 的 `build_meta_key` 是空壳函数

**位置**: `src/ml/scripts/environment_compensation_common.py:91-95`

```python
def build_meta_key(meta_model_type: str) -> str:
    del meta_model_type
    return "fused"
```

参数 `meta_model_type` 永远不被使用，函数恒返回 `"fused"`。如果未来的元学习器要分多键（如 ridge_fused / xgb_fused），现在这层无意义抽象会误导维护者。

**建议**: 直接删除函数，调用方写常量 `"fused"`。

### T-P1-11: `logging_utils.get_logger` 关掉 propagate 影响 pytest caplog

**位置**: `src/ml/patent_model/logging_utils.py:21`

```python
logger.propagate = False
```

虽然 `test_oof_degenerate_path_emits_warning` 用 `assertLogs("patent_model.modeling")` 能正常工作（直接绑定到该 logger），但 pytest 的 `caplog` fixture 默认捕获根 logger，会因 `propagate=False` 失效，导致集成测试或日志驱动断言写起来不直观。

**建议**: 把 `propagate = False` 改为可配置；或者只在 root logger 未配置时才追加 handler，让 propagate 默认开启。

---

## P2 改进建议

### T-P2-1: PatentDataset 构造方式仍未统一到 `dataclasses.replace()`

**位置**:

- `src/ml/patent_model/fault_labels.py:175-189`（`inject_faults`）
- `src/ml/patent_model/robustness.py:105-120, 176-191, 196-211, 237-252`
- `src/ml/patent_model/environment_augmentation.py:18-33, 36-52, 72-87`

`PatentDataset.with_fault_labels` 已经用了 `replace`，其它 5+ 处仍在手写 13 字段复制。原 `CODE_REVIEW.md` 的 P2-4 已经标过，但截止本次审查仍未集中重构。

**建议**: 给 PatentDataset 加 `with_overrides(**kwargs)` 工厂方法，全部转过去。重构后 T-P0-1 也更容易维护（只改一处 environment 派生列计算）。

### T-P2-2: `_legacy_row_from_summary` / `_row_from_summary` 已是死代码

**位置**: `src/ml/scripts/run_four_component_model_grid.py:146-242`

`_rows_from_summary` 是当前唯一入口；前两者保留兼容字段（如 `dynamic_{meta}_macro_RMSE_pp`），grep 显示没有调用者。

**建议**: 删除 `_legacy_row_from_summary` 和 `_row_from_summary`，避免新人误用旧 schema。

### T-P2-3: `run_environment_compensation_experiment.py` 子调用是 CLI argv 字符串拼接

**位置**: `src/ml/scripts/run_environment_compensation_experiment.py:51-85`

```python
argv = ["--data-dir", str(data_dir), "--feature-profile", resolve_feature_profile_name(profile, args.component_mode), ...]
return train_main(argv)
```

调用同进程的 `train_main` 但走 argv 字符串中转。等价于序列化-反序列化一次，浮点数会经过 `str → float` 转换。建议直接构造 `argparse.Namespace` 或共享 dataclass，省一次反序列化路径。

### T-P2-4: `MultiComponentPatentModel` 别名增加心智负担

**位置**: `src/ml/patent_model/modeling.py:472`

```python
MultiComponentPatentModel = TraditionalFusionModel
```

外部脚本既导入 `TraditionalFusionModel`（train_patent_model.py），又导入 `MultiComponentPatentModel`（run_robustness_analysis.py、run_environment_compensation_robustness.py），同一个类两个名字。

**建议**: 选定 `TraditionalFusionModel`，把别名标 deprecated，下个周期删除。

### T-P2-5: 模态预测扰动 24 次 × 5 noise levels × 5 OOF folds = 600 次扰动评估

**位置**: `src/ml/patent_model/modeling.py:226-251` + `robustness.py:265-275`

对 SVR 这种 O(n_support × n_features) 的预测来说，万级数据上单次 evaluate 已经显著。当前实现：

- fit 阶段：5 fold × 3 modality × 24 perturbation = 360 次基础预测
- predict 阶段：3 modality × 24 perturbation = 72 次基础预测
- 鲁棒性 evaluate：每个 noise level × 72 次

无缓存。当 perturbation_scale=0.04 较小、`n_perturbations=24` 是为了估计权重而非误差时，可考虑：

**建议**:

- 在 `ModelConfig` 中加 `cache_dynamic_weights: bool`，对 evaluate 阶段重复计算的权重做 hash 缓存
- 或允许 noise robustness 评估时 `n_perturbations` 单独配置（用更小值估计权重）

### T-P2-6: `select_pressure_slice` 与 `evaluate_pressure_bins` 的 `pd.cut` 极端情况未处理

**位置**: `src/ml/patent_model/robustness.py:324`

```python
metadata["pressure_bin"] = pd.cut(pressure_values, bins=n_bins)
```

如果 `pressure_values` 全为同一个值（极端 case），`pd.cut` 会抛 `ValueError`。当前依赖上游切分有压力多样性，没有兜底。

**建议**: 计算前先校验 `pressure_values.nunique() >= n_bins`，不满足时降级到 stage 分箱或返回单 bin。

### T-P2-7: `feature_extraction.py` / `train_traditional.py` 仍有 `sys.path.insert` hack

**位置**:

- `src/pipeline/feature_extraction.py:7-10`
- `src/pipeline/train_traditional.py:9-15`
- `src/ml/scripts/train_patent_model.py:10-12`

原 `CODE_REVIEW.md` 的 P1-18 已经标过 pipeline 层，但 ml 层入口同样依赖 `sys.path.insert(0, str(SRC_ROOT))` 把 `src/ml` 直接加到 path，让 `patent_model.xxx` 这种顶层 import 成立。pyproject.toml 当前只声明 `xgboost>=2.0`，没有 `[project] name`、`[tool.setuptools.packages]`，无法 `pip install -e .`。

**建议**: 把 `src/ml/pyproject.toml` 补齐 `[project] name = "patent_model"` + `[tool.setuptools.packages.find]`，让 `import patent_model.modeling` 直接成立，消除所有 ml 入口的 sys.path hack。

### T-P2-8: `environment_compensation_common.require_known_profile` 无调用者

**位置**: `src/ml/scripts/environment_compensation_common.py:98-102`

```python
def require_known_profile(profile: str) -> None:
    if base_feature_profile(profile) not in FEATURE_PROFILES:
        raise ValueError(...)
```

grep 显示只有 `require_known_profile_mode` 有外部调用。

**建议**: 删除 `require_known_profile`，避免维护负担。

### T-P2-9: `fault_labels._acoustic_delay_residuals` NaN 残差被静默转为 False

**位置**: `src/ml/patent_model/fault_labels.py:80-86`

```python
residuals = np.abs(_acoustic_delay_residuals(dataset))  # 可能含 NaN
acoustic_fault = (residuals > residual_threshold) | (attenuation < low).to_numpy() | (attenuation > high).to_numpy()
```

`np.nan > x` 是 `False`，导致小 mixture（<3 样本）的 acoustic_fault 标签恒为 False。如果某个 mixture 实际有故障但因样本不足拟合失败，会被标记为 clean，下游分析失真。

**建议**: 显式区分 unknown / clean，可以在 fault_severity 中加 `unknown_acoustic` 类别，或在 build_observed_fault_labels 中显式校验 `metadata["mixture_id"].value_counts().min() >= 3`。

### T-P2-10: `generate_report_figures.py` 默认 `--robustness-dir` 与 V3 命名脱钩

**位置**: `src/ml/scripts/generate_report_figures.py:46`

```python
parser.add_argument("--robustness-dir", default="outputs/environment_compensation_robustness_four_main_svr_ridge")
```

实际 V3 输出目录见 `outputs/exp01_traditional/`、`outputs/exp02_*` 等，命名规则不同。当前默认路径基本不可用。

**建议**: 改为从 `outputs/STATUS.tsv` 自动发现最近成功的 robustness 实验，或显式标注必须传 `--robustness-dir`。

---

## 测试覆盖缺口

### T-P2-T1: `add_profile_environment_noise` 在 V3 env profile 下无端到端测试

`test_traditional_modal_fusion.py` 用 `include_environment=False` 的小数据集验证融合管线；但 `_uses_embedded_derived_environment(v3_waveform_dual_channel_env_four)` 的失真路径（T-P0-1 核心 bug）没有任何测试触发。

**建议**: 加一条用例：构造一个嵌入 T_C/P_MPa/H_RH 列的 acoustic 矩阵，调用 `add_profile_environment_noise(profile=v3_waveform_dual_channel_env_four, sigma_t=10.0, ...)`，断言 acoustic 矩阵中 T_C 列发生变化。当前这条断言会失败 → 暴露 P0 bug。

### T-P2-T2: `_profile_columns(None)` 路径无测试

`derived_env` / `derived_env_four` profile 的 acoustic_columns=None 路径会让 _profile_columns 返回训练表所有非 sample_id 列。没有测试验证：

- 训练表多列时是否被全部纳入
- metadata 列误入特征列时是否被识别拒绝

### T-P2-T3: `inject_faults` 各 case 行为无系统测试

`test_code_review_fixes.py` 只跑了 `acoustic_bias` + `mild` 的 provenance 保留测试。没有：

- 对每个 case（`clean / optical_fail / thermal_drift / acoustic_bias / mixed_fail`） × 每个 severity 的注入幅度断言
- 注入故障后模型预测是否朝预期方向偏移
- 光学乘子取负的回归测试（T-P1-3）

### T-P2-T4: `run_four_component_model_grid.main()` 整流程无端到端测试

只测了 `_rows_from_summary` 这一个单元函数；ProcessPoolExecutor 分支、整套 grid 串联、`build_execution_plan` 中的 grouped_meta 逻辑都没有覆盖。

### T-P2-T5: OOF fallback 后 evaluate 行为无测试

`test_oof_degenerate_path_emits_warning` 验证了 WARNING 输出，但没有验证：

- fallback 后 fit/predict 整流程能否跑通
- fallback 路径与正常 OOF 路径在同一 dataset 上的指标差距

---

## 优点

- **OOF 隔离严格**：`GroupKFold(shuffle=True, random_state=...)` + `groups=mixture_id` 保证同一配气条件不会跨 train/valid。
- **筛选链可追溯**：`filter_report` 把 metadata_filter → physical_range_filter → label_closure_filter → duplicate_filter 四道闸的 before/after 统计逐项保留。
- **配置类已 frozen**：`ModelConfig` / `PatentDataset` 都是 `@dataclass(frozen=True)`，状态变更通过 `replace` 显式发生（虽然部分模块还没切到 `replace`）。
- **进度回传基础设施**：`pipeline.cli_progress.build_cli_progress` 在 grid 主进程能提供阶段化反馈，幂等性也照顾到了。
- **数据准入显式**：`load_patent_dataset` 的 `_require_columns` 在 read 阶段就把缺列阻断，避免下游隐性 broadcast。
- **fit/predict 缓存设计**：`branch_artifacts` + `prediction_cache_holder` 在 grid 内复用，同 branch_model_type 下多 meta_model_type 的扫描成本被显著降低。
- **测试已覆盖核心断言**：`test_traditional_modal_fusion` 锁死了模态/融合输出 schema、列名约定、`dynamic_weights` 归一化条件（`sum(axis=2) == 1`）。

---

## 优先修复路线图

| 优先级    | 编号     | 模块                           | 估难度 | 建议                                                                     |
| ------ | ------ | ---------------------------- | --- | ---------------------------------------------------------------------- |
| **P0** | T-P0-1 | robustness                   | 中   | 把 `_uses_embedded_derived_environment` 改为按 profile 元数据判定，并加 T-P2-T1 测试 |
| **P0** | T-P0-2 | train_patent_model / grid    | 低   | 把 v3 env profile 加入 `_validate_args` 白名单，或在 grid 端不再启用 MC              |
| **P1** | T-P1-2 | modeling                     | 低   | 给 `ModelConfig` 加 `min_groups_for_stacking`，强失败优先                      |
| **P1** | T-P1-1 | env_compensation scripts     | 低   | 删除或注释 `derived_env_mc_aug` 死分支                                         |
| **P1** | T-P1-3 | fault_labels                 | 低   | 光学乘子截断到 `[0.05, 2.0]`                                                  |
| **P1** | T-P1-4 | data_loader                  | 中   | 给 `_profile_columns` 增加 deny-list                                      |
| **P1** | T-P1-5 | data_loader                  | 低   | strict 物理范围过滤前置列校验                                                     |
| **P1** | T-P1-7 | robustness                   | 低   | 固定 `select_pressure_slice` 容差，不再扩展兜底                                   |
| **P1** | T-P1-8 | grid                         | 中   | TrainArgs dataclass 化                                                  |
| **P2** | T-P2-1 | dataset/fault/robust/env_aug | 中   | 统一切到 `dataclasses.replace`                                             |
| **P2** | T-P2-2 | grid                         | 低   | 删除 `_legacy_row_from_summary`                                          |
| **P2** | T-P2-5 | modeling                     | 中   | 评估扰动权重的缓存                                                              |
| **P2** | T-P2-7 | 项目级                          | 中   | pyproject.toml 包化，消除 sys.path hack                                     |

---

## 备注

- 本报告未实际执行任何代码运行（依赖只读 grep/read），所有"会触发"类结论基于代码静态分析，没有用 V3 真实数据复跑验证。
- T-P0-1 的失真严重性建议用单元测试 T-P2-T1 复现 → 该测试一旦写出来会立刻失败，作为修复前的 baseline。
- 与原 `CODE_REVIEW.md` 重合的项（如 matplotlib.use Agg 模块导入、`_concat_datasets` 非空校验、PatentDataset 字段过多）未在本报告重复展开。
