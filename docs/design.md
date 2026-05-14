# 实验设计文档

> 创建日期：2026-05-14（v0.4，已同步 CLI 训练可视化）
> 项目：V3 正式实验（V3.1 dual-channel waveform ML vs DL 公平对比）

## 1. 三个验证目标

### G1：ML vs DL 检测能力

**问题**：在相同的 V3.1 双通道 waveform 数据上，传统机器学习（信号处理 + 浅模型）与深度学习（端到端从波形学习）哪个更好？

**主指标**：

- macro_RMSE：四组分 RMSE 取均值（主指标）
- macro_MAE：四组分 MAE 取均值
- macro_MRE：仅在 x_H2 / x_CH4 上取均值（低含量组分 MRE 会爆炸，单独算）
- per_component R²：x_H2, x_CH4, x_CO2, x_N2 四个
- sum_error：预测和减 100% 的绝对误差均值与最大值
- seed_std：跨 seed 标准差，反映稳定性

### G2：融合策略对比

**问题**：双通道 waveform 与 slow channel 怎么融合最合适？

**主指标**：macro_RMSE + 参数量 + 训练时长。

### G3：复杂工况适应

**问题**：模型在不同温度 / 压力 / 湿度组合下是否稳定？是否能跨工况迁移？

**主指标**：

- domain_gap = test_holdout_domain - test_in_domain（Track A 留一域）
- degradation_ratio = perturbed_RMSE / clean_RMSE（Track B 环境扰动）
- sample_efficiency = RMSE vs fine-tune 样本数（Track C 跨域微调，**可选**）

## 2. 实验矩阵

### G1 实验矩阵

| 实验组 | 模型 | 输入 | seed | 输出目录 |
| -------- | -------------------------- | -------------------------------------------- | -------- | ---------------- |
| 传统 ML core | PLS + Ridge | 双通道派生特征（TOF / tau / 峰值 / 反射峰统计 + slow） | 42/52/62 | exp01/core |
| 传统 ML core | PLS + XGBoost | 同上 | 42/52/62 | exp01/core |
| 传统 ML core | XGBoost + Ridge | 同上 | 42/52/62 | exp01/core |
| 传统 ML core | XGBoost + XGBoost | 同上 | 42/52/62 | exp01/core |
| 传统 ML diagnostic | SVR (RBF) + Ridge | 同上 | 42 或抽样 | exp01/diagnostic |
| DL 端到端 | MultimodalFusionV3 | ultrasonic[1000] + fiber_mic[2000] + slow[8] | 42/52/62 | exp02/v3_wf/ |
| DL 慢通道基线 | LSTM (slow only) | slow[8] | 42/52/62 | exp02/lstm_slow/ |
| DL 慢通道基线 | TCN (slow only) | slow[8] | 42/52/62 | exp02/tcn_slow/ |
| DL 双波形基线 | WaveformOnlyDual (no slow) | ultrasonic[1000] + fiber_mic[2000] | 42/52/62 | exp02/wf_only/ |

补充说明：

- `SVR (RBF)` 不再作为默认正式全量主网格，因为在当前 `31696` 级训练样本上训练成本过高。
- 正式主线先出 `paper_core` 结果，再补 `svr_diagnostic` 作为对照。
- 传统 ML grid summary 现已带 `fit_seconds`、`evaluate_seconds`、`write_seconds`、`total_seconds`、`prediction_cache_reused`。
- 传统训练与深度训练入口现在都支持内置 CLI 阶段进度显示；交互终端默认开启，可用 `--no-ui` 关闭。

**特征定义**（传统 ML 输入）：

当前传统模型输入由 `src/pipeline/feature_extraction.py` 从双通道数据包导出，整体遵循“分阶段抽样 + 双通道波形提要 + slow 环境补充”的流程：

- 先从每条 120 步双通道序列中抽取少量代表性时间点，覆盖基线段、目标气体稳定段和恢复段，作为传统模型的表格样本。
- 对通道 1 超声波形提取传播时延、主峰幅值和频域峰值等信息，并进一步换算声速与声衰减的近似表征。
- 对通道 2 光纤麦克风波形提取包络衰减时间常数、反射峰间隔、反射峰数量和尾部能量等混响衰减特征。
- 结合 slow 通道中的 NDIR、TCS 与温压湿变量，补充相对基线变化量和环境派生量，形成可直接供浅模型训练的表格特征。
- 最终按模态导出 `train_acoustic.csv`、`train_optical.csv`、`train_thermal.csv`，并同步写出完整 `feature_table.csv` 与 `feature_manifest.json`；需要环境补偿对照时，再导出 `*_env.csv` 与 `feature_table_env.csv`。

### G2 融合策略矩阵

| 策略             | 实现                                                                             | 配置文件                               |
| -------------- | ------------------------------------------------------------------------------ | ---------------------------------- |
| Early fusion   | ultrasonic encoder + fiber_mic encoder → concat slow[8] → temporal head → head | configs/deep/fusion_early.yaml     |
| Middle fusion  | 三支路各自编码后在 fusion layer 汇合                                                      | configs/deep/fusion_middle.yaml    |
| Late fusion    | 波形分支与 slow 分支独立预测 → 学习权重加权                                                     | configs/deep/fusion_late.yaml      |
| ML 单模型对照       | XGBoost 单模型                                                                    | configs/traditional/xgb.yaml       |
| ML stacking 对照 | PLS / XGBoost core 组合                                                        | scripts + runtime combo-list       |
| ML 诊断对照 | SVR + Ridge dynamic fusion                                                     | scripts + runtime combo-list       |

每种策略 3 seed，输出到 `exp03/`。

### G3 复杂工况矩阵

#### Track A：留一域评估

**域定义**：基于 condition_grid 的三维分箱

- T_bin：按 T_C 三分位分 3 档（低 / 中 / 高）
- P_bin：按 P_MPa 三分位分 3 档（低 / 中 / 高）
- RH_bin：按 H_RH 中位数分 2 档（干 / 湿）

初始网格 3×3×2 = 18 域，合并样本量 < 500 的稀疏域，最终约 9 个域。具体合并规则在 `src/pipeline/domain_split.py` 实现并记录到 `outputs/exp04_domain/domain_definition.json`。

**评估方式**：轮流 hold-out 1 域，N-1 域训练。每域每模型 3 seed。

**报告**：

- per_domain_holdout_RMSE
- in_domain_RMSE（同 split 全域随机划分对照）
- domain_gap = holdout_RMSE - in_domain_RMSE

#### Track B：环境扰动注入

**训练增强**：训练时给 `T_C / P_MPa / H_RH` 加 N(0, σ_train) 噪声，σ_train ∈ {0, 0.5, 1.0}。

**测试评估**：测试时在 σ_test ∈ {0, 0.5, 1.0, 2.0} 的扰动下评估。

**报告**：degradation_ratio = perturbed_RMSE / clean_RMSE 随 σ_test 变化曲线。

#### Track C：跨工况微调 [optional]

**地位**：可选实验，Track A 与 Track B 跑完后再评估时间预算。不进默认验收。

**流程**：

1. Track A 的预训练模型（N-1 域上训完）
2. 在 hold-out 域上 fine-tune N=50 / 100 / 200 / 500 样本
3. 报告 fine-tune 后 RMSE 与样本量关系

**报告**：sample_efficiency 曲线（每个模型一条线）。

## 3. 公平对比约束（五同）

| 约束       | 实现位置                                                                              |
| -------- | --------------------------------------------------------------------------------- |
| 同数据源     | `configs/paths.yaml` 锁定 `data/waveform_v3/` 的 `ultrasonic + fiber_mic + slow + y` |
| 同划分      | `data/waveform_v3/splits/*.csv` 固定，按 `mixture_id` 分组，V3.1 生成 seed 为 `20260514`    |
| 同标准化     | slow scaler 仅在 train 上拟合；两路 waveform 不单独存 scaler JSON，只做反量化与 train 统计归一化          |
| 同 seed 集 | [42, 52, 62]，所有模型共用                                                               |
| 同指标      | `src/pipeline/evaluate.py` 统一计算                                                   |

## 4. 验收标准

### G1 验收

- [ ] 所有 ML 与 DL 模型用同一 split、同一 slow scaler、同一 waveform 反量化 / 归一化规则、同一 seed 集运行
- [ ] `outputs/summary/results.tsv` 含每个模型每 seed 的全部指标
- [ ] `outputs/summary/results_multiseed.tsv` 含 mean / std / min / max
- [ ] Paired bootstrap CI + Wilcoxon 检验：`outputs/summary/stat_tests.tsv`

### G2 验收

- [ ] 三种 DL 融合策略 + 2 个 ML 对照，每个 3 seed
- [ ] `outputs/exp03_fusion/` 结果入 `outputs/summary/results.tsv`，含 macro_RMSE + 参数量 + 训练时长
- [ ] 出图：`outputs/summary/fig_fusion_strategy.png`

### G3 验收

- [ ] Track A：至少 9 域 hold-out，每域每模型 3 seed
- [ ] Track A 结果含 domain_gap，入 `outputs/summary/results.tsv`
- [ ] Track B：σ_train × σ_test 完整矩阵
- [ ] Track B 输出 degradation_ratio 曲线表
- [ ] Track C 可选；若执行，输出 sample_efficiency 表与曲线图

### 数据集验收（P0 先决条件）

- [ ] 双通道数据文件齐全：`ultrasonic_*`、`fiber_mic_*`、`slow.npy`、`y.npy`
- [ ] 通道 1 方向性通过：L_m ↑ → peak_index ↑，H2 ↑ → peak_index ↓，CO2 ↑ → 峰幅度 ↓
- [ ] 通道 2 方向性通过：CO2 / H2O / H2 ↑ → tau ↓，L_m ↑ → T_round ↑
- [ ] `quality/waveform_quality_summary.json` 中关键项为 passed

## 4.5 代码原则

研究型实验项目，**不做防御性编程**：

- 不写 try/except，错误直接抛栈
- 不做路径 / 参数 / 文件存在性兜底
- 不写“配置缺失就用默认值”的逻辑，缺就让它崩
- 不写自动回退或兼容模式
- 实验脚本和 pipeline 模块只走 happy path
- 文档、状态控制、版本锁注释照常做，但代码本身保持最短

## 5. 控制与反馈机制

| 机制            | 实现                                                                                                                      |
| ------------- | ----------------------------------------------------------------------------------------------------------------------- |
| 单一状态真源        | `outputs/STATUS.tsv` — 每实验每 seed 一行                                                                                     |
| 状态字段          | `exp_id / model / seed / status / started / finished / macro_RMSE / notes`，status ∈ {running, success, failed, skipped} |
| 状态汇总          | `python src/pipeline/status.py`                                                                                         |
| 单实验 stdout 格式 | `[exp_id.model.seedN] OK macro_RMSE=X.XXX took=Ys -> outputs/...` 或 `FAIL reason=... -> log path`；交互终端下训练入口可切换为单屏 CLI 进度界面 |
| 结果统一表         | `outputs/summary/results.tsv` — 所有实验跑完 append                                                                           |
| 拷贝代码版本锁       | 文件头加 `# Copied from <path> on <date>`，本地修改记 `docs/code_changelog.md`                                                    |

## 6. 关键风险与对策

| 风险                       | 对策                                                        |
| ------------------------ | --------------------------------------------------------- |
| 双通道链路物理参数未校准             | 数据卡片明确标注 `calibration_status: pending`，论文中只声明链路结构对齐       |
| 数据包体积从 2.4 GB 升到约 7.2 GB | 必要时把 fiber_mic 窗口从 10 ms 降到 5 ms                          |
| MRE 在低含量组分上爆炸            | 主表用 macro_MRE 仅算 x_H2/x_CH4；macro_SMAPE 报全部组分             |
| 留一域样本失衡                  | 先看 condition_grid 分布，合并样本 < 500 的稀疏域                      |
| 传统 ML 旧特征定义不再适配          | 新写 `extract_dual_waveform_features.py`，明确定义 TOF/tau/反射峰特征 |
| DL 双通道长度不一致              | encoder 侧用统一 embedding 输出，不在上游硬裁剪对齐                       |

## 7. 代码拷贝清单（P0 第二步执行）

拷贝代码前必须检查代码的正确性，保证逻辑流畅，不出现复杂逻辑干简单活。

### A. src/dl/（来自 `深度学习测试/sequence-models/`）

```
src/dl/
├── data/
│   ├── __init__.py
│   ├── channel_groups.py
│   ├── dataset_v2.py               # 仅保留工具函数
│   ├── dataset_waveform.py         # 需要改成双通道加载与反量化
│   ├── scaler_utils.py
│   └── split_utils.py
├── models/
│   ├── __init__.py
│   ├── acoustic_waveform_encoder.py
│   ├── multimodal_fusion_v3.py     # 需要加 fiber_mic encoder 分支
│   ├── registry.py
│   ├── config_utils.py
│   ├── lstm.py / tcn.py / gru.py / cnn1d.py
│   ├── cnn_lstm.py / transformer_encoder.py
│   └── branch_fusion.py
├── training/
│   ├── __init__.py
│   ├── train.py
│   ├── early_stopping.py
│   ├── losses.py
│   ├── metrics.py
│   └── seed.py
└── experiments/
    └── run_waveform_baselines.py
```

### B. src/ml/（来自 `深度学习测试/traditional-models/patent-sim-model/`）

```
src/ml/
├── patent_model/                   # 整包拷贝
├── scripts/                        # 训练 + 网格 + 鲁棒性入口
├── requirements.txt
└── pyproject.toml
```

需要新增：

- `patent_model/feature_profiles.py` 增加 `v3_waveform_dual_channel` profile
- `src/sim/scripts/extract_dual_waveform_features.py` 作为新特征导出入口

### C. src/sim/（来自 `深度学习测试/simulation-data/`）

```
src/sim/
├── scripts/
│   ├── generate_waveform_dataset.py
│   ├── acoustic_waveform_v3.py          # 保留通道 1 逻辑，可重命名内部函数
│   ├── acoustic_fiber_mic_v3.py         # 新建：通道 2 混响衰减链路
│   ├── acoustic_v2.py
│   ├── generate_v1_dataset.py
│   ├── check_waveform_directionality.py # 扩展到双通道检查
│   └── extract_dual_waveform_features.py
├── sim_common/                          # 整包
├── sim_v2/                              # 整包
└── lut/                                 # NDIR 查找表
```

### D. src/pipeline/（新写）

```
src/pipeline/
├── __init__.py
├── status.py                 # 已建（状态汇总）
├── shim.py                   # sys.path 注入，让 dl/ml/sim 互相 import
├── feature_extraction.py     # 包装双通道特征导出
├── cli_progress.py          # 训练命令行进度渲染
├── train_traditional.py      # 包装 src/ml 训练 + CLI 进度
├── train_deep.py             # 包装 src/dl/training/train.py + CLI 进度
├── domain_split.py           # P-T-RH 分箱与留一域 split
├── perturbation.py           # 训练/测试时环境扰动注入
├── evaluate.py               # 统一 RMSE / MRE / R² / SMAPE / bootstrap CI
└── aggregate.py              # 各实验结果 append 到 outputs/summary/results.tsv
```

## 8. 时间预算

| Phase  | 任务                                           | 估时          |
| ------ | -------------------------------------------- | ----------- |
| P0 第二步 | 双通道仿真代码改造 + 数据生成 + 写 shim/feature_extraction | 2-3 天       |
| P1     | 传统 ML 基线（core 4 组合 × 3 seed + diagnostic）    | 1-2 天 + CPU |
| P2     | DL 端到端基线（4 配置 × 3 seed）                      | 1-2 天 + GPU |
| P3     | 融合策略对比（3 策略 × 3 seed）                        | 1-2 天 + GPU |
| P4     | Track A 留一域（9 域 × 多模型）                       | 2-3 天 + GPU |
| P5     | Track B 环境扰动                                 | 2-3 天 + GPU |
| P6     | Track C 跨域微调（可选）                             | 1-2 天 + GPU |
| P7     | 结果汇总 + 论文图                                   | 1 天         |

合计 9-14 天 + 多批 GPU 时长（Track C 可选不计）。

## 9. 论文图清单（P7 产出）

| 图号  | 主题                         | 来源数据                           |
| --- | -------------------------- | ------------------------------ |
| F1  | ML vs DL 主对比（条形图 + 误差棒）    | summary/results.tsv（多 seed 聚合） |
| F2  | per-component R² 雷达图       | summary/results.tsv            |
| F3  | 融合策略对比                     | summary/results.tsv（exp03 子集）  |
| F4  | 留一域 domain_gap 热图          | summary/results.tsv（exp04 子集）  |
| F5  | 环境扰动退化率曲线                  | summary/results.tsv（exp05 子集）  |
| F6  | 跨域微调 sample efficiency（可选） | summary/results.tsv（exp06 子集）  |




