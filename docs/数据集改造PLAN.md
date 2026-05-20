# Mixture-Level Stratified Split With Extrapolation Holdout

## Summary

将 V3.1 数据划分改为 `train / val / test / extrapolation` 四部分。所有划分以 `mixture_id` 为最小单位，同一 `mixture_id` 下的全部序列只能进入一个集合。常规 `train/val/test` 在 `mixture_id` 分组基础上做多变量分层，尽量保持 H2、CO2、N2、压力和声程分布一致。外推测试集从边界浓度、边界压力、边界声程候选组中抽取，用于复杂工况鲁棒性评估。所有 scaler 只用 train 拟合。

## Key Changes

- Split 语义改为 `group_field = mixture_id`。
  - `mixture_id` 是唯一分组键；不再引入 `condition_group_id`。
  - `splits/*.csv` 至少保留 `sequence_id, mixture_id`。
  - 新增 `splits/extrapolation_sequence_ids.csv`。
  - 新增 `splits/split_summary.json`，记录策略、阈值、分箱、各 split 数量和分布。

- 外推集生成。
  - 先按全量数据计算 H2、CO2、N2、`P_MPa_base`、`L_m_base` 的 10% 与 90% 分位阈值。
  - 任一序列命中任一变量上下 10% 尾部时，其所属 `mixture_id` 进入外推候选池。
  - 从候选池按 `mixture_id` 抽取约全量序列数 `15%` 作为 `extrapolation`，抽取时保持边界原因覆盖均衡。
  - 被选中的 `mixture_id` 整组移出常规划分，不再进入 train/val/test。

- 常规集分层划分。
  - 在剩余 `mixture_id` 上划分 `train/val/test = 70/15/15`。
  - 分层变量为 H2、CO2、N2、`P_MPa_base`、`L_m_base`。
  - 每个变量按剩余常规池的三分位分成 `low/mid/high`。
  - 对每个 `mixture_id` 统计其内部序列在各变量分箱中的计数向量。
  - 用确定性 greedy allocator 分配 group，使 train/val/test 的序列数和五个变量的分箱计数同时接近目标比例。
  - seed 使用数据集生成 seed `20260514`，保证可复现。

- 生成流程更新。
  - `src/sim/sim_common/splits.py` 增加四路 split 生成函数，输入 conditions、group field、分层变量、外推比例和 seed。
  - `src/sim/scripts/generate_waveform_dataset.py` 调用新 split 函数，写出四个 split 文件和 summary。
  - `compute_split_distribution` 支持 `extrapolation`，质量摘要中记录 `split_policy = stratified_group_by_mixture_id_with_extrapolation_holdout`。
  - `README.md` 和 `waveform_quality_summary.json` 同步新 split 统计。

- 当前数据包更新。
  - 基于现有 `data/waveform_v3/condition_grid_sequence.csv` 重新生成 `splits/train_sequence_ids.csv`、`val_sequence_ids.csv`、`test_sequence_ids.csv`、`extrapolation_sequence_ids.csv`。
  - 用新 train split 重新拟合 `scalers/scaler_slow_sequence.json` 和 `scalers/scaler_slow_sequence_modal.json`。
  - 当前旧训练结果不回写；它们仍属于旧 split 结果，后续正式比较需要基于新 split 重训。

- 训练读取兼容。
  - `src/dl/data/split_utils.py` 继续要求 train/val/test 三个文件存在。
  - 如果存在 `extrapolation_sequence_ids.csv`，加载为额外 split；不影响现有 train/val/test 训练流程。
  - overlap 校验对所有已加载 split 生效，确保任何 `mixture_id` 不跨集合。

## Test Plan

- 单元测试 split 约束。
  - 同一 `mixture_id` 的多条序列必须全部落入同一个 split。
  - train/val/test/extrapolation 之间不得有 `mixture_id` overlap。
  - 没有 extrapolation 文件的旧数据仍可按原 train/val/test 加载。

- 单元测试外推策略。
  - 人造数据中边界 H2、CO2、N2、压力、声程样本应进入外推候选池。
  - 外推集抽取按 group 执行，不允许只抽单条序列。
  - 外推集规模接近目标 `15%`，允许因 group 粒度产生小幅偏差。

- 单元测试分层策略。
  - 构造多 `mixture_id` 数据，验证常规 train/val/test 的分箱计数接近目标比例。
  - 验证分箱阈值和抽样结果在相同 seed 下稳定。

- 当前数据包验证。
  - 统计四个 split 的序列数、`mixture_id` 数、H2/CO2/N2/P/L 分布。
  - 验证 scaler 的 `fit_scope` 仍为 `train_split_only`。
  - 运行 `python -m pytest tests`。

## Assumptions

- 外推集目标规模固定为全量序列约 `15%`，从边界候选池抽样，而不是把全部边界候选都移出常规集。
- 常规 split 比例 `70/15/15` 是对扣除 extrapolation 后的剩余 group 执行。
- `mixture_id` 作为最小隔离单位是硬约束，分层均衡只能在该约束内尽量优化。
- 当前 `mixture_id` 下组分不完全一致的问题先作为数据语义现状接受；本次重点是保证同一 `mixture_id` 不跨 split，并通过分层降低浓度和工况分布偏斜。
