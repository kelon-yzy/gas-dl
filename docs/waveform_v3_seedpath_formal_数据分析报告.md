# `waveform_v3_seedpath_formal` 数据分析报告

> 生成日期：2026-05-19
> 数据目录：`data/waveform_v3_seedpath_formal`
> 数据版本：`V3.1 dual-channel waveform`
> 分割策略：`stratified_group_by_mixture_id_with_extrapolation_holdout`

## 1. 本次处理范围

本次已完整重新生成 [waveform_v3_seedpath_formal](D:/mydate/项目资料__多模态掺氢天然气/04_代码与实验/code/V3_正式实验/data/waveform_v3_seedpath_formal)，并将旧损坏包隔离到 [waveform_v3_seedpath_formal_corrupted_20260519](D:/mydate/项目资料__多模态掺氢天然气/04_代码与实验/code/V3_正式实验/data/waveform_v3_seedpath_formal_corrupted_20260519)。

本次生成使用固定参数：

```text
seed=20260514
sequence_count=10000
noise_seed_count=3
timesteps=120
storage=memmap
multi_path_phase=steady
```

生成流程末尾已接入 hard gate，自动检查行数、整序列全零、slow 与标签/工况一致性。检查报告位于 [waveform_integrity_report.json](D:/mydate/项目资料__多模态掺氢天然气/04_代码与实验/code/V3_正式实验/data/waveform_v3_seedpath_formal/quality/waveform_integrity_report.json)。

## 2. 数据规模

| 项目 | 数值 |
| --- | ---: |
| 总序列数 | `30000` |
| base condition 数 | `10000` |
| 每序列时间步 | `120` |
| 慢变量通道数 | `8` |
| 标签维度 | `4` |
| 超声波形长度 | `1000` |
| 光纤麦克风波形长度 | `2000` |
| 声学采样率 | `200 kHz` |
| `noise_seed_count` | `3` |

主要数组文件均已写满且通过全零检查：

| 文件 | shape | 全零序列数 |
| --- | --- | ---: |
| `slow.npy` | `[30000, 120, 8]` | `0` |
| `ultrasonic_scale.npy` | `[30000, 120]` | `0` |
| `fiber_mic_scale.npy` | `[30000, 120]` | `0` |
| `ultrasonic_int16.npy` | `[30000, 120, 1000]` | `0` |
| `fiber_mic_int16.npy` | `[30000, 120, 2000]` | `0` |

## 3. Split 结果

所有 split 以 `mixture_id` 为最小分组单位，同一 `mixture_id` 不跨集合。

| split | sequence_count | mixture_count | sequence_ratio |
| --- | ---: | ---: | ---: |
| train | `17466` | `4052` | `0.5822` |
| val | `4026` | `551` | `0.1342` |
| test | `4008` | `586` | `0.1336` |
| extrapolation | `4500` | `238` | `0.1500` |

边界外推候选使用全量数据上下 `10%` 分位阈值：

| 变量 | low | high |
| --- | ---: | ---: |
| `x_H2` | `0.8918` | `21.4350` |
| `x_CO2` | `0.8306` | `10.3753` |
| `x_N2` | `1.9551` | `18.0756` |
| `P_MPa_base` | `0.1600` | `0.6464` |
| `L_m_base` | `0.2` | `1.4` |

外推候选池与抽样结果：

| 项目 | 数值 |
| --- | ---: |
| 边界候选 `mixture_id` 数 | `3686` |
| 实际选入外推集 `mixture_id` 数 | `238` |
| 边界候选序列数 | `24777` |
| 实际外推序列数 | `4500` |

## 4. 完整性检查

完整性 hard gate 结果：`passed`。

关键 slow/condition 一致性：

| 检查项 | 数值 |
| --- | ---: |
| `corr(V_NDIR_CH4, x_CH4)` | `-0.979241` |
| `corr(V_NDIR_CO2, x_CO2)` | `-0.993054` |
| `corr(T_C, T_C_base)` | `0.999999999999` |
| `corr(P_MPa, P_MPa_base)` | `0.999999999999` |
| `corr(H_RH, H_RH_base)` | `0.999999999999` |
| `T_C` 最大绝对误差 | `1.72e-05` |
| `P_MPa` 最大绝对误差 | `3.28e-07` |
| `H_RH` 最大绝对误差 | `3.96e-05` |

说明：NDIR 原始电压随吸收增强而降低，因此 `V_NDIR_CH4` 与 `x_CH4`、`V_NDIR_CO2` 与 `x_CO2` 是强负相关，这是当前仿真模型的正常方向。

## 5. 标准化策略

当前 slow scaler 已按新 `train split` 重新拟合。

证据文件：

- [scaler_slow_sequence.json](D:/mydate/项目资料__多模态掺氢天然气/04_代码与实验/code/V3_正式实验/data/waveform_v3_seedpath_formal/scalers/scaler_slow_sequence.json)
- [scaler_slow_sequence_modal.json](D:/mydate/项目资料__多模态掺氢天然气/04_代码与实验/code/V3_正式实验/data/waveform_v3_seedpath_formal/scalers/scaler_slow_sequence_modal.json)

关键字段：

- `method = z_score`
- `fit_scope = train_split_only`
- `transform_target = slow`

## 6. 物理方向性检查

方向性报告位于 [waveform_directional_report.json](D:/mydate/项目资料__多模态掺氢天然气/04_代码与实验/code/V3_正式实验/data/waveform_v3_seedpath_formal/quality/waveform_directional_report.json)。

结论：

- 超声 `L_m ↑ -> peak_index ↑` 通过，Pearson `r = 0.999999`
- 超声 `x_H2 ↑ -> peak_index ↓` 通过，Pearson `r = -0.997119`
- 超声 `x_CO2 ↑ -> peak_abs ↓ / alpha ↑` 通过
- 光纤麦克风 `x_CO2/H_RH/x_H2 ↑ -> tau ↓` 通过
- 光纤麦克风 `L_m ↑ -> t_round ↑` 通过，Pearson `r = 1.0`

## 7. 结论

新 `waveform_v3_seedpath_formal` 已完成完整重生成，并通过 hard gate、方向性检查和测试回归。旧数据包确认损坏并已隔离，后续训练应只使用新的 formal 目录。

需要注意的是，当前四路 split 仍会把 `15%` 序列抽为 `extrapolation`，训练集约占总序列 `58.22%`。如果后续更关注训练性能，可以按前面讨论改为主 `train/val/test=70/15/15`，外推集独立 benchmark 的方案。
