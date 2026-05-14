# V3.1 双通道 Waveform 数据卡片

> 创建日期：2026-05-14
> 数据状态：设计已冻结，数据待生成
> 数据版本：`V3.1 dual-channel waveform`
> 目标路径：`V3_正式实验/data/waveform_v3/`
> calibration_status：`pending`

## 数据规格

```yaml
dataset_version: V3.1 dual-channel waveform
calibration_status: pending
simulation_level: dual_waveform_dynamic_simulation
storage_format: npy_int16_plus_scale
sequences: 10000
timesteps: 120
ultrasonic_waveform_samples: 1000
fiber_mic_waveform_samples: 2000
slow_channels: 8
labels: [x_H2, x_CH4, x_CO2, x_N2]
generation_seed: 20260514
total_size: ~7.2 GB
```

## 文件结构（生成后）

```
data/waveform_v3/
├── README.md
├── sequence_index.csv
├── condition_grid_sequence.csv
├── sequences/
│   ├── slow_sequence_long.csv
│   ├── slow.npy                     # float32, [10000, 120, 8]
│   ├── ultrasonic_int16.npy        # int16,   [10000, 120, 1000]
│   ├── ultrasonic_scale.npy        # float32, [10000, 120]
│   ├── fiber_mic_int16.npy         # int16,   [10000, 120, 2000]
│   ├── fiber_mic_scale.npy         # float32, [10000, 120]
│   └── waveform_sequence.npz       # 可选 legacy 整合包，含 6 个数组
├── labels/
│   ├── y.npy                       # float32, [10000, 4]
│   └── sequence_labels.csv
├── metadata/
│   ├── sequence_ids.npy
│   ├── slow_channel_names.npy
│   ├── label_names.npy
│   └── waveform_v3_spec.json
├── splits/
│   ├── train_sequence_ids.csv
│   ├── val_sequence_ids.csv
│   └── test_sequence_ids.csv
├── scalers/
│   ├── scaler_slow_sequence.json
│   └── scaler_slow_sequence_modal.json
└── quality/
    └── waveform_quality_summary.json
```

## 张量定义

| 名称 | 类型 | 形状 | 说明 |
| --- | --- | --- | --- |
| ultrasonic | int16 | [10000, 120, 1000] | 通道 1 量化后的超声对射波形 |
| ultrasonic_scale | float32 | [10000, 120] | 通道 1 反量化因子 |
| fiber_mic | int16 | [10000, 120, 2000] | 通道 2 量化后的光纤麦克风波形 |
| fiber_mic_scale | float32 | [10000, 120] | 通道 2 反量化因子 |
| slow | float32 | [10000, 120, 8] | 慢通道，每序列 120 时步 |
| y | float32 | [10000, 4] | 四组分体积分数（vol%），加和约 100 |

反量化规则：`float_wave = int16_wave × scale[:, :, None]`，两路独立。

## 两条物理链路

### 通道 1：超声对射波形

- 物理职责：承载 TOF 主信息
- 生成模型：发射 burst 经气体传播后到达接收端
- 主要观测量：主峰位置、主峰幅值
- 测量窗：5 ms
- sample_rate：200 kHz
- waveform_samples：1000

### 通道 2：光纤麦克风波形

- 物理职责：承载声衰减时间常数 tau 主信息
- 生成模型：直达波 + 多次反射混响叠加
- 主要观测量：包络衰减时间常数、反射峰间隔、反射峰计数
- 测量窗：10 ms
- sample_rate：200 kHz
- waveform_samples：2000

## 慢通道顺序（slow 第三维）

| 索引  | 通道名               | 模态归属 | 单位/范围   |
| --- | ----------------- | ---- | ------- |
| 0   | V_NDIR_CH4        | 光学   | V       |
| 1   | V_NDIR_CO2        | 光学   | V       |
| 2   | V_TCS             | 热导   | V       |
| 3   | T_C               | 环境   | °C      |
| 4   | P_MPa             | 环境   | MPa     |
| 5   | H_RH              | 环境   | %RH     |
| 6   | L_m               | 环境   | m（声程） |
| 7   | piston_position_m | 环境   | m（活塞位置） |

环境通道是 G3 复杂工况验证的核心载体，**全部保留**。

## 物理参数

### 共用参数

| 项 | 值 |
| --- | --- |
| 载频 | 40 kHz |
| 采样率 | 200 kHz |
| 激励 | Hanning 8-cycle burst |
| ADC | 16-bit |
| timesteps | 120 |
| dt_s | 1.0 s |
| calibration | pending |

### 通道 1 专属参数

| 项 | 值 |
| --- | --- |
| measurement_window_s | 0.005 |
| waveform_samples | 1000 |
| noise_std_us | 1e-3 V |

### 通道 2 专属参数

| 项 | 值 |
| --- | --- |
| measurement_window_s | 0.010 |
| waveform_samples | 2000 |
| l_direct_factor | 0.5 |
| wall_reflection_coef | 0.5 |
| max_reflections | 15 |
| noise_std_fm | 1e-3 V |

## Split 策略

- 按 `mixture_id` 分组，避免同 mixture 跨 split
- 比例约为 `7 / 1.5 / 1.5`
- V3.1 数据集使用新的生成 seed：`20260514`
- 文件名保持不变：`train_sequence_ids.csv` / `val_sequence_ids.csv` / `test_sequence_ids.csv`

## Scaler 策略

| 数据 | Scaler | 拟合范围 |
| --- | --- | --- |
| slow [N, 120, 8] | ChannelStandardScaler（按通道 z-score） | 仅 train split |
| ultrasonic 波形 | 不单独保存 scaler；训练时先 `int16 × scale` 反量化，再用 train 统计归一化 | 仅 train split |
| fiber_mic 波形 | 同上 | 仅 train split |

slow scaler 文件保持：`scaler_slow_sequence.json`, `scaler_slow_sequence_modal.json`。

## metadata 规范

`metadata/waveform_v3_spec.json` 至少包含：

```json
{
  "dataset_version": "V3.1",
  "channels": {
    "ultrasonic": {
      "sample_rate_hz": 200000,
      "center_frequency_hz": 40000.0,
      "burst_cycles": 8,
      "measurement_window_s": 0.005,
      "waveform_samples": 1000,
      "noise_std_v": 1e-3,
      "adc_max_int16": 32767,
      "calibration_status": "pending"
    },
    "fiber_mic": {
      "sample_rate_hz": 200000,
      "center_frequency_hz": 40000.0,
      "burst_cycles": 8,
      "measurement_window_s": 0.010,
      "waveform_samples": 2000,
      "noise_std_v": 1e-3,
      "adc_max_int16": 32767,
      "l_direct_factor": 0.5,
      "wall_reflection_coef": 0.5,
      "max_reflections": 15,
      "calibration_status": "pending"
    }
  },
  "slow_channels": [
    "V_NDIR_CH4", "V_NDIR_CO2", "V_TCS",
    "T_C", "P_MPa", "H_RH", "L_m", "piston_position_m"
  ],
  "labels": ["x_H2", "x_CH4", "x_CO2", "x_N2"],
  "sequences": 10000,
  "timesteps": 120,
  "dt_s": 1.0
}
```

## 方向性与质量检查

由 `src/sim/scripts/check_waveform_directionality.py` 输出 `quality/waveform_quality_summary.json`。

### 通道 1 必查项

| 检查 | 通过条件 |
| --- | --- |
| L_m ↑ → peak_index ↑（c_sound 固定） | 单调，Pearson r > 0.99 |
| c_sound ↑（升 H2 含量） → peak_index ↓（L 固定） | 单调下降 |
| alpha ↑（升 CO2） → 峰幅度 ↓ | 单调下降 |
| 噪声 SNR | > 40 dB |

### 通道 2 必查项

| 检查 | 通过条件 |
| --- | --- |
| alpha ↑（升 CO2 / H2O / H2） → 包络 tau ↓ | 单调，Pearson r < -0.85 |
| L_m ↑ → 反射峰间隔 T_round ↑ | 单调 |
| R 调高（人造测试） → 包络尾巴更长 | 单调 |
| 反射峰可识别（前 3 次） | 在 90% 序列中可见 |
| tau 估计与 ground truth `1/(alpha·c)` 相关 | Pearson r > 0.90 |

质量文件至少包含：

```json
{
  "ultrasonic": {
    "peak_index_distribution": { "min": 0, "max": 0, "mean": 0, "p95": 0 },
    "peak_amplitude_v": { "min": 0, "max": 0, "mean": 0 },
    "snr_db_estimate": 0,
    "tof_directionality_passed": true
  },
  "fiber_mic": {
    "decay_tau_ms_distribution": { "min": 0, "max": 0, "mean": 0 },
    "envelope_peak_count_mean": 0,
    "snr_db_estimate": 0,
    "alpha_directionality_passed": true
  }
}
```

## 工况采样与配气

完全复用 V3.0 逻辑，不改 `condition_grid_sequence.csv` 的字段：

| 字段 | 说明 |
| --- | --- |
| sequence_id | Q000001 ~ Q010000 |
| mixture_id | 同 mixture 下多 L_m 配置 |
| x_H2, x_CH4, x_CO2, x_N2 | 四组分，加和 100 |
| T_C_base, P_MPa_base, H_RH_base | 工况基线 |
| L_m_base | 0.2, 0.6, 1.0, 1.4 m 阶梯 |
| status | synthetic_measurement |

## 数据来源与版本

- 生成脚本：`src/sim/scripts/generate_waveform_dataset.py`
- 通道 1 物理模型：`src/sim/scripts/acoustic_waveform_v3.py`
- 通道 2 物理模型：`src/sim/scripts/acoustic_fiber_mic_v3.py`
- 共用物理函数：`src/sim/scripts/acoustic_v2.py`
- 参考设计：`docs/数据集生成设计.md`
- 生成 seed：`20260514`
- 冻结策略：数据生成完成后冻结，不与 V3.0 兼容

## 已知问题

| 问题 | 影响 | 处理 |
| --- | --- | --- |
| calibration_status: pending | 还不能声称等价真实硬件采集 | 在论文与文档中只声明链路结构对齐 |
| 通道 2 反射系数 R 未校准 | tau / 尾部形态存在模型误差 | 把 R 写入 metadata，后续可重生成 |
| 光纤探头位置 l_direct_factor=0.5 是设计值 | 真实几何敏感性未验证 | 写入 metadata，必要时做参数扫描 |
| 数据体积较大 | 生成、训练、存储成本上升 | 必要时把 fiber_mic 窗口从 10 ms 缩到 5 ms |

## 不在本数据包中

- V3.0 单通道 `waveform_int16.npy` / `waveform_scale.npy` 命名接口
- V1 三组分表格数据
- V2 12 通道序列数据（output_sequence_n2）
- 老仓库的中间产物（output_waveform_traditional*）

如需对照，需在 `../深度学习测试/` 中查询。
