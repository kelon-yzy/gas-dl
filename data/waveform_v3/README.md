# V3.1 dual-channel waveform 数据包

```text
dataset_version: V3.1 dual-channel waveform
calibration_status: pending
simulation_level: dual_waveform_dynamic_simulation
storage_format: both
sequences: 16
timesteps: 12
ultrasonic_waveform_samples: 1000
fiber_mic_waveform_samples: 2000
slow_channels: 8
labels: x_H2 / x_CH4 / x_CO2 / x_N2
```

## 文件结构

```text
waveform_v3/
  README.md
  sequence_index.csv
  condition_grid_sequence.csv
  sequences/
    slow_sequence_long.csv
    ultrasonic_int16.npy
    ultrasonic_scale.npy
    fiber_mic_int16.npy
    fiber_mic_scale.npy
    slow.npy
    waveform_sequence.npz
  labels/
    y.npy
    sequence_labels.csv
  metadata/
    sequence_ids.npy
    slow_channel_names.npy
    label_names.npy
    waveform_v3_spec.json
  splits/
    train_sequence_ids.csv
    val_sequence_ids.csv
    test_sequence_ids.csv
  scalers/
    scaler_slow_sequence.json
    scaler_slow_sequence_modal.json
  quality/
    waveform_quality_summary.json
```

| split | sequence_count | mixture_count |
| --- | ---: | ---: |
| train | 13 | 4 |
| val | 1 | 1 |
| test | 2 | 2 |
