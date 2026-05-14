# V2 数据包 README 自动生成。

from .constants import (
    LABEL_FIELDS,
    MULTI_PATH_PHASE_OFF,
    SEQUENCE_CHANNELS,
)


def write_readme(
    path,
    sequence_count,
    timesteps,
    split_distribution,
    label_fields=LABEL_FIELDS,
    dataset_dir_name="output_sequence",
    dataset_version="V2 sequence",
    acoustic_version="v1",
    multi_path_phase=MULTI_PATH_PHASE_OFF,
):
    """自动生成数据包的 README.md 说明文档。

    包含：
    - 数据集元信息摘要
    - 完整文件结构树
    - NPZ 张量形状和通道/标签顺序说明
    - 三组 split 统计表
    - scaler 拟合范围声明
    - 声学特征和标签的边界说明（防止误用）
    """
    split_table = _readme_split_table(split_distribution)
    text = f"""# V2 时间序列数据包

这是面向深度时序模型训练的结构级仿真数据包，用于模拟声学、NDIR 光学、TCS 热导和环境通道在不同配气与工况下的响应、恢复、漂移和采集噪声。

当前数据不是高保真真实测量时序。

```text
dataset_version: {dataset_version}
calibration_status: pending
simulation_level: structure_level_dynamic_simulation
sequences: {sequence_count}
timesteps: {timesteps}
channels: {len(SEQUENCE_CHANNELS)}
labels: {' / '.join(label_fields)}
```

## 文件结构

```text
{dataset_dir_name}/
  README.md
  sequence_index.csv
  condition_grid_sequence.csv
  sequences/
    modal_sequence_long.csv
    acoustic_derived_sequence_long.csv
    modal_sequence.npz
  labels/
    sequence_labels.csv
  splits/
    train_sequence_ids.csv
    val_sequence_ids.csv
    test_sequence_ids.csv
  scalers/
    scaler_sequence.json
    scaler_sequence_modal.json
  quality/
    sequence_quality_summary.json
```

## 模型输入

`sequences/modal_sequence.npz` 是推荐训练入口：

```text
X: float32, shape = [{sequence_count}, {timesteps}, {len(SEQUENCE_CHANNELS)}]
y: float32, shape = [{sequence_count}, {len(label_fields)}]
sequence_ids: shape = [{sequence_count}]
channel_names: shape = [{len(SEQUENCE_CHANNELS)}]
label_names: shape = [{len(label_fields)}]
```

通道顺序：

```text
V_NDIR_CH4, V_NDIR_CO2, V_TCS, T_C, P_MPa, H_RH,
L_m, piston_position_m, TOF, Amp, f_peak, A_fft_max
```

## 划分与 scaler

当前 split 按 `mixture_id` 分组，训练、验证、测试之间不共享 `mixture_id`。

{split_table}

`scalers/scaler_sequence.json` 和 `scalers/scaler_sequence_modal.json` 只基于 train split 拟合。原始 `modal_sequence.npz` 保留未归一化数值。

`sequences/acoustic_derived_sequence_long.csv` 记录基于同一序列 baseline 段幅值派生的声衰减特征：

```text
Amp_ref_baseline, Amp_ref_calibrated, Amp_ref_fit, attenuation_alpha_n2_baseline,
attenuation_alpha_rel, attenuation_alpha_rel_clip, attenuation_alpha_calibrated,
attenuation_alpha_calibrated_clip, attenuation_alpha_fit, fit_r2,
fit_rmse_log_amp, fit_num_paths, fit_L_m_range, fit_status
```

其中 `Amp_ref_baseline` 是同一序列 baseline 段的观测幅值均值。默认 `Amp_ref_calibrated = Amp_ref_baseline * exp(attenuation_alpha_n2_baseline * L_m)`，用 N2 baseline 的预期衰减回推链路参考幅值。`multi_path_phase` 启用且拟合成功（`fit_status` 为 `ok` 或 `low_confidence`）时，`Amp_ref_calibrated` 会优先使用 `Amp_ref_fit`（多 L_m 子段拟合的截距 `exp(b)`）。

`multi_path_phase` 取值含义：

- `off`：不启用多声程扫描；
- `baseline`：在 baseline 段（纯 N2）扫描 5 个 L_m，`attenuation_alpha_fit` 拟合 N2 链路衰减，仅作链路状态校准用；
- `steady`：在 steady 段（目标混合气）扫描 5 个 L_m，`attenuation_alpha_fit` 拟合混合气真实衰减（推荐用于模型特征）。

`fit_status` 三态：`ok`（fit_r2 ≥ 0.7）、`low_confidence`（fit_r2 < 0.7）、`insufficient_*`（数据不足）。下游模型使用 `attenuation_alpha_fit` 作为输入时应按 `fit_status` 过滤。

`attenuation_alpha_calibrated = -ln(Amp / Amp_ref_calibrated) / L_m` 是校准参考后的衰减估计。该侧车文件不改变默认 12 通道 `modal_sequence.npz`。

## 边界说明

- `TOF / Amp / f_peak / A_fft_max` 是高频声学 DAQ 窗口提取后的低频特征序列，不是原始超声波形。
- `{' / '.join(label_fields)}` 表示整段序列的目标配气条件，不表示每个时间步的气室瞬时组分。
- 响应时间常数、漂移幅度、噪声分布和气路滞后尚未真实标定。
- `attenuation_alpha_calibrated` 当前依赖仿真里的 N2 衰减模型，仍不是实测硬件标定结果；真实设备应使用 N2 或标准气多次采样得到参考幅值。
- 默认 split 是固定随机种子的分组随机划分，不声明已经分层均衡。
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _readme_split_table(split_distribution):
    """生成 README 中的 Markdown split 统计表。"""
    rows = [
        "| split | sequence_count | mixture_count |",
        "| --- | ---: | ---: |",
    ]
    for split_name in ("train", "val", "test"):
        stats = split_distribution[split_name]
        rows.append(f"| {split_name} | {stats['sequence_count']} | {stats['mixture_count']} |")
    return "\n".join(rows)
