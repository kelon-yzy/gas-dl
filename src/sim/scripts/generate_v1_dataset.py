# V1 专利约束特征级仿真数据生成器
# 检测 H2/CH4/CO2 三组分，N2 仅作背景气。四阶段实验设计，产出分层 CSV + 模态训练表 + scaler + 质量报告。
# 详细架构说明见 CLAUDE.md。用法: python scripts/generate_v1_dataset.py --output-dir output --sample-count 2400

import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np

# ── 全局配置常量 ──

SAMPLE_LIMIT = 3000
DEFAULT_SAMPLE_COUNT = 2400
DEFAULT_METHANE_RICH_PATCH_COUNT = 400
DEFAULT_SEED = 20260425

# 压力工作范围 0.1–0.709 MPa (1–7 atm)
PRESSURE_MIN_MPA = 0.1
PRESSURE_MAX_MPA = 0.709
BASELINE_PRESSURE_MAX_MPA = 0.3

# NDIR 吸光度饱和阈值
OPTICAL_SATURATION_ABSORBANCE = 4.0

# distance_stage: 5 段预设声程用于 L-TOF 线性回归标定系统延迟
DISTANCE_STAGE_GROUP_SIZE = 5
DISTANCE_STAGE_PATHS_M = (0.2, 0.6, 1.0, 1.4, 1.8)

# 与 V2/V3 四组分任务对齐：测量样本中的 N2 作为目标组分覆盖 0-20%，
# CH4 保持主体组分，baseline/purge 控制样本仍使用高 N2。
MEASUREMENT_N2_RANGE = (0.0, 20.0)
MEASUREMENT_CH4_MIN = 40.0

# 四阶段样本比例: baseline 10%, distance 40%, pressure 40%, purge 10%
STAGE_WEIGHTS = {
    "baseline_stage": 0.10,
    "distance_stage": 0.40,
    "pressure_stage": 0.40,
    "purge_stage": 0.10,
}

# 仿真处理参数 (calibration_status: pending，所有参数待物理标定)
PROCESSING_PARAMS = {
    # 超声链路
    "t_delay_s": 0.00008,
    "ultrasonic_t_delay_s": 0.00008,
    "ultrasonic_timing_noise_s": 0.000003,  # 3μs 计时抖动

    # 光纤麦克风链路
    "amp_reference": 1.0,
    "fiber_mic_gain": 1.0,
    "fiber_mic_baseline": 0.0,
    "fiber_mic_noise_std": 0.006,
    "acoustic_excitation_frequency_hz": 40000.0,
    "daq_fft_noise_hz": 65.0,

    # NDIR 光学链路: CH4 3.31μm, CO2 4.26μm, 短光程避免饱和
    "optical_baseline_ch4_init": 2.5,
    "optical_baseline_co2_init": 2.5,
    "optical_path_ch4_cm": 0.65,
    "optical_path_co2_cm": 0.26,
    "optical_saturation_absorbance": OPTICAL_SATURATION_ABSORBANCE,

    # TCS 热导链路
    "thermal_baseline_init": 1.1,
    "tcs_response_slope": 15.0,
    "tcs_lambda_offset": 0.026,
    "tcs_temperature_response": 0.004,

    "calibration_status": "pending",
}

# 各输出表字段白名单，严格对齐专利硬件采集链路
DATASET_FIELDS = {
    "condition_grid": [
        "sample_id", "mixture_id", "stage_id",
        "x_H2", "x_CH4", "x_CO2", "x_N2",
        "T_C", "P_MPa", "H_RH", "L_m", "piston_position_m",
        "pressure_stage", "distance_stage", "repeat_id", "status",
    ],
    "serial_low_freq": [
        "sample_id", "timestamp_s",
        "T_C", "H_RH", "P_MPa",
        "V_NDIR_CH4", "V_NDIR_CO2", "V_TCS",
        "piston_position_m", "L_m",
    ],
    "acoustic_features": [
        "sample_id",
        "TOF", "Amp", "f_peak", "A_fft_max",
        "L_m", "T_C", "P_MPa", "H_RH",
        "sound_speed", "attenuation_alpha",
    ],
    "optical_features": [
        "sample_id",
        "V_NDIR_CH4", "V_NDIR_CO2",
        "delta_I_CH4", "delta_I_CO2",
        "A_NDIR_CH4", "A_NDIR_CO2",
        "ndir_ch4_saturated", "ndir_co2_saturated",
        "optical_baseline_drift_ch4", "optical_baseline_drift_co2",
        "T_C", "P_MPa", "H_RH",
    ],
    "thermal_features": [
        "sample_id",
        "V_TCS", "lambda_mix_calibrated", "thermal_baseline_drift",
        "T_C", "P_MPa", "H_RH",
    ],
    "env_features": [
        "sample_id",
        "T_C", "P_MPa", "H_RH", "L_m", "piston_position_m",
        "pressure_stage", "distance_stage",
    ],
    "labels": ["sample_id", "x_H2", "x_CH4", "x_CO2"],
    "feature_table": [
        "sample_id", "stage_id",
        "TOF", "Amp", "f_peak", "A_fft_max", "L_m",
        "sound_speed", "attenuation_alpha",
        "V_NDIR_CH4", "V_NDIR_CO2",
        "delta_I_CH4", "delta_I_CO2",
        "A_NDIR_CH4", "A_NDIR_CO2",
        "ndir_ch4_saturated", "ndir_co2_saturated",
        "V_TCS", "lambda_mix_calibrated",
        "optical_baseline_drift_ch4", "optical_baseline_drift_co2",
        "thermal_baseline_drift",
        "T_C", "P_MPa", "H_RH", "piston_position_m",
        "pressure_stage", "distance_stage",
        "x_H2", "x_CH4", "x_CO2", "x_N2",
        "calibration_status",
    ],
    # 三模态独立训练表: 各自的主特征 + 环境补偿变量，不含饱和标记(防泄漏)
    "train_acoustic": [
        "sample_id",
        "TOF", "Amp", "f_peak", "A_fft_max",
        "L_m", "T_C", "P_MPa", "H_RH",
    ],
    "train_optical": [
        "sample_id",
        "V_NDIR_CH4", "V_NDIR_CO2",
        "delta_I_CH4", "delta_I_CO2",
        "T_C", "P_MPa", "H_RH",
    ],
    "train_thermal": [
        "sample_id",
        "V_TCS", "T_C", "P_MPa", "H_RH",
    ],
}

FEATURE_TABLE_OUTPUTS = (
    ("serial_low_freq", "serial_low_freq"),
    ("daq_acoustic_features", "acoustic_features"),
    ("acoustic_features", "acoustic_features"),
    ("optical_features", "optical_features"),
    ("thermal_features", "thermal_features"),
    ("env_features", "env_features"),
    ("feature_table", "feature_table"),
    ("labels", "labels"),
)

TRAINING_MODALITIES = ("acoustic", "optical", "thermal")

# ═══════════════════════════════════════════════════════════════════════════
# 公开 API
# ═══════════════════════════════════════════════════════════════════════════

def generate_dataset(output_dir, sample_count=DEFAULT_SAMPLE_COUNT, seed=DEFAULT_SEED):
    """生成 V1 多模态特征级仿真数据集。

    管道: 校验 → 条件网格 → 传感器仿真+派生特征 → 分层 CSV + 模态训练表 + scaler + 质量报告
    """
    rng = random.Random(seed)
    _validate_sample_count(sample_count)
    conditions = _condition_rows(sample_count, rng)
    return _generate_dataset_artifacts(Path(output_dir), conditions, rng)


def generate_methane_rich_patch_dataset(
    output_dir, sample_count=DEFAULT_METHANE_RICH_PATCH_COUNT,
    seed=DEFAULT_SEED, sample_id_offset=DEFAULT_SAMPLE_COUNT,
):
    """生成甲烷富集补丁(CH4>95%)。sample_id 从 offset 开始，可拼接到 V1 基础数据集。

    配气: CH4 95-98%(40%) + 98-100%(60%)，剩余由 H2+CO2 随机分拆，N2=0。
    阶段: 约 20% distance_stage(5 段变距) + 80% pressure_stage。
    """
    rng = random.Random(seed)
    _validate_sample_count(sample_count)
    if sample_id_offset < 0:
        raise ValueError("sample_id_offset must be non-negative")
    conditions = _methane_rich_patch_condition_rows(sample_count, rng, sample_id_offset=sample_id_offset)
    return _generate_dataset_artifacts(Path(output_dir), conditions, rng)


# ═══════════════════════════════════════════════════════════════════════════
# 数据产物生成管线
# ═══════════════════════════════════════════════════════════════════════════

def _generate_dataset_artifacts(output_dir, conditions, rng):
    """conditions → _feature_row() → CSV + 模态训练表 + scaler + 质量报告。

    所有行写入 condition_grid 和特征层，训练表只含 synthetic_measurement 行。
    """
    _ensure_dirs(output_dir)
    feature_rows = [_feature_row(row, rng, index) for index, row in enumerate(conditions)]
    paths = _build_output_paths(output_dir)

    _write_csv(paths["condition_grid"], DATASET_FIELDS["condition_grid"], conditions)
    for path_name, field_name in FEATURE_TABLE_OUTPUTS:
        _write_picked_csv(paths[path_name], DATASET_FIELDS[field_name], feature_rows)

    train_rows = [row for row in feature_rows if row["status"] == "synthetic_measurement"]
    _write_training_tables(paths, train_rows)

    _write_json(paths["acoustic_delay_calibration"],
                acoustic_delay_calibration_summary(feature_rows, PROCESSING_PARAMS))
    return paths


def _build_output_paths(output_dir):
    """构造输出文件路径映射。"""
    return {
        "condition_grid": output_dir / "condition_grid_v1.csv",
        "serial_low_freq": output_dir / "raw" / "serial_low_freq.csv",
        "daq_acoustic_features": output_dir / "raw" / "daq_acoustic_features.csv",
        "acoustic_features": output_dir / "features" / "acoustic_features.csv",
        "optical_features": output_dir / "features" / "optical_features.csv",
        "thermal_features": output_dir / "features" / "thermal_features.csv",
        "env_features": output_dir / "features" / "env_features.csv",
        "feature_table": output_dir / "features" / "feature_table.csv",
        "labels": output_dir / "labels" / "labels.csv",
        "train_acoustic": output_dir / "training" / "train_acoustic.csv",
        "train_optical": output_dir / "training" / "train_optical.csv",
        "train_thermal": output_dir / "training" / "train_thermal.csv",
        "train_acoustic_scaled": output_dir / "training" / "train_acoustic_scaled.csv",
        "train_optical_scaled": output_dir / "training" / "train_optical_scaled.csv",
        "train_thermal_scaled": output_dir / "training" / "train_thermal_scaled.csv",
        "scaler_acoustic": output_dir / "training" / "scaler_acoustic.json",
        "scaler_optical": output_dir / "training" / "scaler_optical.json",
        "scaler_thermal": output_dir / "training" / "scaler_thermal.json",
        "acoustic_delay_calibration": output_dir / "quality" / "acoustic_delay_calibration_summary.json",
    }


def _validate_sample_count(sample_count):
    if sample_count > SAMPLE_LIMIT:
        raise ValueError(f"sample_count must be <= {SAMPLE_LIMIT}")
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")


# ═══════════════════════════════════════════════════════════════════════════
# 条件网格采样
# ═══════════════════════════════════════════════════════════════════════════

def _condition_rows(sample_count, rng):
    """生成条件网格。distance_stage 采用成组变距策略(每 5 个样本同组分同工况、仅 L_m 变化)。"""
    rows = []
    distance_group_base = None
    distance_group_index = 0
    distance_group_step = 0

    for index in range(sample_count):
        stage_id = _stage_for_index(index, sample_count)

        if stage_id == "distance_stage":
            if distance_group_base is None or distance_group_step >= DISTANCE_STAGE_GROUP_SIZE:
                distance_group_base = _sample_condition_base(stage_id, rng)
                distance_group_index += 1
                distance_group_step = 0

            l_m = DISTANCE_STAGE_PATHS_M[distance_group_step % len(DISTANCE_STAGE_PATHS_M)]
            rows.append(_build_condition_row(
                index=index, stage_id=stage_id, base=distance_group_base, l_m=l_m,
                mixture_id=f"MD{distance_group_index:05d}", repeat_id=distance_group_step + 1,
            ))
            distance_group_step += 1
        else:
            rows.append(_condition_row(index, rng, sample_count))

    return rows


def _condition_row(index, rng, sample_count):
    stage_id = _stage_for_index(index, sample_count)
    base = _sample_condition_base(stage_id, rng)
    l_m = _path_length_for_stage(stage_id, rng)
    return _build_condition_row(
        index=index, stage_id=stage_id, base=base, l_m=l_m,
        mixture_id=f"M{index + 1:05d}", repeat_id=(index % 3) + 1,
    )


def _methane_rich_patch_condition_rows(sample_count, rng, sample_id_offset=0):
    """甲烷富集补丁条件行: 约 20% distance_stage + 80% pressure_stage。"""
    bin_counts = _methane_rich_patch_bin_counts(sample_count)
    distance_rows = _methane_rich_patch_distance_rows(sample_count)

    distance_bin_counts = {
        "95_98": _round_down_to_group_size(distance_rows * bin_counts["95_98"] / sample_count),
    }
    distance_bin_counts["98_100"] = distance_rows - distance_bin_counts["95_98"]

    pressure_bin_counts = {
        "95_98": bin_counts["95_98"] - distance_bin_counts["95_98"],
        "98_100": bin_counts["98_100"] - distance_bin_counts["98_100"],
    }

    rows = []
    index = 0
    distance_group_index = 0

    for band_name, group_rows in (("95_98", distance_bin_counts["95_98"]),
                                   ("98_100", distance_bin_counts["98_100"])):
        ch4_min, ch4_max = _methane_rich_patch_ch4_bounds(band_name)
        for _ in range(group_rows // DISTANCE_STAGE_GROUP_SIZE):
            base = _sample_methane_rich_condition_base("distance_stage", rng, ch4_min, ch4_max)
            distance_group_index += 1
            for repeat_id, l_m in enumerate(DISTANCE_STAGE_PATHS_M, start=1):
                rows.append(_build_condition_row(
                    index=index, stage_id="distance_stage", base=base, l_m=l_m,
                    mixture_id=f"PD{distance_group_index:05d}", repeat_id=repeat_id,
                    sample_id_offset=sample_id_offset,
                ))
                index += 1

    for band_name, row_count in (("95_98", pressure_bin_counts["95_98"]),
                                  ("98_100", pressure_bin_counts["98_100"])):
        ch4_min, ch4_max = _methane_rich_patch_ch4_bounds(band_name)
        for _ in range(row_count):
            base = _sample_methane_rich_condition_base("pressure_stage", rng, ch4_min, ch4_max)
            rows.append(_build_condition_row(
                index=index, stage_id="pressure_stage", base=base,
                l_m=_path_length_for_stage("pressure_stage", rng),
                mixture_id=f"P{sample_id_offset + index + 1:05d}",
                repeat_id=(index % 3) + 1, sample_id_offset=sample_id_offset,
            ))
            index += 1

    return rows


# ═══════════════════════════════════════════════════════════════════════════
# 条件网格原子操作 — 组分/工况概率采样
# ═══════════════════════════════════════════════════════════════════════════

def _sample_condition_base(stage_id, rng):
    """采样配气+工况 (x_H2, x_CH4, x_CO2, x_N2, T, P, H)。

    - calibration_control (baseline/purge): N2 95-100% + 痕量残余
    - synthetic_measurement: V3 同步四组分配气，N2 0-20%，H2 三分段采样(15%低/15%高/70%均匀)
    """
    is_control = stage_id in {"baseline_stage", "purge_stage"}

    if is_control:
        x_n2 = rng.uniform(95.0, 100.0)
        remainder = 100.0 - x_n2
        x_h2 = rng.uniform(0.0, min(1.0, remainder))
        x_co2 = rng.uniform(0.0, max(0.0, remainder - x_h2))
        x_ch4 = 100.0 - x_n2 - x_h2 - x_co2
    else:
        x_h2, x_ch4, x_co2, x_n2 = _sample_v3_synced_measurement_components(rng)

    x_h2, x_ch4, x_co2, x_n2 = _normalize_components(x_h2, x_ch4, x_co2, x_n2)

    return {
        "x_H2": x_h2, "x_CH4": x_ch4, "x_CO2": x_co2, "x_N2": x_n2,
        "T_C": rng.uniform(15.0, 35.0),
        "P_MPa": _pressure_for_stage(stage_id, rng),
        "H_RH": rng.uniform(20.0, 80.0),
    }


def _sample_v3_synced_measurement_components(rng):
    for _ in range(128):
        x_h2 = _sample_hydrogen_percent(rng)
        x_co2 = rng.uniform(0.0, 15.0)
        x_n2 = rng.uniform(*MEASUREMENT_N2_RANGE)
        x_ch4 = 100.0 - x_h2 - x_co2 - x_n2
        if x_ch4 >= MEASUREMENT_CH4_MIN:
            return x_h2, x_ch4, x_co2, x_n2

    x_h2 = _sample_hydrogen_percent(rng)
    x_co2 = rng.uniform(0.0, 15.0)
    max_n2 = min(MEASUREMENT_N2_RANGE[1], 100.0 - x_h2 - x_co2 - MEASUREMENT_CH4_MIN)
    x_n2 = max(MEASUREMENT_N2_RANGE[0], max_n2)
    x_ch4 = 100.0 - x_h2 - x_co2 - x_n2
    return x_h2, x_ch4, x_co2, x_n2


def _sample_methane_rich_condition_base(stage_id, rng, ch4_min, ch4_max):
    """采样高甲烷配气。H2/CO2 分拆: 40%富氢 / 20%等比例 / 40%富CO2。"""
    x_n2 = 0.0
    x_ch4 = rng.uniform(max(ch4_min + 0.001, ch4_min), ch4_max)
    remainder = max(0.0, 100.0 - x_ch4)

    split_marker = rng.random()
    if split_marker < 0.4:
        h2_ratio = rng.uniform(0.7, 1.0)
    elif split_marker < 0.6:
        h2_ratio = rng.uniform(0.35, 0.65)
    else:
        h2_ratio = rng.uniform(0.0, 0.3)

    x_h2 = remainder * h2_ratio
    x_co2 = remainder - x_h2

    x_h2, x_ch4, x_co2, x_n2 = _normalize_components(x_h2, x_ch4, x_co2, x_n2)

    return {
        "x_H2": x_h2, "x_CH4": x_ch4, "x_CO2": x_co2, "x_N2": x_n2,
        "T_C": rng.uniform(15.0, 35.0),
        "P_MPa": _pressure_for_stage(stage_id, rng),
        "H_RH": rng.uniform(20.0, 80.0),
    }


def _build_condition_row(index, stage_id, base, l_m, mixture_id, repeat_id, sample_id_offset=0):
    """组装标准条件行，附加 pressure_band/distance_band 离散标签和 status 字段。"""
    p_mpa = base["P_MPa"]
    pressure_band = "low" if p_mpa < 0.24 else "mid" if p_mpa < 0.47 else "high"
    distance_band = "short" if l_m < 0.7 else "mid" if l_m < 1.3 else "long"
    is_control = stage_id in {"baseline_stage", "purge_stage"}

    return {
        "sample_id": f"S{index + 1 + sample_id_offset:05d}",
        "mixture_id": mixture_id,
        "stage_id": stage_id,
        "x_H2": _fmt(base["x_H2"], 6),
        "x_CH4": _fmt(base["x_CH4"], 6),
        "x_CO2": _fmt(base["x_CO2"], 6),
        "x_N2": _fmt(base["x_N2"], 6),
        "T_C": _fmt(base["T_C"], 4),
        "P_MPa": _fmt(p_mpa, 4),
        "H_RH": _fmt(base["H_RH"], 4),
        "L_m": _fmt(l_m, 4),
        "piston_position_m": _fmt(l_m, 4),
        "pressure_stage": pressure_band,
        "distance_stage": distance_band,
        "repeat_id": str(repeat_id),
        "status": "calibration_control" if is_control else "synthetic_measurement",
    }


# ═══════════════════════════════════════════════════════════════════════════
# 特征行生成: 条件行 → 传感器仿真 + 派生特征 → 全特征行
# ═══════════════════════════════════════════════════════════════════════════

def _feature_row(condition, rng, index):
    """条件行 → 全特征行(主传感器输出 + 派生物理量 + 质量标记 + 时间戳)。"""
    main = _generate_main_features(condition, rng, PROCESSING_PARAMS)

    derived_input = {**main, "L_m": float(condition["L_m"]), "T_C": float(condition["T_C"])}
    derived = extract_derived_features(derived_input, PROCESSING_PARAMS)

    # 0.5s 间隔 + ±15ms 抖动模拟串口异步采集
    timestamp_s = index * 0.5 + rng.uniform(-0.015, 0.015)

    row = dict(condition)
    row.update({
        "timestamp_s": _fmt(timestamp_s, 4),
        "TOF": _fmt(main["TOF"], 8),
        "Amp": _fmt(main["Amp"], 6),
        "f_peak": _fmt(main["f_peak"], 3),
        "A_fft_max": _fmt(main["A_fft_max"], 4),
        "V_NDIR_CH4": _fmt(main["V_NDIR_CH4"], 6),
        "V_NDIR_CO2": _fmt(main["V_NDIR_CO2"], 6),
        "V_TCS": _fmt(main["V_TCS"], 6),
        "sound_speed": _fmt(derived["sound_speed"], 4),
        "attenuation_alpha": _fmt(derived["attenuation_alpha"], 6),
        "delta_I_CH4": _fmt(derived["delta_I_CH4"], 6),
        "delta_I_CO2": _fmt(derived["delta_I_CO2"], 6),
        "A_NDIR_CH4": _fmt(derived["A_NDIR_CH4"], 6),
        "A_NDIR_CO2": _fmt(derived["A_NDIR_CO2"], 6),
        "ndir_ch4_saturated": str(int(main["ndir_ch4_saturated"])),
        "ndir_co2_saturated": str(int(main["ndir_co2_saturated"])),
        "lambda_mix_calibrated": _fmt(derived["lambda_mix_calibrated"], 6),
        "optical_baseline_drift_ch4": _fmt(main["optical_baseline_drift_ch4_observed"], 6),
        "optical_baseline_drift_co2": _fmt(main["optical_baseline_drift_co2_observed"], 6),
        "thermal_baseline_drift": _fmt(main["thermal_baseline_drift_observed"], 6),
        "calibration_status": PROCESSING_PARAMS["calibration_status"],
    })
    return row


# ═══════════════════════════════════════════════════════════════════════════
# 物性中间量 — _hidden_ 前缀函数为无噪声物理真值，传感器仿真在此基础上叠加噪声
# ═══════════════════════════════════════════════════════════════════════════

def _hidden_sound_speed(x_h2, x_co2, x_n2, t_c, p_mpa):
    """混合气体声速经验模型 (m/s)。基准 420 m/s (CH4 室温常压)。
    H2 提高声速(~1300 m/s)，CO2 降低声速(~270 m/s)。占位公式，未验证 NIST 数据。"""
    return (
        420.0 + 6.0 * x_h2 - 1.4 * x_co2 + 0.35 * x_n2
        + 0.65 * (t_c - 20.0) + 10.0 * (p_mpa - 0.1)
    )


def _hidden_attenuation(x_h2, x_co2, h_rh, p_mpa):
    """超声衰减系数经验模型 (m⁻¹)。CO2 分子弛豫效应为主导衰减源。"""
    return 0.015 + 0.0012 * x_h2 + 0.0035 * x_co2 + 0.0003 * h_rh + 0.006 * p_mpa


# ── NDIR LUT 三线性插值器 ──
# LUT 由 radis_ndir_lut.py 基于 HAPI/HITRAN 预计算，不可用时回退线性经验公式

_LUT_DIR = Path(__file__).parent.parent / "lut"


class NdirLut:
    """(x, T_K, P_atm) → A_band 三线性插值器，封装 scipy RegularGridInterpolator。"""

    def __init__(self, path: Path):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        grid = data["grid"]
        self.optical_path_cm = data.get("optical_path_cm", 5.0)

        self.xs = np.array(sorted(set(grid["mole_fractions"])))
        self.ts = np.array(sorted(set(grid["temperatures_K"])))
        self.ps = np.array(sorted(set(grid["pressures_atm"])))

        shape = (len(self.xs), len(self.ts), len(self.ps))
        self.values = np.zeros(shape)
        x_idx = {v: i for i, v in enumerate(self.xs)}
        t_idx = {v: i for i, v in enumerate(self.ts)}
        p_idx = {v: i for i, v in enumerate(self.ps)}
        for entry in data["entries"]:
            self.values[x_idx[entry["x"]], t_idx[entry["T_K"]], p_idx[entry["P_atm"]]] = entry["A_band"]

    def __call__(self, x: float, T_K: float, P_atm: float) -> float:
        from scipy.interpolate import RegularGridInterpolator

        if not hasattr(self, "_interp"):
            self._interp = RegularGridInterpolator(
                (self.xs, self.ts, self.ps), self.values,
                method="linear", bounds_error=False, fill_value=None,
            )
        return float(self._interp(np.array([[x, T_K, P_atm]]))[0])


def _load_lut(channel: str):
    path = _LUT_DIR / f"ndir_lut_{channel.lower()}.json"
    return NdirLut(path) if path.exists() else None


_LUT_CH4 = _load_lut("CH4")
_LUT_CO2 = _load_lut("CO2")


def _hidden_absorption_ch4(x_ch4, h_rh, p_mpa, t_c=25.0):
    """CH4 NDIR 真值吸光度 (3.31μm 波段)。优先用 HITRAN LUT，不可用时回退 0.008*x_CH4。
    交叉敏感度: A_interference = 0.0008*H_RH + 0.015*P_MPa。"""
    if _LUT_CH4 is not None:
        T_K = t_c + 273.15
        P_atm = p_mpa / 0.101325
        A_lut = _LUT_CH4(x_ch4 / 100.0, T_K, P_atm)
        A_gas = A_lut * (PROCESSING_PARAMS["optical_path_ch4_cm"] / _LUT_CH4.optical_path_cm)
    else:
        A_gas = 0.008 * x_ch4
    return max(0.0, A_gas + 0.0008 * h_rh + 0.015 * p_mpa)


def _hidden_absorption_co2(x_co2, h_rh, p_mpa, t_c=25.0):
    """CO2 NDIR 真值吸光度 (4.26μm 波段)。CO2 吸收截面远大于 CH4，光程更短(0.20cm)。"""
    if _LUT_CO2 is not None:
        T_K = t_c + 273.15
        P_atm = p_mpa / 0.101325
        A_lut = _LUT_CO2(x_co2 / 100.0, T_K, P_atm)
        A_gas = A_lut * (PROCESSING_PARAMS["optical_path_co2_cm"] / _LUT_CO2.optical_path_cm)
    else:
        A_gas = 0.055 * x_co2
    return max(0.0, A_gas + 0.0008 * h_rh + 0.015 * p_mpa)


def _hidden_lambda_mix(x_h2, x_co2, t_c):
    """混合气体热导率经验模型 (W·m⁻¹·K⁻¹)。基准 0.026 (CH4 室温)，H2 热导率极高(~0.18)。"""
    return 0.026 + 0.0012 * x_h2 - 0.00018 * x_co2 + 0.00004 * (t_c - 20.0)


# ═══════════════════════════════════════════════════════════════════════════
# 传感器仿真链路: 物性真值 + 噪声模型 → 传感器观测值
# ═══════════════════════════════════════════════════════════════════════════

def _simulate_ultrasonic_tof(condition, rng, params):
    """超声 TOF = L/c + t_delay + timing_noise。c 下限 250 m/s。"""
    x_h2, x_co2, x_n2 = float(condition["x_H2"]), float(condition["x_CO2"]), float(condition["x_N2"])
    t_c, p_mpa, l_m = float(condition["T_C"]), float(condition["P_MPa"]), float(condition["L_m"])

    sound_speed_true = max(
        250.0,
        _hidden_sound_speed(x_h2, x_co2, x_n2, t_c, p_mpa) + rng.gauss(0.0, 1.5),
    )
    tof = l_m / sound_speed_true + params["ultrasonic_t_delay_s"] + rng.gauss(0.0, params["ultrasonic_timing_noise_s"])
    return {"TOF": tof, "sound_speed_true": sound_speed_true, "t_delay_true_s": params["ultrasonic_t_delay_s"]}


def _simulate_fiber_acoustic_features(condition, rng, params):
    """光纤麦克风声学特征: 声压按 e^(-αL) 衰减 → 光电转换 → FFT 提取 Amp/f_peak/A_fft_max。"""
    x_h2, x_co2 = float(condition["x_H2"]), float(condition["x_CO2"])
    t_c, p_mpa, h_rh, l_m = float(condition["T_C"]), float(condition["P_MPa"]), float(condition["H_RH"]), float(condition["L_m"])

    alpha_true = _hidden_attenuation(x_h2, x_co2, h_rh, p_mpa)
    excitation_stability = 1.0 + rng.gauss(0.0, params["fiber_mic_noise_std"])
    pressure_at_probe = params["amp_reference"] * math.exp(-alpha_true * l_m) * excitation_stability

    fiber_mic_voltage = (
        params["fiber_mic_baseline"]
        + params["fiber_mic_gain"] * pressure_at_probe
        + rng.gauss(0.0, params["fiber_mic_noise_std"] * 0.2)
    )
    amp = max(0.05, fiber_mic_voltage)

    f_peak = (
        params["acoustic_excitation_frequency_hz"]
        + 23.0 * x_h2 - 18.0 * x_co2 + 2.0 * (t_c - 20.0)
        + rng.gauss(0.0, params["daq_fft_noise_hz"])
    )
    a_fft_max = amp * (900.0 + rng.gauss(0.0, 20.0))

    return {"Amp": amp, "f_peak": f_peak, "A_fft_max": a_fft_max,
            "fiber_pressure_response": pressure_at_probe, "attenuation_alpha_true": alpha_true}


def _generate_main_features(condition, rng, params, acoustic_version="v1", params_v2=None):
    """聚合所有传感器主特征 + 基线漂移 + 饱和标记。

    Args:
        acoustic_version: 'v1' 走旧链路（默认），'v2' 走 acoustic_v2 模块的分项 alpha + 链路分离
        params_v2: 仅 acoustic_version='v2' 时使用，None 时取 PROCESSING_PARAMS_V2

    噪声/漂移模型（光学/热导，与声学版本无关）:
    - 光学基线漂移: Δ = 0.0007*(H-50) + 0.004*(P-1) + N(0,0.004)
    - 热导基线漂移: Δ = 0.002*(T-25) + 0.004*(P-1) + N(0,0.003)
    - NDIR: V = V_baseline * e^(-A_true) + N(0,0.008)
    - TCS: V = baseline + slope*(λ-λ_offset) + T_resp*(T-20) + N(0,0.006)
    """
    x_ch4, x_co2 = float(condition["x_CH4"]), float(condition["x_CO2"])
    t_c, p_mpa, h_rh = float(condition["T_C"]), float(condition["P_MPa"]), float(condition["H_RH"])
    x_h2 = float(condition["x_H2"])

    ultrasonic = _simulate_ultrasonic_tof(condition, rng, params)

    if acoustic_version == "v2":
        # 延迟 import 避免循环引用
        from scripts.acoustic_v2 import (
            PROCESSING_PARAMS_V2,
            _simulate_fiber_acoustic_features_v2,
        )
        if params_v2 is None:
            params_v2 = PROCESSING_PARAMS_V2
        sound_speed_mix = ultrasonic["sound_speed_true"]
        fiber_acoustic = _simulate_fiber_acoustic_features_v2(
            condition, rng, sound_speed_mix=sound_speed_mix, params=params_v2,
        )
    elif acoustic_version == "v1":
        fiber_acoustic = _simulate_fiber_acoustic_features(condition, rng, params)
    else:
        raise ValueError(f"Unknown acoustic_version: {acoustic_version!r}")
    absorption_ch4_true = _hidden_absorption_ch4(x_ch4, h_rh, p_mpa, t_c)
    absorption_co2_true = _hidden_absorption_co2(x_co2, h_rh, p_mpa, t_c)
    lambda_true = _hidden_lambda_mix(x_h2, x_co2, t_c)

    optical_drift_ch4 = 0.0007 * (h_rh - 50.0) + 0.004 * (p_mpa - 1.0) + rng.gauss(0.0, 0.004)
    optical_drift_co2 = 0.0007 * (h_rh - 50.0) + 0.004 * (p_mpa - 1.0) + rng.gauss(0.0, 0.004)
    thermal_drift = 0.002 * (t_c - 25.0) + 0.004 * (p_mpa - 1.0) + rng.gauss(0.0, 0.003)

    optical_baseline_ch4_now = params["optical_baseline_ch4_init"] + optical_drift_ch4 + rng.gauss(0.0, 0.006)
    optical_baseline_co2_now = params["optical_baseline_co2_init"] + optical_drift_co2 + rng.gauss(0.0, 0.006)
    thermal_baseline_now = params["thermal_baseline_init"] + thermal_drift

    v_ndir_ch4 = max(0.1, optical_baseline_ch4_now * math.exp(-absorption_ch4_true) + rng.gauss(0.0, 0.008))
    v_ndir_co2 = max(0.1, optical_baseline_co2_now * math.exp(-absorption_co2_true) + rng.gauss(0.0, 0.008))
    v_tcs = (
        thermal_baseline_now
        + params["tcs_response_slope"] * (lambda_true - params["tcs_lambda_offset"])
        + params["tcs_temperature_response"] * (t_c - 20.0)
        + rng.gauss(0.0, 0.006)
    )

    return {
        "TOF": ultrasonic["TOF"],
        "Amp": fiber_acoustic["Amp"],
        "f_peak": fiber_acoustic["f_peak"],
        "A_fft_max": fiber_acoustic["A_fft_max"],
        "V_NDIR_CH4": v_ndir_ch4,
        "V_NDIR_CO2": v_ndir_co2,
        "V_TCS": v_tcs,
        "ndir_ch4_saturated": absorption_ch4_true > params["optical_saturation_absorbance"],
        "ndir_co2_saturated": absorption_co2_true > params["optical_saturation_absorbance"],
        "optical_baseline_drift_ch4_observed": optical_drift_ch4,
        "optical_baseline_drift_co2_observed": optical_drift_co2,
        "thermal_baseline_drift_observed": thermal_drift,
        # V2 检查字段（仅当 acoustic_version='v2' 时存在；不影响 V1 行为）
        **{k: fiber_acoustic[k] for k in (
            "Amp_emit_ref", "Amp_chain_gain", "Amp_before_noise",
            "source_drift", "receiver_gain_drift", "measurement_noise",
            "attenuation_alpha_true",
            "alpha_classical_v2", "alpha_co2_v2", "alpha_ch4_v2", "alpha_h2_diffusion_v2",
            "alpha_h2o_v2",
            "f_relax_co2_eff", "f_relax_ch4_eff", "f_relax_h2o_eff",
            "h_w_pct_eff", "c_mix_used",
        ) if k in fiber_acoustic},
    }


# ═══════════════════════════════════════════════════════════════════════════
# 派生特征提取 — 从传感器观测值反推物理量(仅用可达参数，不使用 _hidden_ 函数)
# ═══════════════════════════════════════════════════════════════════════════

def extract_derived_features(main, params):
    """从主特征提取派生物理量。

    - sound_speed = L / (TOF - t_delay)
    - attenuation_alpha = -ln(Amp/A_ref) / L
    - delta_I = V_baseline_init - V
    - A_NDIR = -ln(V / V_baseline_init)  (含基线漂移混合效应，无法单独拆出)
    - lambda_mix_calibrated = λ_offset + (V_TCS_corrected) / slope
    """
    L = float(main["L_m"])
    sound_speed = L / max(float(main["TOF"]) - params["t_delay_s"], 1e-9)
    attenuation_alpha = -math.log(max(float(main["Amp"]) / params["amp_reference"], 1e-12)) / max(L, 1e-9)

    V_NDIR_CH4, V_NDIR_CO2 = float(main["V_NDIR_CH4"]), float(main["V_NDIR_CO2"])
    delta_I_CH4 = params["optical_baseline_ch4_init"] - V_NDIR_CH4
    delta_I_CO2 = params["optical_baseline_co2_init"] - V_NDIR_CO2
    A_NDIR_CH4 = max(0.0, -math.log(max(V_NDIR_CH4, 1e-9) / params["optical_baseline_ch4_init"]))
    A_NDIR_CO2 = max(0.0, -math.log(max(V_NDIR_CO2, 1e-9) / params["optical_baseline_co2_init"]))

    v_tcs_corrected = float(main["V_TCS"]) - params["thermal_baseline_init"] - params["tcs_temperature_response"] * (float(main["T_C"]) - 20.0)
    lambda_mix_calibrated = params["tcs_lambda_offset"] + v_tcs_corrected / params["tcs_response_slope"]

    return {
        "sound_speed": sound_speed,
        "attenuation_alpha": attenuation_alpha,
        "delta_I_CH4": delta_I_CH4,
        "delta_I_CO2": delta_I_CO2,
        "A_NDIR_CH4": A_NDIR_CH4,
        "A_NDIR_CO2": A_NDIR_CO2,
        "lambda_mix_calibrated": lambda_mix_calibrated,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 声学延迟标定: distance_stage 成组 TOF vs L 线性回归 → 估计系统延迟 t_delay
# ═══════════════════════════════════════════════════════════════════════════

def acoustic_delay_calibration_summary(rows, params):
    """按 mixture_id 分组做 TOF=slope*L+intercept 线性回归。截距=系统延迟，斜率=1/声速。"""
    distance_rows = [row for row in rows if row["stage_id"] == "distance_stage"]
    if len(distance_rows) < 2:
        return {"method": "grouped_linear_fit_on_distance_stage",
                "n_samples": len(distance_rows), "status": "insufficient_distance_stage_rows"}

    grouped_rows = {}
    for row in distance_rows:
        grouped_rows.setdefault(row.get("mixture_id", "__ungrouped__"), []).append(row)

    fits = []
    for mixture_id, group_rows in grouped_rows.items():
        fit = _fit_tof_delay_group(group_rows)
        if fit is not None:
            fit["mixture_id"] = mixture_id
            fits.append(fit)

    if not fits:
        return {"method": "grouped_linear_fit_on_distance_stage",
                "n_samples": len(distance_rows), "n_groups": len(grouped_rows),
                "status": "insufficient_grouped_path_length_variation"}

    estimated_delays = [f["estimated_t_delay_s"] for f in fits]
    estimated_speeds = [f["estimated_sound_speed_m_per_s"] for f in fits]
    rms_residuals = [f["rms_tof_residual_s"] for f in fits]

    return {
        "method": "grouped_linear_fit_on_distance_stage",
        "n_samples": len(distance_rows), "n_groups": len(grouped_rows),
        "n_fitted_groups": len(fits),
        "estimated_t_delay_s": _mean(estimated_delays),
        "mean_estimated_t_delay_s": _mean(estimated_delays),
        "std_estimated_t_delay_s": _std(estimated_delays),
        "configured_t_delay_s": params["ultrasonic_t_delay_s"],
        "estimated_sound_speed_m_per_s": _mean(estimated_speeds),
        "mean_estimated_sound_speed_m_per_s": _mean(estimated_speeds),
        "mean_rms_tof_residual_s": _mean(rms_residuals),
        "max_abs_tof_residual_s": max(f["max_abs_tof_residual_s"] for f in fits),
        "status": "grouped_distance_stage_calibration",
    }


def _fit_tof_delay_group(rows):
    """对同组数据做 TOF vs L 最小二乘线性拟合。需 ≥2 个不同 L 值。"""
    if len(rows) < 2:
        return None

    lengths = [float(row["L_m"]) for row in rows]
    tofs = [float(row["TOF"]) for row in rows]
    n = len(rows)

    mean_l = sum(lengths) / n
    mean_tof = sum(tofs) / n

    denominator = sum((l - mean_l) ** 2 for l in lengths)
    if denominator <= 1e-15:
        return None

    slope = sum((l - mean_l) * (t - mean_tof) for l, t in zip(lengths, tofs)) / denominator
    intercept = mean_tof - slope * mean_l
    fitted_sound_speed = 1.0 / slope if slope > 0 else 0.0

    residuals = [t - (intercept + slope * l) for l, t in zip(lengths, tofs)]
    rms_residual = math.sqrt(sum(r ** 2 for r in residuals) / n)

    return {
        "n_samples": n,
        "estimated_t_delay_s": intercept,
        "estimated_sound_speed_m_per_s": fitted_sound_speed,
        "rms_tof_residual_s": rms_residual,
        "max_abs_tof_residual_s": max(abs(r) for r in residuals),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════════════════

def _mean(values):
    return sum(values) / len(values)


def _std(values):
    mean = _mean(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


def _methane_rich_patch_bin_counts(sample_count):
    first_band = int(round(sample_count * 0.4))
    return {"95_98": first_band, "98_100": sample_count - first_band}


def _methane_rich_patch_distance_rows(sample_count):
    distance_rows = _round_down_to_group_size(sample_count * 0.2)
    if distance_rows == 0 and sample_count >= DISTANCE_STAGE_GROUP_SIZE:
        return DISTANCE_STAGE_GROUP_SIZE
    return min(distance_rows, sample_count)


def _methane_rich_patch_ch4_bounds(band_name):
    return (95.0, 98.0) if band_name == "95_98" else (98.0, 100.0)


def _round_down_to_group_size(value):
    return int(value) // DISTANCE_STAGE_GROUP_SIZE * DISTANCE_STAGE_GROUP_SIZE


def _normalize_components(x_h2, x_ch4, x_co2, x_n2):
    total = x_h2 + x_ch4 + x_co2 + x_n2
    x_h2, x_ch4, x_co2, x_n2 = [v * 100.0 / total for v in (x_h2, x_ch4, x_co2, x_n2)]
    x_h2 = round(x_h2, 6)
    x_co2 = round(x_co2, 6)
    x_n2 = round(x_n2, 6)
    x_ch4 = round(100.0 - x_h2 - x_co2 - x_n2, 6)
    return x_h2, x_ch4, x_co2, x_n2


# ═══════════════════════════════════════════════════════════════════════════
# z-score 标准化器
# ═══════════════════════════════════════════════════════════════════════════

def fit_modal_scaler(rows, fields):
    """拟合 z-score 参数 (mean, std)。方差为 0 的字段 std=1.0。"""
    if not rows:
        raise ValueError("Cannot fit scaler on empty rows.")
    n = len(rows)
    means, stds = {}, {}
    for field in fields:
        values = [float(row[field]) for row in rows]
        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / n
        stds[field] = math.sqrt(variance) if variance > 1e-15 else 1.0
        means[field] = mean
    return {"method": "zscore", "n_samples": n, "fields": list(fields), "mean": means, "std": stds}


def apply_modal_scaler(rows, scaler, format_digits=6):
    """z = (x - mean) / std。"""
    out_rows = []
    for row in rows:
        new_row = dict(row)
        for field in scaler["fields"]:
            new_row[field] = _fmt((float(row[field]) - scaler["mean"][field]) / scaler["std"][field], format_digits)
        out_rows.append(new_row)
    return out_rows


def invert_modal_scaler(rows, scaler, format_digits=6):
    """x = z * std + mean。"""
    out_rows = []
    for row in rows:
        new_row = dict(row)
        for field in scaler["fields"]:
            new_row[field] = _fmt(float(row[field]) * scaler["std"][field] + scaler["mean"][field], format_digits)
        out_rows.append(new_row)
    return out_rows


# ═══════════════════════════════════════════════════════════════════════════
# 阶段与工况分配
# ═══════════════════════════════════════════════════════════════════════════

def _stage_for_index(index, sample_count):
    ratio = index / sample_count
    if ratio < STAGE_WEIGHTS["baseline_stage"]:
        return "baseline_stage"
    if ratio < STAGE_WEIGHTS["baseline_stage"] + STAGE_WEIGHTS["distance_stage"]:
        return "distance_stage"
    if ratio < STAGE_WEIGHTS["baseline_stage"] + STAGE_WEIGHTS["distance_stage"] + STAGE_WEIGHTS["pressure_stage"]:
        return "pressure_stage"
    return "purge_stage"


def _sample_hydrogen_percent(rng):
    """H2 三分段采样: 15%低(0-3%) / 15%高(25-30%) / 70%均匀(0-30%)。"""
    marker = rng.random()
    if marker < 0.15:
        return rng.uniform(0.0, 3.0)
    if marker > 0.85:
        return rng.uniform(25.0, 30.0)
    return rng.uniform(0.0, 30.0)


def _pressure_for_stage(stage_id, rng):
    if stage_id == "baseline_stage":
        return rng.uniform(PRESSURE_MIN_MPA, BASELINE_PRESSURE_MAX_MPA)
    return rng.uniform(PRESSURE_MIN_MPA, PRESSURE_MAX_MPA)


def _path_length_for_stage(stage_id, rng):
    if stage_id == "distance_stage":
        return rng.uniform(0.2, 1.8)
    return rng.uniform(0.4, 1.2)


# ═══════════════════════════════════════════════════════════════════════════
# 文件 I/O
# ═══════════════════════════════════════════════════════════════════════════

def _ensure_dirs(output_dir):
    for relative in ("raw", "features", "labels", "training", "quality"):
        (output_dir / relative).mkdir(parents=True, exist_ok=True)


def _write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=False)


def _write_picked_csv(path, fields, rows):
    _write_csv(path, fields, [_pick(row, fields) for row in rows])


def _write_training_tables(paths, train_rows):
    for modality in TRAINING_MODALITIES:
        fields = DATASET_FIELDS[f"train_{modality}"]
        scale_fields = [f for f in fields if f != "sample_id"]
        raw_modal_rows = [_pick(row, fields) for row in train_rows]

        scaler = fit_modal_scaler(raw_modal_rows, scale_fields)
        scaled_modal_rows = apply_modal_scaler(raw_modal_rows, scaler)

        _write_csv(paths[f"train_{modality}"], fields, raw_modal_rows)
        _write_csv(paths[f"train_{modality}_scaled"], fields, scaled_modal_rows)
        _write_json(paths[f"scaler_{modality}"], scaler)


def _pick(row, fields):
    return {field: row[field] for field in fields}


def _fmt(value, digits):
    return f"{value:.{digits}f}"


# ═══════════════════════════════════════════════════════════════════════════
# CLI 入口
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Generate V1 patent-constrained feature-level simulation dataset.")
    parser.add_argument("--output-dir", default=".", help="Output directory for generated CSV files.")
    parser.add_argument("--profile", choices=("v1", "methane-rich-patch"), default="v1")
    parser.add_argument("--sample-count", type=int, default=None, help="Number of samples, max 3000.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--sample-id-offset", type=int, default=None,
                        help="Starting offset for sample_id (methane-rich patch profile).")
    args = parser.parse_args()

    if args.profile == "methane-rich-patch":
        sample_count = args.sample_count or DEFAULT_METHANE_RICH_PATCH_COUNT
        sample_id_offset = args.sample_id_offset if args.sample_id_offset is not None else DEFAULT_SAMPLE_COUNT
        paths = generate_methane_rich_patch_dataset(
            Path(args.output_dir), sample_count=sample_count, seed=args.seed, sample_id_offset=sample_id_offset)
    else:
        sample_count = args.sample_count or DEFAULT_SAMPLE_COUNT
        paths = generate_dataset(Path(args.output_dir), sample_count=sample_count, seed=args.seed)

    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
