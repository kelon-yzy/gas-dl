# V2 声学派生侧车 + 声学质量摘要。
#
# 维护两类输出：
#   - acoustic_derived_sequence_long.csv: 每条序列每时间步的派生衰减特征
#   - quality_summary.json::acoustic: 声学诊断指标（amp/alpha 分布、单调性、湿度可分性等）
#
# acoustic_v2 已归档但仍被本模块按 acoustic_version="v2" 入参调用，
# 用于派生 alpha_calibrated/alpha_fit 的校准链路。

import math

from scripts.generate_v1_dataset import _fmt, _hidden_attenuation

from .constants import (
    ALPHA_FIT_R2_LOW_CONFIDENCE_THRESHOLD,
    MULTI_PATH_PHASE_BASELINE,
    MULTI_PATH_PHASE_OFF,
    MULTI_PATH_PHASE_STEADY,
    normalize_multi_path_phase,
)


def acoustic_derived_sequence_rows(long_rows, acoustic_version="v1", multi_path_phase=MULTI_PATH_PHASE_OFF):
    """Derive baseline-referenced acoustic attenuation features from sequence rows.

    Args:
        long_rows: 完整时序长表行
        acoustic_version: 'v1' / 'v2'
        multi_path_phase: 'off' / 'baseline' / 'steady'。决定 alpha_fit 在哪个相位拟合。
    """
    rows_by_sequence = {}
    for row in long_rows:
        rows_by_sequence.setdefault(row["sequence_id"], []).append(row)

    # 拟合相位选择
    if multi_path_phase == MULTI_PATH_PHASE_STEADY:
        fit_phase = "steady"
    else:
        fit_phase = "baseline"

    derived_rows = []
    for sequence_id, rows in rows_by_sequence.items():
        baseline_rows = [row for row in rows if row["phase_id"] == "baseline"]
        reference_rows = baseline_rows if baseline_rows else rows
        amp_ref = sum(float(row["Amp"]) for row in reference_rows) / len(reference_rows)
        amp_ref = max(amp_ref, 1e-12)
        reference_temperature = sum(float(row["T_C"]) for row in reference_rows) / len(reference_rows)
        reference_length = sum(float(row["L_m"]) for row in reference_rows) / len(reference_rows)
        reference_humidity = sum(float(row["H_RH"]) for row in reference_rows) / len(reference_rows)
        reference_pressure = sum(float(row["P_MPa"]) for row in reference_rows) / len(reference_rows)
        if acoustic_version == "v2":
            from scripts.acoustic_v2 import _hidden_attenuation_v2

            alpha_n2 = _hidden_attenuation_v2(
                0.0,
                0.0,
                0.0,
                100.0,
                t_c=reference_temperature,
                p_mpa=reference_pressure,
                h_rh=reference_humidity,
            )["alpha_true_v2"]
        else:
            alpha_n2 = _hidden_attenuation(0.0, 0.0, reference_humidity, reference_pressure)
        fit_stats = fit_log_amp_vs_length(rows, target_phase=fit_phase)
        if fit_stats["fit_status"] in {"ok", "low_confidence"} and fit_stats["Amp_ref_fit"] != "":
            amp_ref_calibrated = max(float(fit_stats["Amp_ref_fit"]), 1e-12)
        else:
            amp_ref_calibrated = amp_ref * math.exp(alpha_n2 * max(reference_length, 1e-9))
            amp_ref_calibrated = max(amp_ref_calibrated, 1e-12)

        for row in rows:
            amp = max(float(row["Amp"]), 1e-12)
            length = max(float(row["L_m"]), 1e-9)
            alpha_rel = -math.log(amp / amp_ref) / length
            alpha_calibrated = -math.log(amp / amp_ref_calibrated) / length
            derived_rows.append(
                {
                    "sequence_id": sequence_id,
                    "timestep": row["timestep"],
                    "timestamp_s": row["timestamp_s"],
                    "phase_id": row["phase_id"],
                    "Amp_ref_baseline": _fmt(amp_ref, 6),
                    "Amp_ref_calibrated": _fmt(amp_ref_calibrated, 6),
                    "Amp_ref_fit": fit_stats["Amp_ref_fit"],
                    "attenuation_alpha_n2_baseline": _fmt(alpha_n2, 8),
                    "attenuation_alpha_rel": _fmt(alpha_rel, 8),
                    "attenuation_alpha_rel_clip": _fmt(max(alpha_rel, 0.0), 8),
                    "attenuation_alpha_calibrated": _fmt(alpha_calibrated, 8),
                    "attenuation_alpha_calibrated_clip": _fmt(max(alpha_calibrated, 0.0), 8),
                    "attenuation_alpha_fit": fit_stats["attenuation_alpha_fit"],
                    "fit_r2": fit_stats["fit_r2"],
                    "fit_rmse_log_amp": fit_stats["fit_rmse_log_amp"],
                    "fit_num_paths": fit_stats["fit_num_paths"],
                    "fit_L_m_range": fit_stats["fit_L_m_range"],
                    "fit_status": fit_stats["fit_status"],
                }
            )
    return derived_rows


def fit_log_amp_vs_length(rows, target_phase="baseline"):
    """对单序列指定相位段做 ln(Amp) vs L_m 线性拟合。

    Args:
        rows: 序列长表行列表
        target_phase: 'baseline' / 'steady'，决定使用哪个相位的数据拟合
    """
    insufficient_status = (
        "insufficient_baseline_rows" if target_phase == "baseline" else "insufficient_steady_rows"
    )
    result = {
        "Amp_ref_fit": "",
        "attenuation_alpha_fit": "",
        "fit_r2": "",
        "fit_rmse_log_amp": "",
        "fit_num_paths": "0",
        "fit_L_m_range": "0.000000",
        "fit_status": insufficient_status,
    }

    target_rows = [row for row in rows if row["phase_id"] == target_phase]
    if len(target_rows) < 3:
        return result

    path_groups = {}
    for row in target_rows:
        key = round(float(row["L_m"]), 1)
        path_groups.setdefault(key, []).append(max(float(row["Amp"]), 1e-12))

    distinct_path_count = len(path_groups)
    if distinct_path_count == 0:
        return result

    l_values = sorted(path_groups.keys())
    amp_values = [sum(path_groups[path]) / len(path_groups[path]) for path in l_values]
    l_range = max(l_values) - min(l_values)

    result["fit_num_paths"] = str(distinct_path_count)
    result["fit_L_m_range"] = _fmt(l_range, 6)

    if distinct_path_count < 3 or l_range < 0.3:
        result["fit_status"] = "insufficient_path_variation"
        return result

    log_amps = [math.log(value) for value in amp_values]
    mean_l = sum(l_values) / len(l_values)
    mean_log_amp = sum(log_amps) / len(log_amps)
    var_l = sum((value - mean_l) ** 2 for value in l_values)
    if var_l <= 1e-12:
        result["fit_status"] = "insufficient_path_variation"
        return result

    cov = sum((l_values[index] - mean_l) * (log_amps[index] - mean_log_amp) for index in range(len(l_values)))
    slope = cov / var_l
    intercept = mean_log_amp - slope * mean_l
    predictions = [intercept + slope * value for value in l_values]
    residuals = [log_amps[index] - predictions[index] for index in range(len(log_amps))]
    ss_res = sum(value * value for value in residuals)
    ss_tot = sum((value - mean_log_amp) ** 2 for value in log_amps)

    fit_r2_value = 1.0 if ss_tot <= 1e-12 else 1.0 - ss_res / ss_tot
    # 低 fit_r2 拟合标记为 low_confidence。下游使用 attenuation_alpha_fit 时应按 fit_status 过滤。
    final_status = "ok" if fit_r2_value >= ALPHA_FIT_R2_LOW_CONFIDENCE_THRESHOLD else "low_confidence"

    result.update(
        {
            "Amp_ref_fit": _fmt(math.exp(intercept), 6),
            "attenuation_alpha_fit": _fmt(-slope, 8),
            "fit_r2": _fmt(fit_r2_value, 6),
            "fit_rmse_log_amp": _fmt(math.sqrt(ss_res / len(log_amps)), 8),
            "fit_status": final_status,
        }
    )
    return result


def sequence_condition_with_sweep(condition, x_co2=None, h_rh=None):
    """按固定组分/湿度扫值，构造用于质量检查的条件。"""
    x_h2 = float(condition["x_H2"])
    x_n2 = float(condition["x_N2"])
    total_fixed = x_h2 + x_n2
    max_co2 = max(0.0, 100.0 - total_fixed)
    co2_value = float(condition["x_CO2"]) if x_co2 is None else float(x_co2)
    co2_value = min(max(co2_value, 0.0), max_co2)

    return {
        "x_H2": x_h2,
        "x_CH4": max(0.0, 100.0 - total_fixed - co2_value),
        "x_CO2": co2_value,
        "x_N2": x_n2,
        "T_C": float(condition["T_C_base"]),
        "P_MPa": float(condition["P_MPa_base"]),
        "H_RH": float(condition["H_RH_base"]) if h_rh is None else float(h_rh),
    }


def sequence_alpha_true(condition, row, acoustic_version):
    """按序列条件和当前环境行重算声衰减真值，用于质量诊断。"""
    if acoustic_version == "v2":
        from scripts.acoustic_v2 import _hidden_attenuation_v2

        return _hidden_attenuation_v2(
            float(condition["x_H2"]),
            float(condition["x_CH4"]),
            float(condition["x_CO2"]),
            float(condition["x_N2"]),
            float(row["T_C"]),
            float(row["P_MPa"]),
            float(row["H_RH"]),
        )["alpha_true_v2"]

    return _hidden_attenuation(
        float(condition["x_H2"]),
        float(condition["x_CO2"]),
        float(row["H_RH"]),
        float(row["P_MPa"]),
    )


def acoustic_quality_summary(conditions, long_rows, acoustic_derived_rows, acoustic_version, multi_path_phase):
    """汇总声学检查项，写入 quality_summary.json。

    Args:
        multi_path_phase: 'off' / 'baseline' / 'steady'。决定 alpha_fit 参考真值与单调性检查的相位。
    """
    multi_path_phase = normalize_multi_path_phase(multi_path_phase)
    is_baseline_scan = multi_path_phase == MULTI_PATH_PHASE_BASELINE
    is_steady_scan = multi_path_phase == MULTI_PATH_PHASE_STEADY
    fit_phase = "steady" if is_steady_scan else "baseline"

    condition_by_sequence = {row["sequence_id"]: row for row in conditions}
    long_rows_by_sequence = {}
    for row in long_rows:
        long_rows_by_sequence.setdefault(row["sequence_id"], []).append(row)

    derived_rows_by_sequence = {}
    for row in acoustic_derived_rows:
        derived_rows_by_sequence.setdefault(row["sequence_id"], []).append(row)

    amp_values = [float(row["Amp"]) for row in long_rows]
    alpha_true_values = []
    alpha_calibrated_values = []
    # alpha_calibrated 按相位分桶的负值计数（用于诊断到底是哪个阶段产生负值）
    alpha_calibrated_negative_by_phase = {
        "baseline": 0, "exposure": 0, "steady": 0, "recovery": 0, "unknown": 0,
    }
    alpha_calibrated_total_by_phase = {
        "baseline": 0, "exposure": 0, "steady": 0, "recovery": 0, "unknown": 0,
    }
    alpha_fit_abs_errors = []
    alpha_fit_rel_errors = []
    alpha_fit_r2_values = []
    fit_available_count = 0
    fit_ok_count = 0
    fit_low_confidence_count = 0
    monotonic_total = 0
    monotonic_pass = 0
    co2_total = 0
    co2_pass = 0
    humidity_total = 0
    humidity_pass = 0
    humidity_delta_values = []
    humidity_noise_floor_values = []
    humidity_delta_over_noise_values = []
    monotonic_tolerance = 0.0025 if acoustic_version == "v2" else 0.0

    for sequence_id, condition in condition_by_sequence.items():
        seq_rows = long_rows_by_sequence.get(sequence_id, [])
        derived_rows = derived_rows_by_sequence.get(sequence_id, [])
        if not seq_rows or not derived_rows:
            continue

        alpha_true_values.extend(sequence_alpha_true(condition, row, acoustic_version) for row in seq_rows)
        for row in derived_rows:
            value_str = row.get("attenuation_alpha_calibrated", "")
            if value_str == "":
                continue
            value = float(value_str)
            alpha_calibrated_values.append(value)
            phase = row.get("phase_id", "unknown")
            if phase not in alpha_calibrated_total_by_phase:
                phase = "unknown"
            alpha_calibrated_total_by_phase[phase] += 1
            if value < 0.0:
                alpha_calibrated_negative_by_phase[phase] += 1

        # ok 与 low_confidence 都视为有拟合结果，分别计数；统计时仍纳入用于了解分布全貌
        fit_row = next(
            (row for row in derived_rows if row.get("fit_status") in {"ok", "low_confidence"}),
            None,
        )
        if fit_row is not None:
            fit_available_count += 1
            if fit_row["fit_status"] == "ok":
                fit_ok_count += 1
            else:
                fit_low_confidence_count += 1
            alpha_fit = float(fit_row["attenuation_alpha_fit"])
            # 拟合参考真值：steady-scan 用混合气真值，baseline-scan 用 N2 真值
            if is_steady_scan:
                # 用 fit_phase 段第一行重算混合气 alpha_true
                phase_rows = [r for r in seq_rows if r["phase_id"] == fit_phase]
                ref_env_row = phase_rows[0] if phase_rows else seq_rows[0]
                alpha_reference = sequence_alpha_true(condition, ref_env_row, acoustic_version)
            else:
                alpha_reference = float(fit_row["attenuation_alpha_n2_baseline"])
            alpha_fit_abs_errors.append(abs(alpha_fit - alpha_reference))
            alpha_fit_rel_errors.append(abs(alpha_fit - alpha_reference) / max(abs(alpha_reference), 1e-12))
            alpha_fit_r2_values.append(float(fit_row["fit_r2"]))

        # 单调性检查相位与拟合相位保持一致
        if is_baseline_scan or is_steady_scan:
            check_phase = "steady" if is_steady_scan else "baseline"
            phase_rows = [row for row in seq_rows if row["phase_id"] == check_phase]
            path_groups = {}
            for row in phase_rows:
                path_groups.setdefault(round(float(row["L_m"]), 1), []).append(float(row["Amp"]))

            if len(path_groups) >= 3:
                monotonic_total += 1
                grouped_means = [sum(values) / len(values) for _, values in sorted(path_groups.items())]
                if all(
                    grouped_means[index] >= grouped_means[index + 1] - monotonic_tolerance
                    for index in range(len(grouped_means) - 1)
                ):
                    monotonic_pass += 1

        reference_row = seq_rows[0]
        co2_max = min(15.0, max(0.0, 100.0 - float(condition["x_H2"]) - float(condition["x_N2"])))
        if co2_max > 0.0:
            low_co2_condition = sequence_condition_with_sweep(condition, x_co2=0.0)
            high_co2_condition = sequence_condition_with_sweep(condition, x_co2=co2_max)
            low_alpha = sequence_alpha_true(low_co2_condition, reference_row, acoustic_version)
            high_alpha = sequence_alpha_true(high_co2_condition, reference_row, acoustic_version)
            co2_total += 1
            if high_alpha > low_alpha:
                co2_pass += 1

        low_humidity_condition = sequence_condition_with_sweep(condition, h_rh=20.0)
        high_humidity_condition = sequence_condition_with_sweep(condition, h_rh=80.0)
        low_humidity_row = {**reference_row, "H_RH": _fmt(20.0, 4)}
        high_humidity_row = {**reference_row, "H_RH": _fmt(80.0, 4)}
        low_alpha = sequence_alpha_true(low_humidity_condition, low_humidity_row, acoustic_version)
        high_alpha = sequence_alpha_true(high_humidity_condition, high_humidity_row, acoustic_version)
        humidity_total += 1
        alpha_delta = abs(high_alpha - low_alpha)
        # 噪声底用 fit_phase 段 alpha_calibrated 的方差。
        # 在 steady-scan 模式下，baseline 段 alpha_calibrated 反映的是 baseline-vs-steady 的
        # 系统偏差（链路参考来自 steady 拟合），不是真正的观测噪声，会高估 noise_floor。
        noise_phase = "steady" if is_steady_scan else "baseline"
        noise_alpha_values = [
            float(row["attenuation_alpha_calibrated"])
            for row in derived_rows
            if row["phase_id"] == noise_phase and row["attenuation_alpha_calibrated"] != ""
        ]
        alpha_noise_floor = 0.0
        if len(noise_alpha_values) >= 2:
            noise_mean = sum(noise_alpha_values) / len(noise_alpha_values)
            variance = sum((value - noise_mean) ** 2 for value in noise_alpha_values) / len(noise_alpha_values)
            alpha_noise_floor = math.sqrt(max(0.0, variance))

        humidity_delta_values.append(alpha_delta)
        humidity_noise_floor_values.append(alpha_noise_floor)
        humidity_delta_over_noise_values.append(alpha_delta / max(alpha_noise_floor, 1e-12))

        if alpha_delta > max(alpha_noise_floor, 0.0001 if acoustic_version == "v2" else 0.001):
            humidity_pass += 1

    def _mean(values):
        return sum(values) / len(values) if values else 0.0

    def _median(values):
        if not values:
            return 0.0
        ordered = sorted(values)
        mid = len(ordered) // 2
        if len(ordered) % 2 == 1:
            return ordered[mid]
        return 0.5 * (ordered[mid - 1] + ordered[mid])

    return {
        "amp_obs_min": min(amp_values) if amp_values else 0.0,
        "amp_obs_max": max(amp_values) if amp_values else 0.0,
        "amp_obs_mean": _mean(amp_values),
        "amp_negative_count": sum(1 for value in amp_values if value < 0.0),
        "alpha_true_min": min(alpha_true_values) if alpha_true_values else 0.0,
        "alpha_true_max": max(alpha_true_values) if alpha_true_values else 0.0,
        "alpha_true_mean": _mean(alpha_true_values),
        "alpha_true_negative_count": sum(1 for value in alpha_true_values if value < 0.0),
        "alpha_calibrated_negative_count": sum(
            1
            for value in alpha_calibrated_values
            if value < 0.0
        ),
        "alpha_calibrated_negative_by_phase": dict(alpha_calibrated_negative_by_phase),
        "alpha_calibrated_total_by_phase": dict(alpha_calibrated_total_by_phase),
        "alpha_fit_available_ratio": fit_available_count / len(condition_by_sequence) if condition_by_sequence else 0.0,
        "alpha_fit_ok_count": fit_ok_count,
        "alpha_fit_low_confidence_count": fit_low_confidence_count,
        "alpha_fit_low_confidence_ratio": (
            fit_low_confidence_count / fit_available_count if fit_available_count else 0.0
        ),
        "alpha_fit_r2_threshold": ALPHA_FIT_R2_LOW_CONFIDENCE_THRESHOLD,
        "alpha_fit_abs_error_mean": _mean(alpha_fit_abs_errors),
        "alpha_fit_rel_error_median": _median(alpha_fit_rel_errors),
        "alpha_fit_r2_mean": _mean(alpha_fit_r2_values),
        "monotonic_L_amp_pass_ratio": monotonic_pass / monotonic_total if monotonic_total else 0.0,
        "co2_alpha_monotonic_pass_ratio": co2_pass / co2_total if co2_total else 0.0,
        "humidity_effect_detectable_ratio": humidity_pass / humidity_total if humidity_total else 0.0,
        "humidity_alpha_delta_mean": _mean(humidity_delta_values),
        "humidity_alpha_noise_floor_mean": _mean(humidity_noise_floor_values),
        "humidity_alpha_delta_over_noise_median": _median(humidity_delta_over_noise_values),
    }
