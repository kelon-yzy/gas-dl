from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.acoustic_fiber_mic_v3 import simulate_fiber_mic_measurement
from scripts.acoustic_waveform_v3 import simulate_waveform_measurement


_FIXED_T_C = 25.0
_FIXED_P_MPA = 0.1
_FIXED_H_RH = 50.0
_FIXED_L_M = 1.0
_FIXED_X_H2 = 5.0
_FIXED_X_CO2 = 5.0
_FIXED_X_CH4 = 85.0
_FIXED_X_N2 = 5.0
_CO2_GRID = np.linspace(0.0, 14.0, 15).tolist()
_H2_GRID = np.linspace(0.0, 25.0, 11).tolist()
_HUMIDITY_GRID = [20.0, 40.0, 60.0, 80.0]
_LENGTH_GRID = [0.2, 0.6, 1.0, 1.4]


def _pearson(values_a: list[float], values_b: list[float]) -> float:
    if len(values_a) < 2 or len(values_b) < 2:
        return 0.0
    corr = np.corrcoef(np.asarray(values_a, dtype=np.float64), np.asarray(values_b, dtype=np.float64))[0, 1]
    if np.isnan(corr):
        return 0.0
    return float(corr)


def _ultrasonic_sample(x_h2: float, x_ch4: float, x_co2: float, x_n2: float, h_rh: float, l_m: float) -> dict:
    return simulate_waveform_measurement(
        x_h2=x_h2,
        x_ch4=x_ch4,
        x_co2=x_co2,
        x_n2=x_n2,
        t_c=_FIXED_T_C,
        p_mpa=_FIXED_P_MPA,
        h_rh=h_rh,
        l_m=l_m,
        seed=20260514,
        noise_std_v=1e-3,
    )


def _fiber_sample(x_h2: float, x_ch4: float, x_co2: float, x_n2: float, h_rh: float, l_m: float) -> dict:
    return simulate_fiber_mic_measurement(
        x_h2=x_h2,
        x_ch4=x_ch4,
        x_co2=x_co2,
        x_n2=x_n2,
        t_c=_FIXED_T_C,
        p_mpa=_FIXED_P_MPA,
        h_rh=h_rh,
        l_m=l_m,
        seed=20260514,
        noise_std_v=1e-3,
    )


def ultrasonic_length_scan() -> dict:
    peak_indices = []
    for l_m in _LENGTH_GRID:
        result = _ultrasonic_sample(_FIXED_X_H2, _FIXED_X_CH4, _FIXED_X_CO2, _FIXED_X_N2, _FIXED_H_RH, l_m)
        peak_indices.append(int(result["peak_index"]))
    diffs = np.diff(peak_indices)
    return {
        "length_grid": _LENGTH_GRID,
        "peak_indices": peak_indices,
        "pearson_r": _pearson(_LENGTH_GRID, peak_indices),
        "monotonic_increasing": bool(np.all(diffs >= 0)),
    }


def ultrasonic_h2_scan() -> dict:
    peak_indices = []
    valid_grid = []
    for x_h2 in _H2_GRID:
        x_n2 = max(0.0, 100.0 - _FIXED_X_CO2 - _FIXED_X_CH4 - x_h2)
        result = _ultrasonic_sample(x_h2, _FIXED_X_CH4, _FIXED_X_CO2, x_n2, _FIXED_H_RH, _FIXED_L_M)
        valid_grid.append(x_h2)
        peak_indices.append(int(result["peak_index"]))
    diffs = np.diff(peak_indices)
    return {
        "x_h2_grid": valid_grid,
        "peak_indices": peak_indices,
        "pearson_r": _pearson(valid_grid, peak_indices),
        "monotonic_decreasing": bool(np.all(diffs <= 0)),
    }


def ultrasonic_co2_scan() -> dict:
    peaks = []
    alphas = []
    for x_co2 in _CO2_GRID:
        x_other = 100.0 - _FIXED_X_H2 - x_co2
        x_ch4 = 0.85 * x_other
        x_n2 = 0.15 * x_other
        result = _ultrasonic_sample(_FIXED_X_H2, x_ch4, x_co2, x_n2, _FIXED_H_RH, _FIXED_L_M)
        peaks.append(float(result["peak_abs_v"]))
        alphas.append(float(result["alpha_true_npm"]))
    return {
        "x_co2_grid": _CO2_GRID,
        "peak_abs_v": [round(v, 6) for v in peaks],
        "alpha_true_npm": [round(v, 6) for v in alphas],
        "peak_pearson_r": _pearson(_CO2_GRID, peaks),
        "alpha_pearson_r": _pearson(_CO2_GRID, alphas),
        "monotonic_decreasing_peak_abs": bool(np.all(np.diff(peaks) <= 1e-9)),
        "monotonic_increasing_alpha": bool(np.all(np.diff(alphas) >= -1e-9)),
    }


def fiber_co2_scan() -> dict:
    taus = []
    for x_co2 in _CO2_GRID:
        x_other = 100.0 - _FIXED_X_H2 - x_co2
        x_ch4 = 0.85 * x_other
        x_n2 = 0.15 * x_other
        result = _fiber_sample(_FIXED_X_H2, x_ch4, x_co2, x_n2, _FIXED_H_RH, _FIXED_L_M)
        taus.append(float(result["tau_s"]))
    return {
        "x_co2_grid": _CO2_GRID,
        "tau_ms": [round(v * 1000.0, 6) for v in taus],
        "pearson_r": _pearson(_CO2_GRID, taus),
        "monotonic_decreasing": bool(np.all(np.diff(taus) <= 1e-9)),
    }


def fiber_humidity_scan() -> dict:
    taus = []
    for humidity in _HUMIDITY_GRID:
        result = _fiber_sample(_FIXED_X_H2, _FIXED_X_CH4, _FIXED_X_CO2, _FIXED_X_N2, humidity, _FIXED_L_M)
        taus.append(float(result["tau_s"]))
    return {
        "humidity_grid": _HUMIDITY_GRID,
        "tau_ms": [round(v * 1000.0, 6) for v in taus],
        "pearson_r": _pearson(_HUMIDITY_GRID, taus),
        "monotonic_decreasing": bool(np.all(np.diff(taus) <= 1e-9)),
    }


def fiber_h2_scan() -> dict:
    taus = []
    valid_grid = []
    for x_h2 in _H2_GRID:
        x_n2 = max(0.0, 100.0 - _FIXED_X_CO2 - _FIXED_X_CH4 - x_h2)
        result = _fiber_sample(x_h2, _FIXED_X_CH4, _FIXED_X_CO2, x_n2, _FIXED_H_RH, _FIXED_L_M)
        valid_grid.append(x_h2)
        taus.append(float(result["tau_s"]))
    return {
        "x_h2_grid": valid_grid,
        "tau_ms": [round(v * 1000.0, 6) for v in taus],
        "pearson_r": _pearson(valid_grid, taus),
        "monotonic_decreasing": bool(np.all(np.diff(taus) <= 1e-9)),
    }


def fiber_length_scan() -> dict:
    t_rounds = []
    reflection_counts = []
    for l_m in _LENGTH_GRID:
        result = _fiber_sample(_FIXED_X_H2, _FIXED_X_CH4, _FIXED_X_CO2, _FIXED_X_N2, _FIXED_H_RH, l_m)
        t_rounds.append(float(result["t_round_s"]))
        reflection_counts.append(int(result["reflection_count"]))
    return {
        "length_grid": _LENGTH_GRID,
        "t_round_ms": [round(v * 1000.0, 6) for v in t_rounds],
        "reflection_count": reflection_counts,
        "pearson_r": _pearson(_LENGTH_GRID, t_rounds),
        "monotonic_increasing": bool(np.all(np.diff(t_rounds) >= -1e-12)),
    }


def build_waveform_directionality_report() -> dict:
    u_len = ultrasonic_length_scan()
    u_h2 = ultrasonic_h2_scan()
    u_co2 = ultrasonic_co2_scan()
    f_co2 = fiber_co2_scan()
    f_h2o = fiber_humidity_scan()
    f_h2 = fiber_h2_scan()
    f_len = fiber_length_scan()
    return {
        "dataset_version": "V3.1 dual-channel waveform",
        "calibration_status": "pending",
        "ultrasonic": {
            "length_scan": u_len,
            "h2_scan": u_h2,
            "co2_scan": u_co2,
            "tof_directionality_passed": bool(u_len["monotonic_increasing"] and u_h2["monotonic_decreasing"]),
            "alpha_directionality_passed": bool(u_co2["monotonic_decreasing_peak_abs"]),
        },
        "fiber_mic": {
            "co2_scan": f_co2,
            "humidity_scan": f_h2o,
            "h2_scan": f_h2,
            "length_scan": f_len,
            "alpha_directionality_passed": bool(f_co2["monotonic_decreasing"] and f_h2o["monotonic_decreasing"] and f_h2["monotonic_decreasing"]),
            "length_directionality_passed": bool(f_len["monotonic_increasing"]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="data/waveform_v3/quality/waveform_directional_report.json")
    args = parser.parse_args()

    report = build_waveform_directionality_report()
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nReport written to: {report_path}")


if __name__ == "__main__":
    main()
