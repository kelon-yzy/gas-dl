from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim_common.phases import phase_boundaries, phase_for_timestep


SLOW_CHANNELS = [
    "V_NDIR_CH4",
    "V_NDIR_CO2",
    "V_TCS",
    "T_C",
    "P_MPa",
    "H_RH",
    "L_m",
    "piston_position_m",
]


def _load_waveform_package(source_dir: Path) -> dict[str, object]:
    return {
        "condition": pd.read_csv(source_dir / "condition_grid_sequence.csv"),
        "sequence_index": pd.read_csv(source_dir / "sequence_index.csv"),
        "ultrasonic": np.load(source_dir / "sequences" / "ultrasonic_int16.npy", mmap_mode="r"),
        "ultrasonic_scale": np.load(source_dir / "sequences" / "ultrasonic_scale.npy", mmap_mode="r"),
        "fiber_mic": np.load(source_dir / "sequences" / "fiber_mic_int16.npy", mmap_mode="r"),
        "fiber_mic_scale": np.load(source_dir / "sequences" / "fiber_mic_scale.npy", mmap_mode="r"),
        "slow": np.load(source_dir / "sequences" / "slow.npy", mmap_mode="r"),
        "labels": np.load(source_dir / "labels" / "y.npy", mmap_mode="r"),
        "sequence_ids": np.load(source_dir / "metadata" / "sequence_ids.npy", allow_pickle=True),
    }


def _phase_aligned_timesteps(total_timesteps: int) -> list[int]:
    q1, q2, q3 = phase_boundaries(total_timesteps)
    del q1
    return [0, q2, (q2 + q3) // 2, q3]


def _stage_info_for_timestep(timestep: int, ordered_timesteps: list[int]) -> tuple[str, str, str, str]:
    stage_map = {
        0: ("baseline_stage", "calibration_control", "low", "short"),
        1: ("distance_stage", "synthetic_measurement", "mid", "mid"),
        2: ("pressure_stage", "synthetic_measurement", "high", "long"),
        3: ("purge_stage", "synthetic_measurement", "mid", "mid"),
    }
    index = ordered_timesteps.index(timestep)
    return stage_map.get(index, ("pressure_stage", "synthetic_measurement", "mid", "mid"))


def _estimate_ultrasonic_features(waveform_int16, scale_factor: float, l_m: float, t_c: float) -> dict[str, float]:
    waveform = waveform_int16.astype(np.float32) * float(scale_factor)
    peak_index = int(np.argmax(np.abs(waveform)))
    peak_amp = float(np.max(np.abs(waveform)))
    sample_rate_hz = 200000.0
    tof = peak_index / sample_rate_hz
    fft = np.fft.rfft(waveform)
    freqs = np.fft.rfftfreq(len(waveform), d=1.0 / sample_rate_hz)
    fft_index = int(np.argmax(np.abs(fft[1:])) + 1) if len(freqs) > 1 else 0
    f_peak = float(freqs[fft_index]) if fft_index < len(freqs) else 0.0
    a_fft_max = float(np.max(np.abs(fft))) if fft.size else 0.0
    sound_speed = l_m / max(tof, 1e-9)
    attenuation_alpha = -np.log(max(peak_amp, 1e-12)) / max(l_m, 1e-9)
    c_t_norm = sound_speed - 0.6 * (t_c - 25.0)
    return {
        "TOF": float(tof),
        "Amp": peak_amp,
        "f_peak": f_peak,
        "A_fft_max": a_fft_max,
        "sound_speed": float(sound_speed),
        "attenuation_alpha": float(attenuation_alpha),
        "c_sound": float(sound_speed),
        "c_T_norm": float(c_t_norm),
        "delta_Amp": float(peak_amp - 1.0),
    }


def _find_peaks(signal: np.ndarray, threshold: float, min_gap: int = 40) -> list[int]:
    peaks: list[int] = []
    for idx in range(1, len(signal) - 1):
        value = signal[idx]
        if value < threshold:
            continue
        if value < signal[idx - 1] or value < signal[idx + 1]:
            continue
        if peaks and idx - peaks[-1] < min_gap:
            if value > signal[peaks[-1]]:
                peaks[-1] = idx
            continue
        peaks.append(idx)
    return peaks


def _estimate_fiber_features(waveform_int16, scale_factor: float, l_m: float) -> dict[str, float]:
    waveform = waveform_int16.astype(np.float32) * float(scale_factor)
    envelope = np.abs(waveform)
    sample_rate_hz = 200000.0
    peak_threshold = max(float(np.max(envelope)) * 0.08, 3.0 * max(float(np.std(waveform[:100])), 1e-6))
    peak_indices = _find_peaks(envelope, peak_threshold)
    if not peak_indices:
        peak_indices = [int(np.argmax(envelope))]
    direct_index = int(peak_indices[0])
    direct_amp = float(envelope[direct_index])
    fit_end = min(len(envelope), direct_index + 800)
    fit_x = np.arange(direct_index, fit_end, dtype=np.float32) / sample_rate_hz
    fit_y = np.clip(envelope[direct_index:fit_end], 1e-9, None)
    log_y = np.log(fit_y)
    slope, _ = np.polyfit(fit_x, log_y, deg=1)
    tau_s = float(max(-1.0 / min(slope, -1e-9), 1e-6))
    t_round_s = 0.0
    if len(peak_indices) >= 2:
        t_round_s = float((peak_indices[1] - peak_indices[0]) / sample_rate_hz)
    reflection_count = max(0, len(peak_indices) - 1)
    tail_energy = float(np.sum(envelope[direct_index + 1 :]))
    alpha_est = 1.0 / max(tau_s * max(l_m, 1e-9), 1e-9)
    return {
        "tau_s": tau_s,
        "tau_ms": tau_s * 1000.0,
        "t_round_s": float(t_round_s),
        "t_round_ms": float(t_round_s * 1000.0),
        "reflection_count": float(reflection_count),
        "fiber_direct_index": float(direct_index),
        "fiber_direct_amp": direct_amp,
        "fiber_tail_energy": tail_energy,
        "fiber_alpha_est": float(alpha_est),
    }


def _estimate_environment_features(t_c: float, p_mpa: float, h_rh: float) -> dict[str, float]:
    t_k = t_c + 273.15
    p_kpa = p_mpa * 1000.0
    p_ws_kpa = 0.61121 * np.exp(17.502 * t_c / (240.97 + t_c))
    p_h2o_kpa = (h_rh / 100.0) * p_ws_kpa
    x_h2o = p_h2o_kpa / max(p_kpa, 1e-9)
    p_dry_kpa = p_kpa - p_h2o_kpa
    ah_g_m3 = 216.7 * (p_h2o_kpa / max(t_k, 1e-9))
    return {
        "T_K": float(t_k),
        "P_kPa": float(p_kpa),
        "p_ws_kPa": float(p_ws_kpa),
        "p_H2O_kPa": float(p_h2o_kpa),
        "x_H2O": float(x_h2o),
        "P_dry_kPa": float(p_dry_kpa),
        "AH_g_m3": float(ah_g_m3),
    }


def _prepare_output_dirs(output_dir: Path) -> dict[str, Path]:
    paths = {
        "feature_table": output_dir / "features" / "feature_table.csv",
        "feature_table_env": output_dir / "features" / "feature_table_env.csv",
        "feature_manifest": output_dir / "features" / "feature_manifest.json",
        "train_acoustic": output_dir / "training" / "train_acoustic.csv",
        "train_optical": output_dir / "training" / "train_optical.csv",
        "train_thermal": output_dir / "training" / "train_thermal.csv",
        "train_acoustic_env": output_dir / "training" / "train_acoustic_env.csv",
        "train_optical_env": output_dir / "training" / "train_optical_env.csv",
        "train_thermal_env": output_dir / "training" / "train_thermal_env.csv",
        "labels": output_dir / "labels" / "labels.csv",
        "condition_grid": output_dir / "condition_grid_v1.csv",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    return paths


def generate_traditional_from_waveform_v3(source_dir: str | Path, output_dir: str | Path, sequence_limit: int | None = None, timesteps: list[int] | None = None) -> dict[str, Path]:
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    package = _load_waveform_package(source_dir)

    condition = package["condition"].copy().set_index("sequence_id")
    sequence_index = package["sequence_index"].copy().set_index("sequence_id")
    ultrasonic = package["ultrasonic"]
    ultrasonic_scale = package["ultrasonic_scale"]
    fiber_mic = package["fiber_mic"]
    fiber_mic_scale = package["fiber_mic_scale"]
    slow = package["slow"]
    labels = package["labels"]
    sequence_ids = [str(value) for value in package["sequence_ids"]]

    total_timesteps = int(ultrasonic.shape[1])
    if timesteps is None:
        timesteps = _phase_aligned_timesteps(total_timesteps)
    else:
        timesteps = [int(t) for t in timesteps]

    if sequence_limit is not None:
        sequence_ids = sequence_ids[:sequence_limit]

    feature_rows = []
    feature_env_rows = []
    acoustic_rows = []
    optical_rows = []
    thermal_rows = []
    acoustic_env_rows = []
    optical_env_rows = []
    thermal_env_rows = []
    label_rows = []
    condition_rows = []
    sample_counter = 1

    baseline_ch4 = float(np.median(slow[:, timesteps[0], 0]))
    baseline_co2 = float(np.median(slow[:, timesteps[0], 1]))
    baseline_tcs = float(np.median(slow[:, timesteps[0], 2]))

    for seq_index, sequence_id in enumerate(sequence_ids):
        condition_row = condition.loc[sequence_id]
        index_row = sequence_index.loc[sequence_id]
        targets = labels[seq_index]
        for timestep in timesteps:
            sample_id = f"W{sample_counter:05d}"
            sample_counter += 1
            stage_id, status, pressure_stage, distance_stage = _stage_info_for_timestep(timestep, timesteps)
            slow_values = slow[seq_index, timestep]
            t_c = float(slow_values[3])
            p_mpa = float(slow_values[4])
            h_rh = float(slow_values[5])
            l_m = float(slow_values[6])

            ultrasonic_features = _estimate_ultrasonic_features(ultrasonic[seq_index, timestep], float(ultrasonic_scale[seq_index, timestep]), l_m=l_m, t_c=t_c)
            fiber_features = _estimate_fiber_features(fiber_mic[seq_index, timestep], float(fiber_mic_scale[seq_index, timestep]), l_m=l_m)
            env_features = _estimate_environment_features(t_c=t_c, p_mpa=p_mpa, h_rh=h_rh)

            base_row = {
                "sample_id": sample_id,
                "stage_id": stage_id,
                "status": status,
                "sequence_id": sequence_id,
                "mixture_id": index_row["mixture_id"],
                "source_timestep": int(timestep),
                "source_phase_id": phase_for_timestep(int(timestep), total_timesteps),
                "pressure_stage": pressure_stage,
                "distance_stage": distance_stage,
                "L_m": l_m,
                "piston_position_m": float(slow_values[7]),
                "V_NDIR_CH4": float(slow_values[0]),
                "V_NDIR_CO2": float(slow_values[1]),
                "V_TCS": float(slow_values[2]),
                "T_C": t_c,
                "P_MPa": p_mpa,
                "H_RH": h_rh,
                "x_H2": float(targets[0]),
                "x_CH4": float(targets[1]),
                "x_CO2": float(targets[2]),
                "x_N2": float(targets[3]),
                **ultrasonic_features,
                **fiber_features,
                **env_features,
            }
            base_row["delta_I_CH4"] = float(slow_values[0] - baseline_ch4)
            base_row["delta_I_CO2"] = float(slow_values[1] - baseline_co2)
            base_row["A_NDIR_CH4"] = abs(base_row["delta_I_CH4"])
            base_row["A_NDIR_CO2"] = abs(base_row["delta_I_CO2"])
            base_row["R_CH4"] = float(slow_values[0] / max(baseline_ch4, 1e-9))
            base_row["R_CO2"] = float(slow_values[1] / max(baseline_co2, 1e-9))
            base_row["A_CH4"] = base_row["A_NDIR_CH4"]
            base_row["A_CO2"] = base_row["A_NDIR_CO2"]
            base_row["ndir_ch4_saturated"] = False
            base_row["ndir_co2_saturated"] = False
            base_row["optical_baseline_drift_ch4"] = base_row["delta_I_CH4"]
            base_row["optical_baseline_drift_co2"] = base_row["delta_I_CO2"]
            base_row["lambda_mix_calibrated"] = float(base_row["V_TCS"])
            base_row["thermal_baseline_drift"] = float(base_row["V_TCS"] - baseline_tcs)
            base_row["delta_V_TCS"] = base_row["thermal_baseline_drift"]
            base_row["calibration_status"] = "pending"

            feature_rows.append(dict(base_row))
            feature_env_row = dict(base_row)
            feature_env_rows.append(feature_env_row)

            acoustic_rows.append({
                "sample_id": sample_id,
                "TOF": base_row["TOF"],
                "Amp": base_row["Amp"],
                "f_peak": base_row["f_peak"],
                "A_fft_max": base_row["A_fft_max"],
            })
            acoustic_env_rows.append({
                "sample_id": sample_id,
                "TOF": base_row["TOF"],
                "Amp": base_row["Amp"],
                "f_peak": base_row["f_peak"],
                "A_fft_max": base_row["A_fft_max"],
                "T_C": base_row["T_C"],
                "P_MPa": base_row["P_MPa"],
                "H_RH": base_row["H_RH"],
                "T_K": base_row["T_K"],
                "P_kPa": base_row["P_kPa"],
                "p_H2O_kPa": base_row["p_H2O_kPa"],
                "x_H2O": base_row["x_H2O"],
                "AH_g_m3": base_row["AH_g_m3"],
                "P_dry_kPa": base_row["P_dry_kPa"],
                "sound_speed": base_row["sound_speed"],
                "attenuation_alpha": base_row["attenuation_alpha"],
                "c_sound": base_row["c_sound"],
                "c_T_norm": base_row["c_T_norm"],
                "delta_Amp": base_row["delta_Amp"],
            })

            optical_rows.append({
                "sample_id": sample_id,
                "V_NDIR_CH4": base_row["V_NDIR_CH4"],
                "V_NDIR_CO2": base_row["V_NDIR_CO2"],
                "delta_I_CH4": base_row["delta_I_CH4"],
                "delta_I_CO2": base_row["delta_I_CO2"],
                "A_NDIR_CH4": base_row["A_NDIR_CH4"],
                "A_NDIR_CO2": base_row["A_NDIR_CO2"],
            })
            optical_env_rows.append({
                "sample_id": sample_id,
                "V_NDIR_CH4": base_row["V_NDIR_CH4"],
                "V_NDIR_CO2": base_row["V_NDIR_CO2"],
                "delta_I_CH4": base_row["delta_I_CH4"],
                "delta_I_CO2": base_row["delta_I_CO2"],
                "A_NDIR_CH4": base_row["A_NDIR_CH4"],
                "A_NDIR_CO2": base_row["A_NDIR_CO2"],
                "T_C": base_row["T_C"],
                "P_MPa": base_row["P_MPa"],
                "H_RH": base_row["H_RH"],
                "T_K": base_row["T_K"],
                "P_kPa": base_row["P_kPa"],
                "p_H2O_kPa": base_row["p_H2O_kPa"],
                "x_H2O": base_row["x_H2O"],
                "AH_g_m3": base_row["AH_g_m3"],
                "P_dry_kPa": base_row["P_dry_kPa"],
            })

            thermal_rows.append({
                "sample_id": sample_id,
                "V_TCS": base_row["V_TCS"],
                "T_C": base_row["T_C"],
                "P_MPa": base_row["P_MPa"],
                "H_RH": base_row["H_RH"],
            })
            thermal_env_rows.append({
                "sample_id": sample_id,
                "V_TCS": base_row["V_TCS"],
                "delta_V_TCS": base_row["delta_V_TCS"],
                "T_C": base_row["T_C"],
                "P_MPa": base_row["P_MPa"],
                "H_RH": base_row["H_RH"],
                "T_K": base_row["T_K"],
                "P_kPa": base_row["P_kPa"],
                "p_H2O_kPa": base_row["p_H2O_kPa"],
                "x_H2O": base_row["x_H2O"],
                "AH_g_m3": base_row["AH_g_m3"],
                "P_dry_kPa": base_row["P_dry_kPa"],
            })

            label_rows.append({
                "sample_id": sample_id,
                "x_H2": base_row["x_H2"],
                "x_CH4": base_row["x_CH4"],
                "x_CO2": base_row["x_CO2"],
                "x_N2": base_row["x_N2"],
            })
            condition_rows.append({
                "sample_id": sample_id,
                "mixture_id": index_row["mixture_id"],
                "stage_id": stage_id,
                "x_H2": base_row["x_H2"],
                "x_CH4": base_row["x_CH4"],
                "x_CO2": base_row["x_CO2"],
                "x_N2": base_row["x_N2"],
                "T_C": base_row["T_C"],
                "P_MPa": base_row["P_MPa"],
                "H_RH": base_row["H_RH"],
                "L_m": base_row["L_m"],
                "piston_position_m": base_row["piston_position_m"],
                "pressure_stage": pressure_stage,
                "distance_stage": distance_stage,
                "repeat_id": 0,
                "status": status,
                "source_timestep": int(timestep),
                "source_phase_id": phase_for_timestep(int(timestep), total_timesteps),
            })

    paths = _prepare_output_dirs(output_dir)
    pd.DataFrame(feature_rows).to_csv(paths["feature_table"], index=False)
    pd.DataFrame(feature_env_rows).to_csv(paths["feature_table_env"], index=False)
    pd.DataFrame(acoustic_rows).to_csv(paths["train_acoustic"], index=False)
    pd.DataFrame(optical_rows).to_csv(paths["train_optical"], index=False)
    pd.DataFrame(thermal_rows).to_csv(paths["train_thermal"], index=False)
    pd.DataFrame(acoustic_env_rows).to_csv(paths["train_acoustic_env"], index=False)
    pd.DataFrame(optical_env_rows).to_csv(paths["train_optical_env"], index=False)
    pd.DataFrame(thermal_env_rows).to_csv(paths["train_thermal_env"], index=False)
    pd.DataFrame(label_rows).to_csv(paths["labels"], index=False)
    pd.DataFrame(condition_rows).to_csv(paths["condition_grid"], index=False)
    paths["feature_manifest"].write_text(
        json.dumps(
            {
                "dataset_version": "V3.1 dual-channel waveform",
                "sampling_timesteps": timesteps,
                "sampling_policy": "phase_aligned_default",
                "acoustic_columns": list(pd.DataFrame(acoustic_rows).columns),
                "optical_columns": list(pd.DataFrame(optical_rows).columns),
                "training_acoustic_columns": list(pd.DataFrame(acoustic_rows).columns),
                "training_optical_columns": list(pd.DataFrame(optical_rows).columns),
                "feature_table_columns": list(pd.DataFrame(feature_rows).columns),
                "feature_table_env_columns": list(pd.DataFrame(feature_env_rows).columns),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate traditional tables from V3.1 dual-channel waveform package.")
    parser.add_argument("--source-dir", default="data/waveform_v3")
    parser.add_argument("--output-dir", default="outputs/exp01_traditional")
    parser.add_argument("--sequence-limit", type=int, default=None)
    parser.add_argument("--timesteps", default=None)
    return parser


def _parse_timestep_arg(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    generate_traditional_from_waveform_v3(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        sequence_limit=args.sequence_limit,
        timesteps=_parse_timestep_arg(args.timesteps),
    )


if __name__ == "__main__":
    main()
