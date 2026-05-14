"""从 V3 waveform 数据包抽样生成传统模型可用的表格特征数据包。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
  sys.path.insert(0, str(Path(__file__).resolve().parent))
  sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
  from generate_v1_dataset import PROCESSING_PARAMS
  from sim_common.phases import phase_boundaries, phase_for_timestep
else:
  from .generate_v1_dataset import PROCESSING_PARAMS
  from sim_common.phases import phase_boundaries, phase_for_timestep


# stage_id / status 映射沿用 V1 patent 表语义；timesteps 列仅用作显式 CLI 回退默认值，
# 新代码默认走 ``_phase_aligned_timesteps`` 按源数据 total_timesteps 自适应。
TRADITIONAL_SAMPLE_STAGES = [
  (0, "baseline_stage", "calibration_control"),
  (20, "distance_stage", "synthetic_measurement"),
  (70, "pressure_stage", "synthetic_measurement"),
  (100, "purge_stage", "synthetic_measurement"),
]


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="Generate patent-style traditional tables from V3 waveform package.")
  parser.add_argument("--source-dir", default="output_waveform_sequence")
  parser.add_argument("--output-dir", default="output_waveform_traditional")
  parser.add_argument("--sequence-limit", type=int, default=None)
  parser.add_argument(
    "--timesteps",
    default=None,
    help="Comma-separated timestep list. Defaults to phase-aligned sampling from the source package.",
  )
  return parser


def _load_waveform_package(source_dir: Path) -> dict[str, object]:
  return {
    "condition": pd.read_csv(source_dir / "condition_grid_sequence.csv"),
    "sequence_index": pd.read_csv(source_dir / "sequence_index.csv"),
    "waveform": np.load(source_dir / "sequences" / "waveform_int16.npy", mmap_mode="r"),
    "waveform_scale": np.load(source_dir / "sequences" / "waveform_scale.npy", mmap_mode="r"),
    "slow": np.load(source_dir / "sequences" / "slow.npy", mmap_mode="r"),
    "labels": np.load(source_dir / "labels" / "y.npy", mmap_mode="r"),
    "sequence_ids": np.load(source_dir / "metadata" / "sequence_ids.npy", allow_pickle=True),
    "slow_channel_names": np.load(source_dir / "metadata" / "slow_channel_names.npy", allow_pickle=True),
    "label_names": np.load(source_dir / "metadata" / "label_names.npy", allow_pickle=True),
  }


def _parse_timestep_arg(value: str | None) -> list[int] | None:
  if not value:
    return None
  return [int(item.strip()) for item in value.split(",") if item.strip()]


def _phase_aligned_timesteps(total_timesteps: int) -> list[int]:
  """按 sim_common.phases 边界返回默认采样点 ``[0, q2, (q2+q3)//2, q3]``。

  对 ``total_timesteps=120``：phase_boundaries → (20, 70, 100)，
  采样点 ``[0, 70, 85, 100]``，phase 归属 ``[baseline, steady, steady, recovery]``。
  """
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


def _estimate_acoustic_features(waveform_int16, scale_factor: float, l_m: float, t_c: float, p_mpa: float, h_rh: float) -> dict[str, float]:
  waveform = waveform_int16.astype(np.float32) * float(scale_factor)
  peak_index = int(np.argmax(np.abs(waveform)))
  peak_amp = float(np.max(np.abs(waveform)))
  sample_rate_hz = 200000.0
  tof = peak_index / sample_rate_hz
  fft = np.fft.rfft(waveform)
  freqs = np.fft.rfftfreq(len(waveform), d=1.0 / sample_rate_hz)
  if len(freqs) > 1:
    fft_index = int(np.argmax(np.abs(fft[1:])) + 1)
  else:
    fft_index = 0
  f_peak = float(freqs[fft_index]) if fft_index < len(freqs) else 0.0
  a_fft_max = float(np.max(np.abs(fft))) if fft.size else 0.0
  sound_speed = l_m / max(tof, 1e-9)
  attenuation_alpha = -np.log(max(peak_amp / float(PROCESSING_PARAMS["amp_reference"]), 1e-12)) / max(l_m, 1e-9)
  c_t_norm = sound_speed - 0.6 * (t_c - 25.0)
  delta_amp = peak_amp - float(PROCESSING_PARAMS["amp_reference"])
  return {
    "TOF": float(tof),
    "Amp": peak_amp,
    "f_peak": f_peak,
    "A_fft_max": a_fft_max,
    "sound_speed": float(sound_speed),
    "attenuation_alpha": float(attenuation_alpha),
    "c_sound": float(sound_speed),
    "c_T_norm": float(c_t_norm),
    "delta_Amp": float(delta_amp),
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


def _modal_rows_for_sample(sample_id: str, base: dict[str, float]) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
  acoustic = {
    "sample_id": sample_id,
    "TOF": base["TOF"],
    "Amp": base["Amp"],
    "f_peak": base["f_peak"],
    "A_fft_max": base["A_fft_max"],
    "L_m": base["L_m"],
    "T_C": base["T_C"],
    "P_MPa": base["P_MPa"],
    "H_RH": base["H_RH"],
  }
  optical = {
    "sample_id": sample_id,
    "V_NDIR_CH4": base["V_NDIR_CH4"],
    "V_NDIR_CO2": base["V_NDIR_CO2"],
    "delta_I_CH4": base["delta_I_CH4"],
    "delta_I_CO2": base["delta_I_CO2"],
    "T_C": base["T_C"],
    "P_MPa": base["P_MPa"],
    "H_RH": base["H_RH"],
  }
  thermal = {
    "sample_id": sample_id,
    "V_TCS": base["V_TCS"],
    "T_C": base["T_C"],
    "P_MPa": base["P_MPa"],
    "H_RH": base["H_RH"],
  }
  return acoustic, optical, thermal


def _env_modal_rows_for_sample(sample_id: str, base: dict[str, float]) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
  acoustic = {
    "sample_id": sample_id,
    "TOF": base["TOF"],
    "Amp": base["Amp"],
    "f_peak": base["f_peak"],
    "A_fft_max": base["A_fft_max"],
    "L_m": base["L_m"],
    "c_sound": base["c_sound"],
    "c_T_norm": base["c_T_norm"],
    "delta_Amp": base["delta_Amp"],
    "T_C": base["T_C"],
    "P_MPa": base["P_MPa"],
    "H_RH": base["H_RH"],
    "T_K": base["T_K"],
    "P_kPa": base["P_kPa"],
    "p_H2O_kPa": base["p_H2O_kPa"],
    "x_H2O": base["x_H2O"],
    "AH_g_m3": base["AH_g_m3"],
    "P_dry_kPa": base["P_dry_kPa"],
  }
  optical = {
    "sample_id": sample_id,
    "V_NDIR_CH4": base["V_NDIR_CH4"],
    "V_NDIR_CO2": base["V_NDIR_CO2"],
    "delta_I_CH4": base["delta_I_CH4"],
    "delta_I_CO2": base["delta_I_CO2"],
    "R_CH4": base["R_CH4"],
    "R_CO2": base["R_CO2"],
    "A_CH4": base["A_CH4"],
    "A_CO2": base["A_CO2"],
    "T_C": base["T_C"],
    "P_MPa": base["P_MPa"],
    "H_RH": base["H_RH"],
    "T_K": base["T_K"],
    "P_kPa": base["P_kPa"],
    "p_H2O_kPa": base["p_H2O_kPa"],
    "x_H2O": base["x_H2O"],
    "AH_g_m3": base["AH_g_m3"],
    "P_dry_kPa": base["P_dry_kPa"],
  }
  thermal = {
    "sample_id": sample_id,
    "V_TCS": base["V_TCS"],
    "delta_V_TCS": base["delta_V_TCS"],
    "T_C": base["T_C"],
    "P_MPa": base["P_MPa"],
    "H_RH": base["H_RH"],
    "T_K": base["T_K"],
    "P_kPa": base["P_kPa"],
    "p_H2O_kPa": base["p_H2O_kPa"],
    "x_H2O": base["x_H2O"],
    "AH_g_m3": base["AH_g_m3"],
    "P_dry_kPa": base["P_dry_kPa"],
  }
  return acoustic, optical, thermal


def generate_traditional_from_waveform_v3(
  source_dir: str | Path,
  output_dir: str | Path,
  sequence_limit: int | None = None,
  timesteps: list[int] | None = None,
) -> dict[str, Path]:
  source_dir = Path(source_dir)
  output_dir = Path(output_dir)
  package = _load_waveform_package(source_dir)

  condition = package["condition"].copy()
  sequence_index = package["sequence_index"].copy()
  waveform = package["waveform"]
  waveform_scale = package["waveform_scale"]
  slow = package["slow"]
  labels = package["labels"]
  sequence_ids = [str(value) for value in package["sequence_ids"]]

  total_timesteps = int(waveform.shape[1])
  if timesteps is None:
    timesteps = _phase_aligned_timesteps(total_timesteps)
    sampling_policy = "phase_aligned_default"
  else:
    timesteps = [int(t) for t in timesteps]
    sampling_policy = "explicit_timesteps"

  source_phase_lookup = {int(t): phase_for_timestep(int(t), total_timesteps) for t in timesteps}

  sequence_index = sequence_index.set_index("sequence_id")
  condition = condition.set_index("sequence_id")
  if sequence_limit is not None:
    sequence_ids = sequence_ids[:sequence_limit]

  acoustic_rows = []
  optical_rows = []
  thermal_rows = []
  acoustic_env_rows = []
  optical_env_rows = []
  thermal_env_rows = []
  feature_rows = []
  feature_env_rows = []
  label_rows = []
  condition_rows = []

  sample_counter = 1
  baseline_amp = float(np.median(np.abs(waveform[0, timesteps[0]].astype(np.float32) * waveform_scale[0, timesteps[0]]))) if sequence_ids else 1.0
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
      waveform_features = _estimate_acoustic_features(
        waveform[seq_index, timestep],
        float(waveform_scale[seq_index, timestep]),
        l_m=l_m,
        t_c=t_c,
        p_mpa=p_mpa,
        h_rh=h_rh,
      )
      env_features = _estimate_environment_features(t_c=t_c, p_mpa=p_mpa, h_rh=h_rh)
      base_row = {
        "sample_id": sample_id,
        "stage_id": stage_id,
        "L_m": l_m,
        "TOF": waveform_features["TOF"],
        "Amp": waveform_features["Amp"],
        "f_peak": waveform_features["f_peak"],
        "A_fft_max": waveform_features["A_fft_max"],
        "sound_speed": waveform_features["sound_speed"],
        "attenuation_alpha": waveform_features["attenuation_alpha"],
        "V_NDIR_CH4": float(slow_values[0]),
        "V_NDIR_CO2": float(slow_values[1]),
        "V_TCS": float(slow_values[2]),
        "T_C": t_c,
        "P_MPa": p_mpa,
        "H_RH": h_rh,
        "piston_position_m": float(slow_values[7]),
        "x_H2": float(targets[0]),
        "x_CH4": float(targets[1]),
        "x_CO2": float(targets[2]),
        "x_N2": float(targets[3]),
        "calibration_status": "pending",
        "waveform_scale": float(waveform_scale[seq_index, timestep]),
        "mixture_id": str(index_row["mixture_id"]),
      }
      base_row["delta_I_CH4"] = baseline_ch4 - base_row["V_NDIR_CH4"]
      base_row["delta_I_CO2"] = baseline_co2 - base_row["V_NDIR_CO2"]
      base_row["A_NDIR_CH4"] = max(0.0, -np.log(max(base_row["V_NDIR_CH4"], 1e-9) / max(baseline_ch4, 1e-9)))
      base_row["A_NDIR_CO2"] = max(0.0, -np.log(max(base_row["V_NDIR_CO2"], 1e-9) / max(baseline_co2, 1e-9)))
      base_row["R_CH4"] = base_row["V_NDIR_CH4"] / max(baseline_ch4, 1e-9)
      base_row["R_CO2"] = base_row["V_NDIR_CO2"] / max(baseline_co2, 1e-9)
      base_row["A_CH4"] = base_row["A_NDIR_CH4"]
      base_row["A_CO2"] = base_row["A_NDIR_CO2"]
      base_row["ndir_ch4_saturated"] = int(base_row["A_NDIR_CH4"] > 2.0)
      base_row["ndir_co2_saturated"] = int(base_row["A_NDIR_CO2"] > 2.0)
      base_row["optical_baseline_drift_ch4"] = base_row["delta_I_CH4"]
      base_row["optical_baseline_drift_co2"] = base_row["delta_I_CO2"]
      base_row["thermal_baseline_drift"] = base_row["V_TCS"] - baseline_tcs
      base_row["delta_V_TCS"] = base_row["thermal_baseline_drift"]
      base_row["lambda_mix_calibrated"] = 0.026 + 0.0012 * base_row["x_H2"] - 0.00018 * base_row["x_CO2"] + 0.00004 * (t_c - 20.0)
      base_row.update(env_features)
      base_row["c_sound"] = waveform_features["c_sound"]
      base_row["c_T_norm"] = waveform_features["c_T_norm"]
      base_row["delta_Amp"] = waveform_features["delta_Amp"]
      acoustic_row, optical_row, thermal_row = _modal_rows_for_sample(sample_id, base_row)
      acoustic_env_row, optical_env_row, thermal_env_row = _env_modal_rows_for_sample(sample_id, base_row)
      acoustic_rows.append(acoustic_row)
      optical_rows.append(optical_row)
      thermal_rows.append(thermal_row)
      acoustic_env_rows.append(acoustic_env_row)
      optical_env_rows.append(optical_env_row)
      thermal_env_rows.append(thermal_env_row)
      feature_rows.append(
        {
          "sample_id": sample_id,
          "stage_id": stage_id,
          "TOF": base_row["TOF"],
          "Amp": base_row["Amp"],
          "f_peak": base_row["f_peak"],
          "A_fft_max": base_row["A_fft_max"],
          "L_m": base_row["L_m"],
          "sound_speed": base_row["sound_speed"],
          "attenuation_alpha": base_row["attenuation_alpha"],
          "V_NDIR_CH4": base_row["V_NDIR_CH4"],
          "V_NDIR_CO2": base_row["V_NDIR_CO2"],
          "delta_I_CH4": base_row["delta_I_CH4"],
          "delta_I_CO2": base_row["delta_I_CO2"],
          "A_NDIR_CH4": base_row["A_NDIR_CH4"],
          "A_NDIR_CO2": base_row["A_NDIR_CO2"],
          "ndir_ch4_saturated": base_row["ndir_ch4_saturated"],
          "ndir_co2_saturated": base_row["ndir_co2_saturated"],
          "V_TCS": base_row["V_TCS"],
          "lambda_mix_calibrated": base_row["lambda_mix_calibrated"],
          "optical_baseline_drift_ch4": base_row["optical_baseline_drift_ch4"],
          "optical_baseline_drift_co2": base_row["optical_baseline_drift_co2"],
          "thermal_baseline_drift": base_row["thermal_baseline_drift"],
          "T_C": base_row["T_C"],
          "P_MPa": base_row["P_MPa"],
          "H_RH": base_row["H_RH"],
          "piston_position_m": base_row["piston_position_m"],
          "pressure_stage": pressure_stage,
          "distance_stage": distance_stage,
          "x_H2": base_row["x_H2"],
          "x_CH4": base_row["x_CH4"],
          "x_CO2": base_row["x_CO2"],
          "x_N2": base_row["x_N2"],
          "calibration_status": base_row["calibration_status"],
          "waveform_scale": base_row["waveform_scale"],
        }
      )
      feature_env_row = dict(feature_rows[-1])
      feature_env_row.update(
        {
          "T_K": base_row["T_K"],
          "P_kPa": base_row["P_kPa"],
          "p_ws_kPa": base_row["p_ws_kPa"],
          "p_H2O_kPa": base_row["p_H2O_kPa"],
          "x_H2O": base_row["x_H2O"],
          "P_dry_kPa": base_row["P_dry_kPa"],
          "AH_g_m3": base_row["AH_g_m3"],
          "c_sound": base_row["c_sound"],
          "c_T_norm": base_row["c_T_norm"],
          "delta_Amp": base_row["delta_Amp"],
          "R_CH4": base_row["R_CH4"],
          "R_CO2": base_row["R_CO2"],
          "A_CH4": base_row["A_CH4"],
          "A_CO2": base_row["A_CO2"],
          "delta_V_TCS": base_row["delta_V_TCS"],
        }
      )
      feature_env_rows.append(feature_env_row)
      label_rows.append(
        {
          "sample_id": sample_id,
          "x_H2": base_row["x_H2"],
          "x_CH4": base_row["x_CH4"],
          "x_CO2": base_row["x_CO2"],
        }
      )
      condition_rows.append(
        {
          "sample_id": sample_id,
          "mixture_id": base_row["mixture_id"],
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
          "repeat_id": timesteps.index(timestep) + 1,
          "status": status,
          "source_timestep": int(timestep),
          "source_phase_id": source_phase_lookup[int(timestep)],
        }
      )

  (output_dir / "training").mkdir(parents=True, exist_ok=True)
  (output_dir / "features").mkdir(parents=True, exist_ok=True)
  (output_dir / "labels").mkdir(parents=True, exist_ok=True)

  pd.DataFrame(acoustic_rows).to_csv(output_dir / "training" / "train_acoustic.csv", index=False)
  pd.DataFrame(optical_rows).to_csv(output_dir / "training" / "train_optical.csv", index=False)
  pd.DataFrame(thermal_rows).to_csv(output_dir / "training" / "train_thermal.csv", index=False)
  pd.DataFrame(acoustic_env_rows).to_csv(output_dir / "training" / "train_acoustic_env.csv", index=False)
  pd.DataFrame(optical_env_rows).to_csv(output_dir / "training" / "train_optical_env.csv", index=False)
  pd.DataFrame(thermal_env_rows).to_csv(output_dir / "training" / "train_thermal_env.csv", index=False)
  pd.DataFrame(feature_rows).to_csv(output_dir / "features" / "feature_table.csv", index=False)
  pd.DataFrame(feature_env_rows).to_csv(output_dir / "features" / "feature_table_env.csv", index=False)
  pd.DataFrame(label_rows).to_csv(output_dir / "labels" / "labels.csv", index=False)
  pd.DataFrame(condition_rows).to_csv(output_dir / "condition_grid_v1.csv", index=False)

  summary = {
    "source_dir": str(source_dir.resolve()),
    "output_dir": str(output_dir.resolve()),
    "sequence_count": len(sequence_ids),
    "timesteps": list(timesteps),
    "sampling_policy": sampling_policy,
    "source_phases": [source_phase_lookup[int(t)] for t in timesteps],
    "sample_count": len(condition_rows),
    "dataset_version": "waveform_v3_to_traditional_features",
    "calibration_status": "pending",
  }
  (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
  return {
    "train_acoustic": output_dir / "training" / "train_acoustic.csv",
    "train_optical": output_dir / "training" / "train_optical.csv",
    "train_thermal": output_dir / "training" / "train_thermal.csv",
    "train_acoustic_env": output_dir / "training" / "train_acoustic_env.csv",
    "train_optical_env": output_dir / "training" / "train_optical_env.csv",
    "train_thermal_env": output_dir / "training" / "train_thermal_env.csv",
    "feature_table": output_dir / "features" / "feature_table.csv",
    "feature_table_env": output_dir / "features" / "feature_table_env.csv",
    "labels": output_dir / "labels" / "labels.csv",
    "condition_grid": output_dir / "condition_grid_v1.csv",
    "summary": output_dir / "summary.json",
  }


def main(argv: list[str] | None = None) -> dict[str, Path]:
  args = build_parser().parse_args(argv)
  return generate_traditional_from_waveform_v3(
    args.source_dir,
    args.output_dir,
    sequence_limit=args.sequence_limit,
    timesteps=_parse_timestep_arg(args.timesteps),
  )


if __name__ == "__main__":
  main()
