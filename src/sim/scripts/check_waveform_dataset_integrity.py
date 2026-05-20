from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim_common import phase_boundaries


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
LABEL_FIELDS = ["x_H2", "x_CH4", "x_CO2", "x_N2"]
SEQUENCE_ARRAYS = [
    "slow.npy",
    "ultrasonic_scale.npy",
    "fiber_mic_scale.npy",
    "ultrasonic_int16.npy",
    "fiber_mic_int16.npy",
]


class WaveformDatasetIntegrityError(RuntimeError):
    pass


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"Pearson inputs must share shape, got {x.shape} and {y.shape}")
    x_std = float(x.std())
    y_std = float(y.std())
    if x_std == 0.0 or y_std == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _find_all_zero_sequences(array: np.ndarray, *, chunk_size: int = 64) -> list[int]:
    zero_indices: list[int] = []
    for start in range(0, array.shape[0], chunk_size):
        block = np.asarray(array[start : start + chunk_size])
        flat = block.reshape(block.shape[0], -1)
        hits = np.where(np.all(flat == 0, axis=1))[0]
        zero_indices.extend((start + int(index)) for index in hits)
    return zero_indices


def _sequence_rows_match(sequence_ids: np.ndarray, conditions: list[dict[str, str]]) -> bool:
    if len(sequence_ids) != len(conditions):
        return False
    return all(str(sequence_id) == row["sequence_id"] for sequence_id, row in zip(sequence_ids, conditions))


def _load_spec(dataset_dir: Path) -> dict:
    spec_path = dataset_dir / "metadata" / "waveform_v3_spec.json"
    return json.loads(spec_path.read_text(encoding="utf-8"))


def run_integrity_checks(
    dataset_dir: str | Path,
    *,
    min_sensor_corr: float = 0.5,
    min_env_corr: float = 0.99,
    min_correlation_rows: int = 100,
    max_t_abs_error: float = 1e-3,
    max_p_abs_error: float = 1e-5,
    max_h_abs_error: float = 1e-3,
    report_path: str | Path | None = None,
) -> dict:
    dataset_dir = Path(dataset_dir)
    sequence_dir = dataset_dir / "sequences"
    conditions = _read_csv_rows(dataset_dir / "condition_grid_sequence.csv")
    labels_csv = _read_csv_rows(dataset_dir / "labels" / "sequence_labels.csv")
    spec = _load_spec(dataset_dir)
    expected_rows = len(conditions)
    errors: list[str] = []

    sequence_ids = np.load(dataset_dir / "metadata" / "sequence_ids.npy", allow_pickle=True)
    y = np.load(dataset_dir / "labels" / "y.npy", mmap_mode="r")
    if not _sequence_rows_match(sequence_ids, conditions):
        errors.append("metadata/sequence_ids.npy is not aligned with condition_grid_sequence.csv")
    if len(labels_csv) != expected_rows:
        errors.append(f"sequence_labels.csv row count {len(labels_csv)} != condition rows {expected_rows}")
    if y.shape[0] != expected_rows:
        errors.append(f"labels/y.npy row count {y.shape[0]} != condition rows {expected_rows}")
    if y.shape[1] != len(LABEL_FIELDS):
        errors.append(f"labels/y.npy label width {y.shape[1]} != {len(LABEL_FIELDS)}")

    array_reports = {}
    loaded_arrays = {}
    for filename in SEQUENCE_ARRAYS:
        path = sequence_dir / filename
        array = np.load(path, mmap_mode="r")
        loaded_arrays[filename] = array
        zero_indices = _find_all_zero_sequences(array)
        array_reports[filename] = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "zero_sequence_count": len(zero_indices),
            "zero_sequence_first": zero_indices[:10],
        }
        if array.shape[0] != expected_rows:
            errors.append(f"sequences/{filename} row count {array.shape[0]} != condition rows {expected_rows}")
        if zero_indices:
            errors.append(f"sequences/{filename} has {len(zero_indices)} all-zero sequences")

    condition_values = {
        key: np.array([float(row[key]) for row in conditions], dtype=np.float64)
        for key in ["x_CH4", "x_CO2", "T_C_base", "P_MPa_base", "H_RH_base", "L_m_base"]
    }
    slow = loaded_arrays["slow.npy"]
    timesteps = int(spec["timesteps"])
    _, steady_start, steady_end = phase_boundaries(timesteps)
    slow_steady = np.asarray(slow[:, steady_start:steady_end, :].mean(axis=1), dtype=np.float64)
    channel_index = {name: index for index, name in enumerate(SLOW_CHANNELS)}
    correlations = {
        "V_NDIR_CH4_vs_x_CH4": _pearson(slow_steady[:, channel_index["V_NDIR_CH4"]], condition_values["x_CH4"]),
        "V_NDIR_CO2_vs_x_CO2": _pearson(slow_steady[:, channel_index["V_NDIR_CO2"]], condition_values["x_CO2"]),
        "T_C_vs_T_C_base": _pearson(slow_steady[:, channel_index["T_C"]], condition_values["T_C_base"]),
        "P_MPa_vs_P_MPa_base": _pearson(slow_steady[:, channel_index["P_MPa"]], condition_values["P_MPa_base"]),
        "H_RH_vs_H_RH_base": _pearson(slow_steady[:, channel_index["H_RH"]], condition_values["H_RH_base"]),
    }
    correlation_gate_enabled = expected_rows >= min_correlation_rows
    if correlation_gate_enabled:
        if abs(correlations["V_NDIR_CH4_vs_x_CH4"]) < min_sensor_corr:
            errors.append(
                f"abs(corr(V_NDIR_CH4, x_CH4))={abs(correlations['V_NDIR_CH4_vs_x_CH4']):.6f} < {min_sensor_corr}"
            )
        if abs(correlations["V_NDIR_CO2_vs_x_CO2"]) < min_sensor_corr:
            errors.append(
                f"abs(corr(V_NDIR_CO2, x_CO2))={abs(correlations['V_NDIR_CO2_vs_x_CO2']):.6f} < {min_sensor_corr}"
            )
        for key in ["T_C_vs_T_C_base", "P_MPa_vs_P_MPa_base", "H_RH_vs_H_RH_base"]:
            if correlations[key] < min_env_corr:
                errors.append(f"corr({key})={correlations[key]:.6f} < {min_env_corr}")

    env_abs_errors = {
        "T_C_max_abs_error": float(np.max(np.abs(slow_steady[:, channel_index["T_C"]] - condition_values["T_C_base"]))),
        "P_MPa_max_abs_error": float(np.max(np.abs(slow_steady[:, channel_index["P_MPa"]] - condition_values["P_MPa_base"]))),
        "H_RH_max_abs_error": float(np.max(np.abs(slow_steady[:, channel_index["H_RH"]] - condition_values["H_RH_base"]))),
    }
    if env_abs_errors["T_C_max_abs_error"] > max_t_abs_error:
        errors.append(f"T_C steady/base max abs error {env_abs_errors['T_C_max_abs_error']:.6g} > {max_t_abs_error}")
    if env_abs_errors["P_MPa_max_abs_error"] > max_p_abs_error:
        errors.append(f"P_MPa steady/base max abs error {env_abs_errors['P_MPa_max_abs_error']:.6g} > {max_p_abs_error}")
    if env_abs_errors["H_RH_max_abs_error"] > max_h_abs_error:
        errors.append(f"H_RH steady/base max abs error {env_abs_errors['H_RH_max_abs_error']:.6g} > {max_h_abs_error}")

    if spec.get("multi_path_phase") == "off":
        l_error = float(np.max(np.abs(slow_steady[:, channel_index["L_m"]] - condition_values["L_m_base"])))
        env_abs_errors["L_m_max_abs_error"] = l_error
        if l_error > 1e-5:
            errors.append(f"L_m steady/base max abs error {l_error:.6g} > 1e-5")

    report = {
        "dataset_dir": str(dataset_dir),
        "status": "failed" if errors else "passed",
        "condition_rows": expected_rows,
        "array_reports": array_reports,
        "correlations": correlations,
        "correlation_gate_enabled": correlation_gate_enabled,
        "min_correlation_rows": min_correlation_rows,
        "env_abs_errors": env_abs_errors,
        "steady_window": [steady_start, steady_end],
        "errors": errors,
    }
    if report_path is not None:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if errors:
        raise WaveformDatasetIntegrityError("; ".join(errors))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Check V3 waveform dataset integrity.")
    parser.add_argument("dataset_dir")
    parser.add_argument("--min-sensor-corr", type=float, default=0.5)
    parser.add_argument("--min-env-corr", type=float, default=0.99)
    parser.add_argument("--min-correlation-rows", type=int, default=100)
    parser.add_argument("--report-path")
    args = parser.parse_args()
    report = run_integrity_checks(
        args.dataset_dir,
        min_sensor_corr=args.min_sensor_corr,
        min_env_corr=args.min_env_corr,
        min_correlation_rows=args.min_correlation_rows,
        report_path=args.report_path,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
