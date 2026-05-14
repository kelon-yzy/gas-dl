import argparse
import random
import sys
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.acoustic_fiber_mic_v3 import (
    CALIBRATION_STATUS as FIBER_MIC_CALIBRATION_STATUS,
    FiberMicV3Spec,
    simulate_fiber_mic_measurement,
)
from scripts.acoustic_waveform_v3 import (
    CALIBRATION_STATUS as ULTRASONIC_CALIBRATION_STATUS,
    CENTER_FREQUENCY_HZ,
    MEASUREMENT_WINDOW_S as ULTRASONIC_MEASUREMENT_WINDOW_S,
    SAMPLE_RATE_HZ,
    WAVEFORM_SAMPLES as ULTRASONIC_WAVEFORM_SAMPLES,
    WaveformV3Spec,
    simulate_waveform_measurement,
)
from scripts.acoustic_v2 import _hidden_sound_speed_v2
from scripts.generate_v1_dataset import PROCESSING_PARAMS, _generate_main_features
from sim_common import (
    FOUR_COMPONENT_LABEL_FIELDS,
    build_common_summary,
    build_index_rows,
    build_label_rows,
    build_split_rows,
    build_synthetic_condition_rows_four_component,
    collect_split_warnings,
    compute_split_distribution,
    fit_z_score_scalers,
    fmt,
    phase_boundaries,
    split_mixture_ids,
    write_csv,
    write_json,
)
from sim_v2.constants import DEFAULT_DT_S, DEFAULT_TIMESTEPS


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
SLOW_SEQUENCE_FIELDS = ["sequence_id", "timestep", "timestamp_s", "phase_id"] + SLOW_CHANNELS
SLOW_MODAL_GROUPS = {
    "optical": ["V_NDIR_CH4", "V_NDIR_CO2"],
    "thermal": ["V_TCS"],
    "environment": ["T_C", "P_MPa", "H_RH", "L_m", "piston_position_m"],
}
VALID_STORAGE_FORMATS = {"memmap", "npz", "both"}
DATASET_VERSION = "V3.1"
DATASET_VERSION_LABEL = "V3.1 dual-channel waveform"
GENERATION_SEED = 20260514


def generate_waveform_dataset(
    output_dir,
    sequence_count=10000,
    timesteps=DEFAULT_TIMESTEPS,
    seed=GENERATION_SEED,
    dt_s=DEFAULT_DT_S,
    storage="memmap",
):
    if timesteps < 4:
        raise ValueError("timesteps must be >= 4")
    if storage not in VALID_STORAGE_FORMATS:
        raise ValueError(f"storage must be one of {sorted(VALID_STORAGE_FORMATS)}, got {storage}")

    output_dir = Path(output_dir)
    rng = random.Random(seed)
    paths = build_waveform_output_paths(output_dir)
    ultrasonic_spec = WaveformV3Spec()
    fiber_mic_spec = FiberMicV3Spec()
    conditions = _waveform_sequence_condition_rows_four_component(sequence_count, rng, ultrasonic_spec)
    sequence_ids = [row["sequence_id"] for row in conditions]
    labels = np.array(
        [[float(row[name]) for name in FOUR_COMPONENT_LABEL_FIELDS] for row in conditions],
        dtype=np.float32,
    )
    arrays = _build_waveform_sequence_arrays(
        conditions,
        timesteps,
        dt_s,
        rng,
        paths,
        storage,
        ultrasonic_spec,
        fiber_mic_spec,
    )

    train_ids, val_ids, test_ids = split_mixture_ids([row["mixture_id"] for row in conditions], seed=seed)
    split_rows = build_split_rows(conditions, train_ids, val_ids, test_ids)
    train_sequence_ids = {row["sequence_id"] for row in split_rows["train"]}
    train_indexes = [index for index, sequence_id in enumerate(sequence_ids) if sequence_id in train_sequence_ids]
    slow_scaler, slow_modal_scaler = fit_z_score_scalers(
        arrays["slow"],
        train_indexes,
        channel_names=SLOW_CHANNELS,
        modal_groups=SLOW_MODAL_GROUPS,
        transform_target="slow",
    )

    write_csv(paths["sequence_index"], ["sequence_id", "mixture_id", "stage_profile", "status", "n_timesteps", "dt_s"], build_index_rows(conditions, timesteps, dt_s))
    write_csv(paths["condition_grid_sequence"], ["sequence_id", "mixture_id", "x_H2", "x_CH4", "x_CO2", "x_N2", "T_C_base", "P_MPa_base", "H_RH_base", "L_m_base", "status"], conditions)
    write_csv(paths["slow_sequence_long"], SLOW_SEQUENCE_FIELDS, arrays["slow_rows"])
    write_csv(paths["sequence_labels"], ["sequence_id", *FOUR_COMPONENT_LABEL_FIELDS], build_label_rows(conditions, FOUR_COMPONENT_LABEL_FIELDS))
    write_csv(paths["train_split"], ["sequence_id", "mixture_id"], split_rows["train"])
    write_csv(paths["val_split"], ["sequence_id", "mixture_id"], split_rows["val"])
    write_csv(paths["test_split"], ["sequence_id", "mixture_id"], split_rows["test"])

    _write_waveform_metadata(paths, sequence_ids, ultrasonic_spec, fiber_mic_spec, timesteps, dt_s)
    _write_label_array(paths, labels, storage)
    if storage in {"npz", "both"}:
        np.savez_compressed(
            paths["waveform_sequence_npz"],
            ultrasonic=np.asarray(arrays["ultrasonic"]),
            ultrasonic_scale=np.asarray(arrays["ultrasonic_scale"]),
            fiber_mic=np.asarray(arrays["fiber_mic"]),
            fiber_mic_scale=np.asarray(arrays["fiber_mic_scale"]),
            slow=np.asarray(arrays["slow"]),
            y=labels,
            sequence_ids=np.array(sequence_ids),
            slow_channel_names=np.array(SLOW_CHANNELS),
            label_names=np.array(FOUR_COMPONENT_LABEL_FIELDS),
        )

    write_json(paths["scaler_slow_sequence"], slow_scaler)
    write_json(paths["scaler_slow_sequence_modal"], slow_modal_scaler)

    split_distribution = compute_split_distribution(conditions, split_rows, FOUR_COMPONENT_LABEL_FIELDS)
    split_warnings = collect_split_warnings(split_distribution, FOUR_COMPONENT_LABEL_FIELDS)
    write_json(
        paths["quality_summary"],
        _waveform_quality_summary(
            sequence_ids=sequence_ids,
            timesteps=timesteps,
            split_distribution=split_distribution,
            split_warnings=split_warnings,
            storage=storage,
            ultrasonic_spec=ultrasonic_spec,
            fiber_mic_spec=fiber_mic_spec,
        ),
    )
    paths["readme"].parent.mkdir(parents=True, exist_ok=True)
    paths["readme"].write_text(
        _waveform_readme(
            sequence_count=len(sequence_ids),
            timesteps=timesteps,
            split_distribution=split_distribution,
            storage=storage,
        ),
        encoding="utf-8",
    )
    return paths


def build_waveform_output_paths(output_dir):
    output_dir = Path(output_dir)
    return {
        "sequence_index": output_dir / "sequence_index.csv",
        "condition_grid_sequence": output_dir / "condition_grid_sequence.csv",
        "slow_sequence_long": output_dir / "sequences" / "slow_sequence_long.csv",
        "ultrasonic_int16": output_dir / "sequences" / "ultrasonic_int16.npy",
        "ultrasonic_scale": output_dir / "sequences" / "ultrasonic_scale.npy",
        "fiber_mic_int16": output_dir / "sequences" / "fiber_mic_int16.npy",
        "fiber_mic_scale": output_dir / "sequences" / "fiber_mic_scale.npy",
        "slow_npy": output_dir / "sequences" / "slow.npy",
        "waveform_sequence_npz": output_dir / "sequences" / "waveform_sequence.npz",
        "y_npy": output_dir / "labels" / "y.npy",
        "sequence_labels": output_dir / "labels" / "sequence_labels.csv",
        "sequence_ids_npy": output_dir / "metadata" / "sequence_ids.npy",
        "slow_channel_names_npy": output_dir / "metadata" / "slow_channel_names.npy",
        "label_names_npy": output_dir / "metadata" / "label_names.npy",
        "waveform_v3_spec": output_dir / "metadata" / "waveform_v3_spec.json",
        "train_split": output_dir / "splits" / "train_sequence_ids.csv",
        "val_split": output_dir / "splits" / "val_sequence_ids.csv",
        "test_split": output_dir / "splits" / "test_sequence_ids.csv",
        "scaler_slow_sequence": output_dir / "scalers" / "scaler_slow_sequence.json",
        "scaler_slow_sequence_modal": output_dir / "scalers" / "scaler_slow_sequence_modal.json",
        "quality_summary": output_dir / "quality" / "waveform_quality_summary.json",
        "readme": output_dir / "README.md",
    }


def _waveform_sequence_condition_rows_four_component(sequence_count, rng, spec):
    rows = []
    requested = max(8, sequence_count)
    while len(rows) < sequence_count:
        candidates = build_synthetic_condition_rows_four_component(requested, rng)
        for candidate in candidates:
            if _condition_fits_waveform_window(candidate, spec):
                rows.append(candidate)
                if len(rows) == sequence_count:
                    break
        requested = int(requested * 1.25) + 1

    for index, row in enumerate(rows, start=1):
        row["sequence_id"] = f"Q{index:06d}"
    return rows


def _condition_fits_waveform_window(condition, spec):
    c_sound = _hidden_sound_speed_v2(
        0.0,
        0.0,
        0.0,
        100.0,
        float(condition["T_C_base"]),
    )
    peak_index = int(round(float(condition["L_m_base"]) / c_sound * spec.sample_rate_hz))
    return 0 <= peak_index < spec.waveform_samples


def _build_waveform_sequence_arrays(conditions, timesteps, dt_s, rng, paths, storage, ultrasonic_spec, fiber_mic_spec):
    sequence_count = len(conditions)
    arrays = _open_waveform_arrays(paths, sequence_count, timesteps, storage, ultrasonic_spec, fiber_mic_spec)
    slow_rows = []
    q1, q2, q3 = phase_boundaries(timesteps)

    for seq_index, condition in enumerate(conditions):
        baseline_main = _generate_main_features(
            {
                "x_H2": "0.000000",
                "x_CH4": "0.000000",
                "x_CO2": "0.000000",
                "x_N2": "100.000000",
                "T_C": condition["T_C_base"],
                "P_MPa": condition["P_MPa_base"],
                "H_RH": condition["H_RH_base"],
                "L_m": condition["L_m_base"],
            },
            rng,
            PROCESSING_PARAMS,
        )
        target_main = _generate_main_features(
            {
                "x_H2": condition["x_H2"],
                "x_CH4": condition["x_CH4"],
                "x_CO2": condition["x_CO2"],
                "x_N2": condition["x_N2"],
                "T_C": condition["T_C_base"],
                "P_MPa": condition["P_MPa_base"],
                "H_RH": condition["H_RH_base"],
                "L_m": condition["L_m_base"],
            },
            rng,
            PROCESSING_PARAMS,
        )
        for timestep in range(timesteps):
            phase_id = _phase_for_timestep_with_bounds(timestep, q1, q2, q3)
            blend = _phase_blend(timestep, q1, q2, q3)
            current = _blend_main_features(baseline_main, target_main, blend)
            composition = _blend_composition(condition, blend)
            current["T_C"] = float(condition["T_C_base"])
            current["P_MPa"] = float(condition["P_MPa_base"])
            current["H_RH"] = float(condition["H_RH_base"])
            current["L_m"] = float(condition["L_m_base"])
            current["piston_position_m"] = float(condition["L_m_base"])
            slow_values = [
                float(current["V_NDIR_CH4"]),
                float(current["V_NDIR_CO2"]),
                float(current["V_TCS"]),
                float(current["T_C"]),
                float(current["P_MPa"]),
                float(current["H_RH"]),
                float(current["L_m"]),
                float(current["piston_position_m"]),
            ]
            arrays["slow"][seq_index, timestep, :] = np.array(slow_values, dtype=np.float32)

            ultrasonic_result = simulate_waveform_measurement(
                x_h2=composition["x_H2"],
                x_ch4=composition["x_CH4"],
                x_co2=composition["x_CO2"],
                x_n2=composition["x_N2"],
                t_c=float(current["T_C"]),
                p_mpa=float(current["P_MPa"]),
                h_rh=float(current["H_RH"]),
                l_m=float(current["L_m"]),
                seed=rng.randrange(0, 2**32),
                spec=ultrasonic_spec,
            )
            fiber_result = simulate_fiber_mic_measurement(
                x_h2=composition["x_H2"],
                x_ch4=composition["x_CH4"],
                x_co2=composition["x_CO2"],
                x_n2=composition["x_N2"],
                t_c=float(current["T_C"]),
                p_mpa=float(current["P_MPa"]),
                h_rh=float(current["H_RH"]),
                l_m=float(current["L_m"]),
                seed=rng.randrange(0, 2**32),
                spec=fiber_mic_spec,
            )
            arrays["ultrasonic"][seq_index, timestep, :] = ultrasonic_result["waveform_int16"]
            arrays["ultrasonic_scale"][seq_index, timestep] = ultrasonic_result["scale_factor"]
            arrays["fiber_mic"][seq_index, timestep, :] = fiber_result["waveform_int16"]
            arrays["fiber_mic_scale"][seq_index, timestep] = fiber_result["scale_factor"]
            slow_rows.append(
                {
                    "sequence_id": condition["sequence_id"],
                    "timestep": str(timestep),
                    "timestamp_s": fmt(timestep * dt_s, 1),
                    "phase_id": phase_id,
                    "V_NDIR_CH4": fmt(float(current["V_NDIR_CH4"]), 6),
                    "V_NDIR_CO2": fmt(float(current["V_NDIR_CO2"]), 6),
                    "V_TCS": fmt(float(current["V_TCS"]), 6),
                    "T_C": fmt(float(current["T_C"]), 4),
                    "P_MPa": fmt(float(current["P_MPa"]), 5),
                    "H_RH": fmt(float(current["H_RH"]), 4),
                    "L_m": fmt(float(current["L_m"]), 5),
                    "piston_position_m": fmt(float(current["piston_position_m"]), 5),
                }
            )
    _flush_array(arrays["ultrasonic"])
    _flush_array(arrays["ultrasonic_scale"])
    _flush_array(arrays["fiber_mic"])
    _flush_array(arrays["fiber_mic_scale"])
    _flush_array(arrays["slow"])
    arrays["slow_rows"] = slow_rows
    return arrays


def _open_waveform_arrays(paths, sequence_count, timesteps, storage, ultrasonic_spec, fiber_mic_spec):
    ultrasonic_shape = (sequence_count, timesteps, ultrasonic_spec.waveform_samples)
    fiber_shape = (sequence_count, timesteps, fiber_mic_spec.waveform_samples)
    slow_shape = (sequence_count, timesteps, len(SLOW_CHANNELS))
    if storage in {"memmap", "both"}:
        paths["ultrasonic_int16"].parent.mkdir(parents=True, exist_ok=True)
        paths["slow_npy"].parent.mkdir(parents=True, exist_ok=True)
        return {
            "ultrasonic": np.lib.format.open_memmap(paths["ultrasonic_int16"], mode="w+", dtype=np.int16, shape=ultrasonic_shape),
            "ultrasonic_scale": np.lib.format.open_memmap(paths["ultrasonic_scale"], mode="w+", dtype=np.float32, shape=(sequence_count, timesteps)),
            "fiber_mic": np.lib.format.open_memmap(paths["fiber_mic_int16"], mode="w+", dtype=np.int16, shape=fiber_shape),
            "fiber_mic_scale": np.lib.format.open_memmap(paths["fiber_mic_scale"], mode="w+", dtype=np.float32, shape=(sequence_count, timesteps)),
            "slow": np.lib.format.open_memmap(paths["slow_npy"], mode="w+", dtype=np.float32, shape=slow_shape),
        }
    return {
        "ultrasonic": np.zeros(ultrasonic_shape, dtype=np.int16),
        "ultrasonic_scale": np.zeros((sequence_count, timesteps), dtype=np.float32),
        "fiber_mic": np.zeros(fiber_shape, dtype=np.int16),
        "fiber_mic_scale": np.zeros((sequence_count, timesteps), dtype=np.float32),
        "slow": np.zeros(slow_shape, dtype=np.float32),
    }


def _flush_array(array):
    if hasattr(array, "flush"):
        array.flush()


def _write_label_array(paths, labels, storage):
    if storage in {"memmap", "both"}:
        paths["y_npy"].parent.mkdir(parents=True, exist_ok=True)
        y_memmap = np.lib.format.open_memmap(paths["y_npy"], mode="w+", dtype=np.float32, shape=labels.shape)
        y_memmap[:] = labels
        y_memmap.flush()


def _write_waveform_metadata(paths, sequence_ids, ultrasonic_spec, fiber_mic_spec, timesteps, dt_s):
    paths["sequence_ids_npy"].parent.mkdir(parents=True, exist_ok=True)
    np.save(paths["sequence_ids_npy"], np.array(sequence_ids))
    np.save(paths["slow_channel_names_npy"], np.array(SLOW_CHANNELS))
    np.save(paths["label_names_npy"], np.array(FOUR_COMPONENT_LABEL_FIELDS))
    write_json(
        paths["waveform_v3_spec"],
        {
            "dataset_version": DATASET_VERSION,
            "channels": {
                "ultrasonic": ultrasonic_spec.to_dict(),
                "fiber_mic": fiber_mic_spec.to_dict(),
            },
            "slow_channels": list(SLOW_CHANNELS),
            "labels": list(FOUR_COMPONENT_LABEL_FIELDS),
            "sequences": len(sequence_ids),
            "timesteps": timesteps,
            "dt_s": dt_s,
        },
    )


def _phase_for_timestep_with_bounds(timestep, q1, q2, q3):
    if timestep < q1:
        return "baseline"
    if timestep < q2:
        return "exposure"
    if timestep < q3:
        return "steady"
    return "recovery"


def _phase_blend(timestep, q1, q2, q3):
    if timestep < q1:
        return 0.0
    if timestep < q2:
        return (timestep - q1 + 1) / max(q2 - q1, 1)
    if timestep < q3:
        return 1.0
    recovery_length = max(1, q3 - q2)
    return max(0.0, 1.0 - ((timestep - q3 + 1) / recovery_length))


def _blend_main_features(baseline_main, target_main, blend):
    out = {}
    for key in ("V_NDIR_CH4", "V_NDIR_CO2", "V_TCS"):
        base = float(baseline_main[key])
        target = float(target_main[key])
        out[key] = base + (target - base) * blend
    return out


def _blend_composition(condition, blend):
    target = {name: float(condition[name]) for name in FOUR_COMPONENT_LABEL_FIELDS}
    return {
        "x_H2": target["x_H2"] * blend,
        "x_CH4": target["x_CH4"] * blend,
        "x_CO2": target["x_CO2"] * blend,
        "x_N2": 100.0 + (target["x_N2"] - 100.0) * blend,
    }


def _waveform_quality_summary(sequence_ids, timesteps, split_distribution, split_warnings, storage, ultrasonic_spec, fiber_mic_spec):
    summary = build_common_summary(
        sequence_ids=sequence_ids,
        timesteps=timesteps,
        split_distribution=split_distribution,
        split_warnings=split_warnings,
        label_fields=FOUR_COMPONENT_LABEL_FIELDS,
        dataset_version=DATASET_VERSION_LABEL,
        simulation_level="dual_waveform_dynamic_simulation",
        channel_names=SLOW_CHANNELS,
        shape={
            "sequences": len(sequence_ids),
            "timesteps": timesteps,
            "ultrasonic_waveform_samples": ultrasonic_spec.waveform_samples,
            "fiber_mic_waveform_samples": fiber_mic_spec.waveform_samples,
            "slow_channels": len(SLOW_CHANNELS),
            "labels": len(FOUR_COMPONENT_LABEL_FIELDS),
        },
    )
    summary["acoustic_version"] = DATASET_VERSION
    summary["ultrasonic"] = {
        "dtype": "int16",
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "center_frequency_hz": CENTER_FREQUENCY_HZ,
        "measurement_window_s": ULTRASONIC_MEASUREMENT_WINDOW_S,
        "calibration_status": ULTRASONIC_CALIBRATION_STATUS,
    }
    summary["fiber_mic"] = {
        "dtype": "int16",
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "center_frequency_hz": CENTER_FREQUENCY_HZ,
        "measurement_window_s": fiber_mic_spec.measurement_window_s,
        "l_direct_factor": fiber_mic_spec.l_direct_factor,
        "wall_reflection_coef": fiber_mic_spec.wall_reflection_coef,
        "max_reflections": fiber_mic_spec.max_reflections,
        "calibration_status": FIBER_MIC_CALIBRATION_STATUS,
    }
    summary["slow_channel_names"] = SLOW_CHANNELS
    summary["calibration_status"] = "pending"
    summary["storage_format"] = storage
    summary["composition_timeline"] = "blended_baseline_to_target"
    summary["waveform_files"] = {
        "ultrasonic_int16": "sequences/ultrasonic_int16.npy",
        "ultrasonic_scale": "sequences/ultrasonic_scale.npy",
        "fiber_mic_int16": "sequences/fiber_mic_int16.npy",
        "fiber_mic_scale": "sequences/fiber_mic_scale.npy",
        "slow": "sequences/slow.npy",
        "y": "labels/y.npy",
    }
    summary["spec"] = {
        "ultrasonic": ultrasonic_spec.to_dict(),
        "fiber_mic": fiber_mic_spec.to_dict(),
    }
    return summary


def _waveform_readme(sequence_count, timesteps, split_distribution, storage):
    rows = [
        "| split | sequence_count | mixture_count |",
        "| --- | ---: | ---: |",
    ]
    for split_name in ("train", "val", "test"):
        stats = split_distribution[split_name]
        rows.append(f"| {split_name} | {stats['sequence_count']} | {stats['mixture_count']} |")
    split_table = "\n".join(rows)
    return f"""# V3.1 dual-channel waveform 数据包

```text
dataset_version: {DATASET_VERSION_LABEL}
calibration_status: pending
simulation_level: dual_waveform_dynamic_simulation
storage_format: {storage}
sequences: {sequence_count}
timesteps: {timesteps}
ultrasonic_waveform_samples: {ULTRASONIC_WAVEFORM_SAMPLES}
fiber_mic_waveform_samples: {FiberMicV3Spec().waveform_samples}
slow_channels: {len(SLOW_CHANNELS)}
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

{split_table}
"""


def main():
    parser = argparse.ArgumentParser(description="Generate V3.1 dual-channel waveform dataset.")
    parser.add_argument("--output-dir", default="data/waveform_v3")
    parser.add_argument("--sequence-count", type=int, default=10000)
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--seed", type=int, default=GENERATION_SEED)
    parser.add_argument("--storage", choices=sorted(VALID_STORAGE_FORMATS), default="memmap")
    args = parser.parse_args()
    generate_waveform_dataset(
        args.output_dir,
        sequence_count=args.sequence_count,
        timesteps=args.timesteps,
        seed=args.seed,
        storage=args.storage,
    )


if __name__ == "__main__":
    main()
