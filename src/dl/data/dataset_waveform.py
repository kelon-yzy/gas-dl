from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data.channel_groups import resolve_time_indices
from data.dataset_v2 import _to_str_list, load_sequence_metadata

SIM_ROOT = Path(__file__).resolve().parents[2] / "sim"
if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))

from sim_common.phases import phase_boundaries


EXPECTED_SLOW_CHANNEL_NAMES = [
    "V_NDIR_CH4",
    "V_NDIR_CO2",
    "V_TCS",
    "T_C",
    "P_MPa",
    "H_RH",
    "L_m",
    "piston_position_m",
]
EXPECTED_WAVEFORM_LABEL_NAMES = ["x_H2", "x_CH4", "x_CO2", "x_N2"]
DEFAULT_ULTRASONIC_SAMPLES = 1000
DEFAULT_FIBER_MIC_SAMPLES = 2000
DEFAULT_STAGE_DIM = 4


def _to_tensor_ready_array(array: np.ndarray, dtype) -> np.ndarray:
    if array.dtype != dtype or not array.flags.c_contiguous or not array.flags.writeable:
        return np.array(array, dtype=dtype, copy=True)
    return array


def _build_stage_one_hot(total_timesteps: int, time_indices: list[int] | None, stage_dim: int = DEFAULT_STAGE_DIM) -> np.ndarray:
    if stage_dim != DEFAULT_STAGE_DIM:
        raise ValueError(f"Expected stage_dim == {DEFAULT_STAGE_DIM}, got {stage_dim}")
    q1, q2, q3 = phase_boundaries(total_timesteps)
    positions = np.arange(total_timesteps, dtype=np.int64) if time_indices is None else np.asarray(time_indices, dtype=np.int64)
    stage_ids = np.zeros(positions.shape[0], dtype=np.int64)
    stage_ids[positions >= q1] = 1
    stage_ids[positions >= q2] = 2
    stage_ids[positions >= q3] = 3
    return np.eye(stage_dim, dtype=np.float32)[stage_ids]


def _load_waveform_sequence_ids(path: str | Path) -> list[str]:
    path = Path(path)
    if path.is_file():
        with np.load(path, allow_pickle=True) as data:
            if "sequence_ids" in data.files:
                return _to_str_list(data["sequence_ids"])
            if "slow" in data.files:
                return _to_str_list(np.arange(data["slow"].shape[0]))
        raise ValueError(f"sequence_ids and slow are both missing in {path}")
    sequence_ids_path = path / "metadata" / "sequence_ids.npy"
    if sequence_ids_path.exists():
        return _to_str_list(np.load(sequence_ids_path, allow_pickle=True))
    data = load_waveform_package(path)
    return _to_str_list(data.get("sequence_ids", np.arange(len(data["ultrasonic"]))))


def load_waveform_npz(npz_path: str | Path) -> dict:
    path = Path(npz_path)
    if not path.exists():
        raise FileNotFoundError(f"Waveform npz file not found: {path}")
    with np.load(path, allow_pickle=True) as data:
        required = {"ultrasonic", "ultrasonic_scale", "fiber_mic", "fiber_mic_scale", "slow", "y", "slow_channel_names", "label_names"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"Missing keys in {path}: {sorted(missing)}")
        output = {key: data[key].copy() for key in data.files}
    validate_waveform_arrays(
        output["ultrasonic"],
        output["ultrasonic_scale"],
        output["fiber_mic"],
        output["fiber_mic_scale"],
        output["slow"],
        output["y"],
        output["slow_channel_names"],
        output["label_names"],
        ultrasonic_samples=output["ultrasonic"].shape[2],
        fiber_mic_samples=output["fiber_mic"].shape[2],
    )
    output["ultrasonic_samples"] = int(output["ultrasonic"].shape[2])
    output["fiber_mic_samples"] = int(output["fiber_mic"].shape[2])
    return output


def load_waveform_package(path: str | Path) -> dict:
    path = Path(path)
    if path.is_file():
        return load_waveform_npz(path)
    if not path.exists():
        raise FileNotFoundError(f"Waveform package not found: {path}")

    spec_path = path / "metadata" / "waveform_v3_spec.json"
    if not spec_path.exists():
        raise FileNotFoundError(f"Waveform spec file not found: {spec_path}")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    ultrasonic_spec = spec["channels"]["ultrasonic"]
    fiber_spec = spec["channels"]["fiber_mic"]
    output = {
        "ultrasonic": np.load(path / "sequences" / "ultrasonic_int16.npy", mmap_mode="r"),
        "ultrasonic_scale": np.load(path / "sequences" / "ultrasonic_scale.npy", mmap_mode="r"),
        "fiber_mic": np.load(path / "sequences" / "fiber_mic_int16.npy", mmap_mode="r"),
        "fiber_mic_scale": np.load(path / "sequences" / "fiber_mic_scale.npy", mmap_mode="r"),
        "slow": np.load(path / "sequences" / "slow.npy", mmap_mode="r"),
        "y": np.load(path / "labels" / "y.npy", mmap_mode="r"),
        "sequence_ids": np.load(path / "metadata" / "sequence_ids.npy", allow_pickle=True),
        "slow_channel_names": np.load(path / "metadata" / "slow_channel_names.npy", allow_pickle=True),
        "label_names": np.load(path / "metadata" / "label_names.npy", allow_pickle=True),
        "spec": spec,
        "ultrasonic_samples": int(ultrasonic_spec.get("waveform_samples", DEFAULT_ULTRASONIC_SAMPLES)),
        "fiber_mic_samples": int(fiber_spec.get("waveform_samples", DEFAULT_FIBER_MIC_SAMPLES)),
    }
    validate_waveform_arrays(
        output["ultrasonic"],
        output["ultrasonic_scale"],
        output["fiber_mic"],
        output["fiber_mic_scale"],
        output["slow"],
        output["y"],
        output["slow_channel_names"],
        output["label_names"],
        ultrasonic_samples=output["ultrasonic_samples"],
        fiber_mic_samples=output["fiber_mic_samples"],
    )
    return output


def validate_waveform_arrays(
    ultrasonic,
    ultrasonic_scale,
    fiber_mic,
    fiber_mic_scale,
    slow,
    y,
    slow_channel_names,
    label_names,
    ultrasonic_samples=DEFAULT_ULTRASONIC_SAMPLES,
    fiber_mic_samples=DEFAULT_FIBER_MIC_SAMPLES,
) -> None:
    if ultrasonic.ndim != 3:
        raise ValueError(f"Expected ultrasonic.ndim == 3, got {ultrasonic.ndim}")
    if ultrasonic_scale.ndim != 2:
        raise ValueError(f"Expected ultrasonic_scale.ndim == 2, got {ultrasonic_scale.ndim}")
    if fiber_mic.ndim != 3:
        raise ValueError(f"Expected fiber_mic.ndim == 3, got {fiber_mic.ndim}")
    if fiber_mic_scale.ndim != 2:
        raise ValueError(f"Expected fiber_mic_scale.ndim == 2, got {fiber_mic_scale.ndim}")
    if slow.ndim != 3:
        raise ValueError(f"Expected slow.ndim == 3, got {slow.ndim}")
    if y.ndim != 2:
        raise ValueError(f"Expected y.ndim == 2, got {y.ndim}")
    if ultrasonic.shape[:2] != ultrasonic_scale.shape:
        raise ValueError("ultrasonic and ultrasonic_scale leading dimensions must match")
    if fiber_mic.shape[:2] != fiber_mic_scale.shape:
        raise ValueError("fiber_mic and fiber_mic_scale leading dimensions must match")
    if ultrasonic.shape[:2] != slow.shape[:2] or fiber_mic.shape[:2] != slow.shape[:2]:
        raise ValueError("waveform and slow leading dimensions must match")
    if ultrasonic.shape[2] != ultrasonic_samples:
        raise ValueError(f"Expected ultrasonic.shape[2] == {ultrasonic_samples}, got {ultrasonic.shape[2]}")
    if fiber_mic.shape[2] != fiber_mic_samples:
        raise ValueError(f"Expected fiber_mic.shape[2] == {fiber_mic_samples}, got {fiber_mic.shape[2]}")
    if slow.shape[2] != 8:
        raise ValueError(f"Expected slow.shape[2] == 8, got {slow.shape[2]}")
    if y.shape[1] != 4:
        raise ValueError(f"Expected y.shape[1] == 4, got {y.shape[1]}")
    if _to_str_list(slow_channel_names) != EXPECTED_SLOW_CHANNEL_NAMES:
        raise ValueError(f"Unexpected slow_channel_names: {_to_str_list(slow_channel_names)}")
    if _to_str_list(label_names) != EXPECTED_WAVEFORM_LABEL_NAMES:
        raise ValueError(f"Unexpected label_names: {_to_str_list(label_names)}")


class WaveformSequenceDataset(Dataset):
    def __init__(
        self,
        npz_path: str | Path,
        indices,
        slow_scaler=None,
        time_indices=None,
        index_path: str | Path | None = None,
        preloaded_data: dict | None = None,
    ):
        self.npz_path = Path(npz_path)
        self.index_path = index_path
        self._preloaded_data = preloaded_data
        self.ultrasonic = None
        self.ultrasonic_scale = None
        self.fiber_mic = None
        self.fiber_mic_scale = None
        self.slow = None
        self.y = None
        self.metadata = None
        self.total_timesteps = None
        self.stage_dim = DEFAULT_STAGE_DIM
        if preloaded_data is not None:
            validate_waveform_arrays(
                preloaded_data["ultrasonic"],
                preloaded_data["ultrasonic_scale"],
                preloaded_data["fiber_mic"],
                preloaded_data["fiber_mic_scale"],
                preloaded_data["slow"],
                preloaded_data["y"],
                preloaded_data["slow_channel_names"],
                preloaded_data["label_names"],
                ultrasonic_samples=preloaded_data["ultrasonic"].shape[2],
                fiber_mic_samples=preloaded_data["fiber_mic"].shape[2],
            )
            self.sequence_ids = _to_str_list(preloaded_data.get("sequence_ids", np.arange(len(preloaded_data["slow"]))))
        else:
            self.sequence_ids = _load_waveform_sequence_ids(self.npz_path)
        self.time_indices = resolve_time_indices(time_indices)
        self.slow_scaler = slow_scaler

        if isinstance(indices, pd.Series):
            indices = indices.tolist()
        if len(indices) > 0 and isinstance(indices[0], str):
            lookup = {sid: i for i, sid in enumerate(self.sequence_ids)}
            self.indices = np.array([lookup[sid] for sid in indices], dtype=np.int64)
        else:
            self.indices = np.array(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def _ensure_loaded(self) -> None:
        if self.ultrasonic is not None:
            return
        data = self._preloaded_data if self._preloaded_data is not None else load_waveform_package(self.npz_path)
        self.ultrasonic = data["ultrasonic"].astype(np.int16, copy=False)
        self.ultrasonic_scale = data["ultrasonic_scale"].astype(np.float32, copy=False)
        self.fiber_mic = data["fiber_mic"].astype(np.int16, copy=False)
        self.fiber_mic_scale = data["fiber_mic_scale"].astype(np.float32, copy=False)
        self.slow = data["slow"].astype(np.float32, copy=False)
        self.y = data["y"].astype(np.float32, copy=False)
        self.total_timesteps = int(self.slow.shape[1])
        self.metadata = load_sequence_metadata(self.index_path, self.sequence_ids)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["ultrasonic"] = None
        state["ultrasonic_scale"] = None
        state["fiber_mic"] = None
        state["fiber_mic_scale"] = None
        state["slow"] = None
        state["y"] = None
        state["metadata"] = None
        state["_preloaded_data"] = None
        return state

    def __getitem__(self, idx):
        self._ensure_loaded()
        source_idx = int(self.indices[idx])
        ultrasonic = self.ultrasonic[source_idx]
        ultrasonic_scale = self.ultrasonic_scale[source_idx]
        fiber_mic = self.fiber_mic[source_idx]
        fiber_mic_scale = self.fiber_mic_scale[source_idx]
        slow = self.slow[source_idx]
        target = self.y[source_idx]
        stage_one_hot = _build_stage_one_hot(int(self.total_timesteps), self.time_indices, stage_dim=self.stage_dim)

        if self.time_indices is not None:
            ultrasonic = ultrasonic[self.time_indices, :]
            ultrasonic_scale = ultrasonic_scale[self.time_indices]
            fiber_mic = fiber_mic[self.time_indices, :]
            fiber_mic_scale = fiber_mic_scale[self.time_indices]
            slow = slow[self.time_indices, :]
        if self.slow_scaler is not None:
            slow = self.slow_scaler.transform(slow)

        # memmap / 只读视图在 worker 内按样本转成可写连续数组，避免整包复制 7GB+ 波形数据
        ultrasonic = _to_tensor_ready_array(ultrasonic, np.int16)
        ultrasonic_scale = _to_tensor_ready_array(ultrasonic_scale, np.float32)
        fiber_mic = _to_tensor_ready_array(fiber_mic, np.int16)
        fiber_mic_scale = _to_tensor_ready_array(fiber_mic_scale, np.float32)
        slow = _to_tensor_ready_array(slow, np.float32)
        target = _to_tensor_ready_array(target, np.float32)
        stage_one_hot = _to_tensor_ready_array(stage_one_hot, np.float32)

        meta = self.metadata.iloc[source_idx].to_dict()
        meta["sample_id"] = self.sequence_ids[source_idx]
        return {
            "ultrasonic": torch.from_numpy(ultrasonic),
            "ultrasonic_scale": torch.from_numpy(ultrasonic_scale),
            "fiber_mic": torch.from_numpy(fiber_mic),
            "fiber_mic_scale": torch.from_numpy(fiber_mic_scale),
            "slow": torch.from_numpy(slow),
            "stage_one_hot": torch.from_numpy(stage_one_hot),
            "target": torch.from_numpy(target),
            "meta": meta,
        }
