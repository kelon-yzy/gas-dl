from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data.channel_groups import (
    EXPECTED_CHANNEL_NAMES,
    EXPECTED_LABEL_NAMES,
    resolve_channel_indices,
    resolve_time_indices,
)


def _to_str_list(values) -> list[str]:
    return [str(v) for v in list(values)]


def _encode_feature(frame: pd.DataFrame, feature: str) -> pd.Series:
    if feature == "fit_status":
        return frame[feature].map({"ok": 1.0, "low_confidence": 0.5}).fillna(0.0).astype(np.float32)
    return pd.to_numeric(frame[feature], errors="raise").astype(np.float32)


def load_acoustic_feature_array(
    feature_path: str | Path,
    sequence_ids: list[str],
    timesteps: int,
    feature_names: list[str],
) -> np.ndarray:
    path = Path(feature_path)
    if not path.exists():
        raise FileNotFoundError(f"Acoustic feature file not found: {path}")
    frame = pd.read_csv(path)
    required = {"sequence_id", "timestep", *feature_names}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")

    base = pd.MultiIndex.from_product(
        [sequence_ids, list(range(timesteps))],
        names=["sequence_id", "timestep"],
    ).to_frame(index=False)
    selected = frame[["sequence_id", "timestep", *feature_names]].copy()
    selected["sequence_id"] = selected["sequence_id"].astype(str)
    merged = base.merge(selected, on=["sequence_id", "timestep"], how="left", validate="one_to_one")
    if merged[feature_names].isna().any().any():
        raise ValueError(f"Acoustic feature rows are incomplete for {path}")

    columns = [_encode_feature(merged, feature).to_numpy(dtype=np.float32) for feature in feature_names]
    return np.stack(columns, axis=1).reshape(len(sequence_ids), timesteps, len(feature_names))


def resolve_dataset_channel_indices(channel_indices, channel_names: list[str]):
    if channel_indices is None or channel_indices == "all":
        return None
    if isinstance(channel_indices, str):
        return resolve_channel_indices(channel_indices)
    values = list(channel_indices)
    if not values:
        return []
    if all(isinstance(value, str) for value in values):
        lookup = {name: index for index, name in enumerate(channel_names)}
        missing = [value for value in values if value not in lookup]
        if missing:
            raise ValueError(f"Unknown channel names: {missing}")
        return [lookup[value] for value in values]
    return resolve_channel_indices(values)


def load_v2_npz(npz_path: str | Path) -> dict:
    path = Path(npz_path)
    if not path.exists():
        raise FileNotFoundError(f"V2 npz file not found: {path}")
    with np.load(path, allow_pickle=True) as data:
        required = {"X", "y", "channel_names", "label_names"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"Missing keys in {path}: {sorted(missing)}")
        output = {key: data[key].copy() for key in data.files}
    validate_v2_arrays(output["X"], output["y"], output["channel_names"], output["label_names"])
    return output


def validate_v2_arrays(X, y, channel_names, label_names) -> None:
    if X.ndim != 3:
        raise ValueError(f"Expected X.ndim == 3, got {X.ndim}")
    if y.ndim != 2:
        raise ValueError(f"Expected y.ndim == 2, got {y.ndim}")
    if X.shape[1] < 4:
        raise ValueError(f"Expected X.shape[1] >= 4, got {X.shape[1]}")
    if X.shape[2] != 12:
        raise ValueError(f"Expected X.shape[2] == 12, got {X.shape[2]}")
    if y.shape[1] != len(EXPECTED_LABEL_NAMES):
        raise ValueError(f"Expected y.shape[1] == {len(EXPECTED_LABEL_NAMES)}, got {y.shape[1]}")
    actual_channels = _to_str_list(channel_names)
    actual_labels = _to_str_list(label_names)
    if actual_channels != EXPECTED_CHANNEL_NAMES:
        raise ValueError(f"Unexpected channel names: {actual_channels}")
    if actual_labels != EXPECTED_LABEL_NAMES:
        raise ValueError(f"Unexpected label names: {actual_labels}")


def load_sequence_metadata(index_path: str | Path | None, sequence_ids: Iterable[str]) -> pd.DataFrame:
    ids = pd.DataFrame({"sequence_id": list(sequence_ids)})
    if index_path is None:
        ids["mixture_id"] = ids["sequence_id"]
        return ids
    path = Path(index_path)
    if not path.exists():
        ids["mixture_id"] = ids["sequence_id"]
        return ids
    frame = pd.read_csv(path)
    if "sequence_id" not in frame.columns:
        raise ValueError(f"Missing sequence_id column in {path}")
    if "mixture_id" not in frame.columns:
        frame["mixture_id"] = frame["sequence_id"]
    merged = ids.merge(frame, on="sequence_id", how="left")
    merged["mixture_id"] = merged["mixture_id"].fillna(merged["sequence_id"])
    return merged


class V2SequenceDataset(Dataset):
    def __init__(
        self,
        npz_path: str | Path,
        indices,
        scaler=None,
        input_format: str = "NTC",
        channel_indices=None,
        time_indices=None,
        index_path: str | Path | None = None,
        acoustic_feature_path: str | Path | None = None,
        acoustic_features: list[str] | None = None,
    ):
        data = load_v2_npz(npz_path)
        self.X = data["X"].astype(np.float32)
        self.y = data["y"].astype(np.float32)
        self.sequence_ids = _to_str_list(data.get("sequence_ids", np.arange(len(self.X))))
        self.channel_names = list(EXPECTED_CHANNEL_NAMES)
        if acoustic_features:
            acoustic = load_acoustic_feature_array(
                acoustic_feature_path,
                self.sequence_ids,
                self.X.shape[1],
                list(acoustic_features),
            )
            self.X = np.concatenate([self.X, acoustic], axis=2)
            self.channel_names.extend(acoustic_features)
        self.metadata = load_sequence_metadata(index_path, self.sequence_ids)

        if isinstance(indices, pd.Series):
            indices = indices.tolist()
        if len(indices) > 0 and isinstance(indices[0], str):
            lookup = {sid: i for i, sid in enumerate(self.sequence_ids)}
            self.indices = np.array([lookup[sid] for sid in indices], dtype=np.int64)
        else:
            self.indices = np.array(indices, dtype=np.int64)

        self.input_format = input_format.upper()
        if self.input_format not in {"NTC", "NCT"}:
            raise ValueError("input_format must be NTC or NCT")
        self.channel_indices = resolve_dataset_channel_indices(channel_indices, self.channel_names)
        self.time_indices = resolve_time_indices(time_indices)
        self.scaler = scaler

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        source_idx = int(self.indices[idx])
        x = self.X[source_idx]
        y = self.y[source_idx]

        if self.time_indices is not None:
            x = x[self.time_indices, :]
        if self.channel_indices is not None:
            x = x[:, self.channel_indices]
        if self.scaler is not None:
            x = self.scaler.transform(x)
        if self.input_format == "NCT":
            x = np.transpose(x, (1, 0))

        row = self.metadata.iloc[source_idx].to_dict()
        row["sample_id"] = self.sequence_ids[source_idx]
        return torch.from_numpy(np.asarray(x, dtype=np.float32)), torch.from_numpy(y), row
