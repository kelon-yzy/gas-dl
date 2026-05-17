from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.channel_groups import EXPECTED_CHANNEL_NAMES, resolve_time_indices
from data.dataset_v2 import (
    V2SequenceDataset,
    load_acoustic_feature_array,
    load_sequence_metadata,
    load_v2_npz,
    resolve_dataset_channel_indices,
)
from data.dataset_waveform import WaveformSequenceDataset, load_waveform_package
from data.scaler_utils import load_or_fit_scaler
from data.split_utils import generate_group_splits, load_existing_splits, split_indices_from_frames


def resolve_path(value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _ensure_scaler_path(data_config: dict, output_dir: Path) -> None:
    if data_config.get("scaler_path"):
        return
    filename = "scaler_slow_sequence.json" if data_config.get("dataset_type") == "waveform_v3" else "scaler_sequence.json"
    data_config["scaler_path"] = str(output_dir / filename)


def _build_waveform_datasets(data_config: dict, seed: int):
    npz_path = resolve_path(data_config["npz_path"])
    index_path = resolve_path(data_config.get("index_path", "../simulation-data/output_sequence/sequence_index.csv"))
    split_dir = resolve_path(data_config.get("split_dir", "../simulation-data/output_sequence/splits"))
    scaler_path = resolve_path(data_config.get("scaler_path"))
    time_indices = resolve_time_indices(data_config.get("time_window", "all"))

    data = load_waveform_package(npz_path)
    slow = data["slow"].astype(np.float32)
    sequence_ids = [str(value) for value in data.get("sequence_ids", np.arange(len(slow)))]
    metadata = load_sequence_metadata(index_path, sequence_ids)

    splits = None
    if data_config.get("split_strategy", "existing_or_group_mixture") == "existing_or_group_mixture":
        splits = load_existing_splits(split_dir) if split_dir is not None else None
    if splits is None:
        splits = generate_group_splits(metadata, seed=seed)
    split_indices = split_indices_from_frames(splits, sequence_ids)

    slow_train = slow[split_indices["train"]]
    if time_indices is not None:
        slow_train = slow_train[:, time_indices, :]
    slow_scaler = load_or_fit_scaler(scaler_path, slow_train, channel_names=list(data["slow_channel_names"]))

    common = {
        "npz_path": npz_path,
        "slow_scaler": slow_scaler,
        "time_indices": time_indices,
        "index_path": index_path,
        "preloaded_data": data,
    }
    datasets = {name: WaveformSequenceDataset(indices=indices, **common) for name, indices in split_indices.items()}
    return datasets, splits


def _build_v2_datasets(data_config: dict, seed: int):
    npz_path = resolve_path(data_config["npz_path"])
    index_path = resolve_path(data_config.get("index_path", "../simulation-data/output_sequence/sequence_index.csv"))
    split_dir = resolve_path(data_config.get("split_dir", "../simulation-data/output_sequence/splits"))
    scaler_path = resolve_path(data_config.get("scaler_path"))
    input_format = data_config.get("input_format", "NTC")
    time_indices = resolve_time_indices(data_config.get("time_window", "all"))
    acoustic_feature_path = resolve_path(data_config.get("acoustic_feature_path"))
    acoustic_features = data_config.get("acoustic_features") or []

    data = load_v2_npz(npz_path)
    X = data["X"].astype(np.float32)
    sequence_ids = [str(value) for value in data.get("sequence_ids", np.arange(len(X)))]
    channel_names = list(EXPECTED_CHANNEL_NAMES)
    if acoustic_features:
        acoustic = load_acoustic_feature_array(acoustic_feature_path, sequence_ids, X.shape[1], list(acoustic_features))
        X = np.concatenate([X, acoustic], axis=2)
        channel_names.extend(acoustic_features)
    channel_indices = resolve_dataset_channel_indices(data_config.get("channels", "all"), channel_names)
    metadata = load_sequence_metadata(index_path, sequence_ids)

    splits = None
    if data_config.get("split_strategy", "existing_or_group_mixture") == "existing_or_group_mixture":
        splits = load_existing_splits(split_dir)
    if splits is None:
        splits = generate_group_splits(metadata, seed=seed)
    split_indices = split_indices_from_frames(splits, sequence_ids)

    X_train = X[split_indices["train"]]
    if time_indices is not None:
        X_train = X_train[:, time_indices, :]
    scaler = load_or_fit_scaler(scaler_path, X_train, channel_names=channel_names)
    dataset_scaler = scaler.subset(channel_indices)

    common = {
        "npz_path": npz_path,
        "scaler": dataset_scaler,
        "input_format": input_format,
        "channel_indices": channel_indices,
        "time_indices": time_indices,
        "index_path": index_path,
        "acoustic_feature_path": acoustic_feature_path,
        "acoustic_features": acoustic_features,
        "preloaded_data": data,
    }
    datasets = {name: V2SequenceDataset(indices=indices, **common) for name, indices in split_indices.items()}
    return datasets, splits


def build_datasets(config: dict):
    data_config = config["data"]
    dataset_type = data_config.get("dataset_type", "v2")
    seed = int(config["run"].get("seed", 42))
    if dataset_type == "waveform_v3":
        return _build_waveform_datasets(data_config, seed)
    return _build_v2_datasets(data_config, seed)


def _load_label_names(config: dict):
    data_config = config["data"]
    npz_path = resolve_path(data_config["npz_path"])
    dataset_type = data_config.get("dataset_type", "v2")
    if dataset_type == "waveform_v3":
        data_preview = load_waveform_package(npz_path)
    else:
        data_preview = load_v2_npz(npz_path)
    return [str(value) for value in data_preview["label_names"]]
