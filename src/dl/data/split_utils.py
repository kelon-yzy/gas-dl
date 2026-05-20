from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_existing_splits(split_dir: str | Path) -> dict[str, pd.DataFrame] | None:
    split_dir = Path(split_dir)
    files = {
        "train": split_dir / "train_sequence_ids.csv",
        "val": split_dir / "val_sequence_ids.csv",
        "test": split_dir / "test_sequence_ids.csv",
    }
    if not all(path.exists() for path in files.values()):
        return None
    splits = {name: pd.read_csv(path) for name, path in files.items()}
    extrapolation_path = split_dir / "extrapolation_sequence_ids.csv"
    if extrapolation_path.exists():
        splits["extrapolation"] = pd.read_csv(extrapolation_path)
    for name, frame in splits.items():
        source_path = files.get(name, extrapolation_path)
        if "sequence_id" not in frame.columns:
            raise ValueError(f"Missing sequence_id column in {source_path}")
        # 每条序列独立，不按 mixture_id 分组
        frame["mixture_id"] = frame["sequence_id"]
    validate_group_splits(splits)
    return splits


def generate_group_splits(
    metadata: pd.DataFrame,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    if abs(train_size + val_size + test_size - 1.0) > 1e-6:
        raise ValueError("train_size + val_size + test_size must equal 1")
    groups = metadata["sequence_id"].drop_duplicates().to_numpy()
    train_groups, temp_groups = train_test_split(
        groups, train_size=train_size, random_state=seed, shuffle=True
    )
    relative_val = val_size / (val_size + test_size)
    val_groups, test_groups = train_test_split(
        temp_groups, train_size=relative_val, random_state=seed, shuffle=True
    )
    mapping = {
        "train": set(train_groups),
        "val": set(val_groups),
        "test": set(test_groups),
    }
    splits = {}
    for split, group_set in mapping.items():
        matched = metadata.loc[
            metadata["sequence_id"].isin(group_set), ["sequence_id"]
        ].reset_index(drop=True)
        matched["mixture_id"] = matched["sequence_id"]
        splits[split] = matched
    validate_group_splits(splits)
    return splits


def validate_group_splits(splits: dict[str, pd.DataFrame]) -> None:
    group_sets = {
        name: set(frame["sequence_id"].astype(str).tolist()) for name, frame in splits.items()
    }
    for left in group_sets:
        for right in group_sets:
            if left >= right:
                continue
            overlap = group_sets[left].intersection(group_sets[right])
            if overlap:
                raise ValueError(f"sequence_id overlap between {left} and {right}: {sorted(overlap)[:5]}")


def save_splits(splits: dict[str, pd.DataFrame], split_dir: str | Path) -> None:
    split_dir = Path(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in splits.items():
        frame.to_csv(split_dir / f"{name}_sequence_ids.csv", index=False)


def split_indices_from_frames(splits: dict[str, pd.DataFrame], sequence_ids: list[str]) -> dict[str, list[int]]:
    lookup = {sid: i for i, sid in enumerate(sequence_ids)}
    output = {}
    for name, frame in splits.items():
        missing = [sid for sid in frame["sequence_id"].astype(str) if sid not in lookup]
        if missing:
            raise ValueError(f"{name} split contains unknown sequence_id: {missing[:5]}")
        output[name] = [lookup[sid] for sid in frame["sequence_id"].astype(str)]
    return output


def write_split_summary(splits: dict[str, pd.DataFrame], path: str | Path) -> None:
    summary = {
        name: {
            "n_sequences": int(len(frame)),
            "n_mixtures": int(frame["sequence_id"].nunique()),
        }
        for name, frame in splits.items()
    }
    Path(path).write_text(json.dumps(summary, indent=2), encoding="utf-8")
