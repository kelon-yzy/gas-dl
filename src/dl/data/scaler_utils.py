from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from data.channel_groups import EXPECTED_CHANNEL_NAMES


class ChannelStandardScaler:
    def __init__(self, mean, std, channel_indices=None, channel_names=None):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.std = np.where(self.std < 1e-12, 1.0, self.std)
        self.channel_indices = channel_indices
        self.channel_names = channel_names or EXPECTED_CHANNEL_NAMES

    @classmethod
    def fit(cls, X: np.ndarray, channel_names=None):
        if X.ndim != 3:
            raise ValueError(f"Expected 3D X, got shape {X.shape}")
        mean = X.mean(axis=(0, 1))
        std = X.std(axis=(0, 1))
        return cls(mean=mean, std=std, channel_names=channel_names)

    def subset(self, channel_indices):
        if channel_indices is None:
            return self
        return ChannelStandardScaler(
            mean=self.mean[channel_indices],
            std=self.std[channel_indices],
            channel_indices=channel_indices,
            channel_names=[self.channel_names[i] for i in channel_indices],
        )

    def transform(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 2:
            return (x - self.mean.reshape(1, -1)) / self.std.reshape(1, -1)
        if x.ndim == 3:
            return (x - self.mean.reshape(1, 1, -1)) / self.std.reshape(1, 1, -1)
        raise ValueError(f"Expected 2D or 3D input, got shape {x.shape}")

    def to_dict(self):
        return {
            "method": "z_score",
            "fit_scope": "train_split_only",
            "transform_target": "X",
            "channel_axis": 2,
            "channel_names": list(self.channel_names),
            "mean": self.mean.astype(float).tolist(),
            "std": self.std.astype(float).tolist(),
        }


def load_scaler(path: str | Path) -> ChannelStandardScaler:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ChannelStandardScaler(
        mean=payload["mean"],
        std=payload["std"],
        channel_names=payload.get("channel_names", EXPECTED_CHANNEL_NAMES),
    )


def save_scaler(scaler: ChannelStandardScaler, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scaler.to_dict(), indent=2), encoding="utf-8")


def load_or_fit_scaler(path: str | Path | None, X_train: np.ndarray, channel_names=None) -> ChannelStandardScaler:
    if path is not None and Path(path).exists():
        return load_scaler(path)
    scaler = ChannelStandardScaler.fit(X_train, channel_names=channel_names)
    if path is not None:
        save_scaler(scaler, path)
    return scaler

