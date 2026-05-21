from __future__ import annotations

import torch
from torch import nn


class CNN1DRegressor(nn.Module):
    input_format = "NCT"
    _VALID_TEMPORAL_POOLING = {"avg", "mean_max"}

    def __init__(
        self,
        in_channels: int = 12,
        out_dim: int = 3,
        hidden_channels=None,
        kernel_size: int = 5,
        dropout: float = 0.1,
        extra_dim: int = 0,
        temporal_pooling: str = "avg",
    ):
        super().__init__()
        if temporal_pooling not in self._VALID_TEMPORAL_POOLING:
            raise ValueError(f"Unknown temporal_pooling: {temporal_pooling!r}. Expected one of {sorted(self._VALID_TEMPORAL_POOLING)}")
        hidden_channels = hidden_channels or [32, 64, 64]
        layers = []
        current = in_channels
        for i, hidden in enumerate(hidden_channels):
            k = kernel_size if i < 2 else 3
            layers.extend(
                [
                    nn.Conv1d(current, hidden, kernel_size=k, padding=k // 2, bias=False),
                    nn.BatchNorm1d(hidden),
                    nn.ReLU(),
                ]
            )
            if i < 2:
                layers.append(nn.Dropout(dropout))
            current = hidden
        self.encoder = nn.Sequential(*layers)
        self.extra_dim = extra_dim
        self.temporal_pooling = temporal_pooling
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        pooled_dim = current if temporal_pooling == "avg" else current * 2
        self.head = nn.Sequential(
            nn.Linear(pooled_dim + extra_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, out_dim),
        )

    def forward(self, x, extra: torch.Tensor | None = None):
        encoded = self.encoder(x)
        if self.temporal_pooling == "avg":
            feats = self.avg_pool(encoded).flatten(1)
        else:
            avg = self.avg_pool(encoded).flatten(1)
            mx = self.max_pool(encoded).flatten(1)
            feats = torch.cat([avg, mx], dim=-1)
        if extra is not None:
            feats = torch.cat([feats, extra], dim=-1)
        return self.head(feats)
