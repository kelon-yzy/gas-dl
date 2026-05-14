from __future__ import annotations

import torch
from torch import nn


class CNN1DRegressor(nn.Module):
    def __init__(
        self,
        in_channels: int = 12,
        out_dim: int = 3,
        hidden_channels=None,
        kernel_size: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden_channels = hidden_channels or [32, 64, 64]
        layers = []
        current = in_channels
        for i, hidden in enumerate(hidden_channels):
            k = kernel_size if i < 2 else 3
            layers.extend(
                [
                    nn.Conv1d(current, hidden, kernel_size=k, padding=k // 2),
                    nn.BatchNorm1d(hidden),
                    nn.ReLU(),
                ]
            )
            if i < 2:
                layers.append(nn.Dropout(dropout))
            current = hidden
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(current, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, out_dim),
        )

    def forward(self, x):
        return self.head(self.encoder(x))

