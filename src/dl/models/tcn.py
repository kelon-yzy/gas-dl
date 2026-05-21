from __future__ import annotations

import torch
from torch import nn


class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, bias: bool = True):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation, bias=bias)

    def forward(self, x):
        out = self.conv(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(in_channels, out_channels, kernel_size, dilation, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            CausalConv1d(out_channels, out_channels, kernel_size, dilation, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        self.proj = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False) if in_channels != out_channels else nn.Identity()
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.net(x) + self.proj(x))


class TCNRegressor(nn.Module):
    input_format = "NCT"

    def __init__(
        self,
        in_channels: int = 12,
        out_dim: int = 3,
        channels=None,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        channels = channels or [32, 64, 64]
        layers = []
        current = in_channels
        for i, hidden in enumerate(channels):
            layers.append(TemporalBlock(current, hidden, kernel_size, dilation=2**i, dropout=dropout))
            current = hidden
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(current, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, out_dim),
        )

    def forward(self, x):
        return self.head(self.encoder(x))
