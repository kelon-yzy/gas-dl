from __future__ import annotations

import torch
from torch import nn

from models.config_utils import merge_model_kwargs


class CNNLSTMRegressor(nn.Module):
    def __init__(self, config: dict | None = None, **kwargs):
        super().__init__()
        settings = merge_model_kwargs(
            {
                "in_channels": 12,
                "conv_channels": 64,
                "hidden_size": 64,
                "num_layers": 1,
                "kernel_size": 5,
                "dropout": 0.1,
                "out_dim": 3,
            },
            config,
            kwargs,
        )
        in_channels = int(settings["in_channels"])
        conv_channels = int(settings["conv_channels"])
        hidden_size = int(settings["hidden_size"])
        num_layers = int(settings["num_layers"])
        kernel_size = int(settings["kernel_size"])
        dropout = float(settings["dropout"])
        out_dim = int(settings["out_dim"])
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, conv_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
        )
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, out_dim),
        )

    def forward(self, x):
        features = self.conv(x).transpose(1, 2)
        _, (hidden, _) = self.lstm(features)
        return self.head(hidden[-1])
