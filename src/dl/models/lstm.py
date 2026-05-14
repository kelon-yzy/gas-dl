from __future__ import annotations

import torch
from torch import nn

from models.config_utils import merge_model_kwargs


class LSTMRegressor(nn.Module):
    def __init__(self, config: dict | None = None, **kwargs):
        super().__init__()
        settings = merge_model_kwargs(
            {
                "input_size": 12,
                "hidden_size": 64,
                "num_layers": 1,
                "bidirectional": False,
                "dropout": 0.1,
                "out_dim": 3,
            },
            config,
            kwargs,
        )
        input_size = int(settings["input_size"])
        hidden_size = int(settings["hidden_size"])
        num_layers = int(settings["num_layers"])
        bidirectional = bool(settings["bidirectional"])
        dropout = float(settings["dropout"])
        out_dim = int(settings["out_dim"])
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
            bidirectional=bidirectional,
        )
        factor = 2 if bidirectional else 1
        self.head = nn.Sequential(
            nn.Linear(hidden_size * factor, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, out_dim),
        )

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        if self.lstm.bidirectional:
            last = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            last = hidden[-1]
        return self.head(last)
