from __future__ import annotations

import math

import torch
from torch import nn

from models.config_utils import merge_model_kwargs


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerRegressor(nn.Module):
    input_format = "NTC"

    def __init__(self, config: dict | None = None, **kwargs):
        super().__init__()
        settings = merge_model_kwargs(
            {
                "input_dim": 12,
                "d_model": 64,
                "nhead": 4,
                "num_layers": 2,
                "dim_feedforward": 128,
                "dropout": 0.1,
                "out_dim": 3,
            },
            config,
            kwargs,
        )
        input_dim = int(settings["input_dim"])
        d_model = int(settings["d_model"])
        nhead = int(settings["nhead"])
        num_layers = int(settings["num_layers"])
        dim_feedforward = int(settings["dim_feedforward"])
        dropout = float(settings["dropout"])
        out_dim = int(settings["out_dim"])
        if num_layers > 4 or d_model > 128 or nhead > 8:
            raise ValueError("Transformer config exceeds the V2 plan limits")
        self.proj = nn.Linear(input_dim, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, out_dim),
        )

    def forward(self, x):
        encoded = self.encoder(self.pos(self.proj(x)))
        return self.head(encoded.mean(dim=1))
