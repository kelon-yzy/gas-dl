from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn

from models.config_utils import merge_model_kwargs


DEFAULT_CHANNEL_GROUPS: Dict[str, List[int]] = {
    "optical": [0, 1],
    "thermal": [2],
    "environment": [3, 4, 5, 6, 7],
    "acoustic": [8, 9, 10, 11],
}


class BranchEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, embedding_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels, embedding_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class BranchFusionRegressor(nn.Module):
    def __init__(self, config: dict | None = None, **kwargs):
        super().__init__()
        settings = merge_model_kwargs(
            {
                "hidden_channels": 32,
                "embedding_dim": 32,
                "dropout": 0.1,
                "out_dim": 3,
                "channel_groups": None,
            },
            config,
            kwargs,
        )
        hidden_channels = int(settings["hidden_channels"])
        embedding_dim = int(settings["embedding_dim"])
        dropout = float(settings["dropout"])
        out_dim = int(settings["out_dim"])
        channel_groups: Dict[str, List[int]] | None = settings["channel_groups"]
        groups = channel_groups or DEFAULT_CHANNEL_GROUPS
        self.group_indices = {name: torch.tensor(indices, dtype=torch.long) for name, indices in groups.items()}
        self.encoders = nn.ModuleDict(
            {
                name: BranchEncoder(len(indices), hidden_channels, embedding_dim, dropout)
                for name, indices in groups.items()
            }
        )
        self.head = nn.Sequential(
            nn.Linear(embedding_dim * len(groups), 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, out_dim),
        )

    def forward(self, x):
        parts = []
        for name, indices in self.group_indices.items():
            idx = indices.to(x.device)
            parts.append(self.encoders[name](x.index_select(1, idx)))
        return self.head(torch.cat(parts, dim=1))
