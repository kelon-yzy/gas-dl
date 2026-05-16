from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import nn

from models.acoustic_waveform_encoder import AcousticWaveformEncoder
from models.config_utils import merge_model_kwargs


class MultimodalFusionV3Regressor(nn.Module):
    def __init__(self, config: dict | None = None, **kwargs):
        super().__init__()
        settings = merge_model_kwargs(
            {
                "slow_dim": 8,
                "waveform_embedding_dim": 64,
                "hidden_size": 64,
                "num_layers": 1,
                "dropout": 0.1,
                "out_dim": 4,
                "use_waveform": True,
                "use_ultrasonic": True,
                "use_fiber_mic": True,
            },
            config,
            kwargs,
        )
        slow_dim = int(settings["slow_dim"])
        waveform_embedding_dim = int(settings["waveform_embedding_dim"])
        hidden_size = int(settings["hidden_size"])
        num_layers = int(settings["num_layers"])
        dropout = float(settings["dropout"])
        out_dim = int(settings["out_dim"])
        use_waveform = bool(settings["use_waveform"])
        use_ultrasonic = bool(settings["use_ultrasonic"])
        use_fiber_mic = bool(settings["use_fiber_mic"])
        self.use_waveform = use_waveform
        self.use_ultrasonic = use_waveform and use_ultrasonic
        self.use_fiber_mic = use_waveform and use_fiber_mic
        if waveform_embedding_dim != 64:
            raise ValueError("V3 waveform plan expects waveform embedding_dim = 64")
        if slow_dim != 8:
            raise ValueError("V3 waveform plan expects slow_dim = 8")
        if self.use_waveform and not (self.use_ultrasonic or self.use_fiber_mic):
            raise ValueError("At least one waveform branch must be enabled when use_waveform=True")

        self.ultrasonic_encoder = AcousticWaveformEncoder(embedding_dim=waveform_embedding_dim) if self.use_ultrasonic else None
        self.fiber_mic_encoder = AcousticWaveformEncoder(embedding_dim=waveform_embedding_dim) if self.use_fiber_mic else None
        fused_dim = slow_dim
        if self.use_ultrasonic:
            fused_dim += waveform_embedding_dim
        if self.use_fiber_mic:
            fused_dim += waveform_embedding_dim
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.sequence_backbone = nn.LSTM(
            input_size=fused_dim,
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

    def forward(
        self,
        ultrasonic_int16: torch.Tensor | None = None,
        ultrasonic_scale: torch.Tensor | None = None,
        fiber_mic_int16: torch.Tensor | None = None,
        fiber_mic_scale: torch.Tensor | None = None,
        slow: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if slow is None:
            raise ValueError("slow is required")
        if slow.ndim != 3:
            raise ValueError(f"slow must be 3D [batch, timesteps, channels], got {tuple(slow.shape)}")
        batch_size, timesteps, _ = slow.shape

        embeddings: list[torch.Tensor] = []
        if self.use_ultrasonic:
            if ultrasonic_int16 is None or ultrasonic_scale is None:
                raise ValueError("ultrasonic_int16 and ultrasonic_scale are required when use_ultrasonic=True")
            if ultrasonic_int16.shape[:2] != (batch_size, timesteps) or ultrasonic_scale.shape != (batch_size, timesteps):
                raise ValueError("ultrasonic tensors must match slow leading dimensions")
            flat_waveform = ultrasonic_int16.reshape(batch_size * timesteps, ultrasonic_int16.size(-1))
            flat_scale = ultrasonic_scale.reshape(batch_size * timesteps)
            embeddings.append(self.ultrasonic_encoder(flat_waveform, flat_scale).reshape(batch_size, timesteps, -1))

        if self.use_fiber_mic:
            if fiber_mic_int16 is None or fiber_mic_scale is None:
                raise ValueError("fiber_mic_int16 and fiber_mic_scale are required when use_fiber_mic=True")
            if fiber_mic_int16.shape[:2] != (batch_size, timesteps) or fiber_mic_scale.shape != (batch_size, timesteps):
                raise ValueError("fiber_mic tensors must match slow leading dimensions")
            flat_waveform = fiber_mic_int16.reshape(batch_size * timesteps, fiber_mic_int16.size(-1))
            flat_scale = fiber_mic_scale.reshape(batch_size * timesteps)
            embeddings.append(self.fiber_mic_encoder(flat_waveform, flat_scale).reshape(batch_size, timesteps, -1))

        fused = torch.cat([*embeddings, slow], dim=2) if embeddings else slow
        # LSTM 在 FP16 autocast 下数值不稳定，强制 FP32
        with torch.amp.autocast(device_type=slow.device.type, enabled=False) if slow.device.type == "cuda" else nullcontext():
            _, (hidden, _) = self.sequence_backbone(fused)
        last = hidden[-1]
        return self.head(last)
