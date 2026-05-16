from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import nn


class AcousticWaveformEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 64):
        super().__init__()
        if embedding_dim != 64:
            raise ValueError("V3 waveform plan locks waveform embedding_dim to 64")
        self.embedding_dim = embedding_dim
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, stride=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=11, stride=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=7, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, waveform_int16: torch.Tensor, scale_factor: torch.Tensor) -> torch.Tensor:
        if waveform_int16.ndim != 2:
            raise ValueError(f"waveform_int16 must be 2D [batch, samples], got {tuple(waveform_int16.shape)}")
        if scale_factor.ndim != 1:
            raise ValueError(f"scale_factor must be 1D [batch], got {tuple(scale_factor.shape)}")
        if waveform_int16.shape[0] != scale_factor.shape[0]:
            raise ValueError("batch size mismatch between waveform_int16 and scale_factor")

        # 波形编码器强制 FP32：int16 转 float32 + scale_factor 乘法在 FP16 下易梯度 underflow
        with torch.amp.autocast(device_type=waveform_int16.device.type, enabled=False) if waveform_int16.device.type == "cuda" else nullcontext():
            waveform = waveform_int16.to(torch.float32) * scale_factor.unsqueeze(1)
            encoded = self.features(waveform.unsqueeze(1))
        return encoded.squeeze(-1)
