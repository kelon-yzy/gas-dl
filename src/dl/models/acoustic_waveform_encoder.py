from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import nn


# int16 ADC 满量程常数：与 src/sim/scripts/acoustic_waveform_v3.py 的 ADC_MAX_INT16 对齐。
# 用作"波形形状"的归一化分母——固定常量、与样本无关，BN 不会因为它而被绕过。
_ADC_MAX_INT16: float = 32767.0


class AcousticWaveformEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        if embedding_dim != 64:
            raise ValueError("V3 waveform plan locks waveform embedding_dim to 64")
        self.embedding_dim = embedding_dim

        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=11, stride=2, padding=5, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        # projection 输入 = [avg(64) || max(64) || log(scale_factor) 标量] = 129 维。
        # 把绝对幅值信号以独立标量送进线性层，避免被任何 per-sample 归一化抹掉。
        self.projection = nn.Sequential(
            nn.Linear(64 * 2 + 1, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, waveform_int16: torch.Tensor, scale_factor: torch.Tensor) -> torch.Tensor:
        if waveform_int16.ndim != 2:
            raise ValueError(f"waveform_int16 must be 2D [batch, samples], got {tuple(waveform_int16.shape)}")
        if scale_factor.ndim != 1:
            raise ValueError(f"scale_factor must be 1D [batch], got {tuple(scale_factor.shape)}")
        if waveform_int16.shape[0] != scale_factor.shape[0]:
            raise ValueError("batch size mismatch between waveform_int16 and scale_factor")

        # 波形编码器强制 FP32：int16 转 float32、除常量、对数运算在 FP16 下数值范围易越界。
        with torch.amp.autocast(device_type=waveform_int16.device.type, enabled=False) if waveform_int16.device.type == "cuda" else nullcontext():
            # 方案 B：用固定常数 ADC 满量程归一化，把波形形状压到 [-1, 1]；
            # 不再 * scale_factor、不再 per-sample zscore，绝对幅值改走 log_scale 标量旁路。
            waveform = waveform_int16.to(torch.float32) / _ADC_MAX_INT16
            x = self.features(waveform.unsqueeze(1))
            avg = self.avg_pool(x).squeeze(-1)
            mx = self.max_pool(x).squeeze(-1)
            # log(scale_factor) 携带"本帧物理电压幅值"信号；clamp_min 防御退化输入（sim 已保证正数）。
            log_scale = torch.log(scale_factor.to(torch.float32).clamp_min(1e-12)).unsqueeze(-1)
            x = torch.cat([avg, mx, log_scale], dim=1)
            x = self.projection(x)
        return x
