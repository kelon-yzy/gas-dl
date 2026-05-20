"""1DCNN 声学编码 + TCN 时序融合的多模态回归模型。

设计目标：
    1. 用专用 1DCNN 把超声 / 光纤麦克风波形分别压成 (B, T, D) 高维嵌入。
    2. 与 (B, T, slow_dim) 慢变量沿通道拼接成 (B, T, 2D + slow_dim)。
    3. 转 NCT 送入 TCN backbone，输出 (B, C_out, T)。
    4. 末位 + 均值 + 最大三池化拼接 → MLP → (B, out_dim)。

前向签名与 MultimodalFusionV3Regressor 对齐，runtime 通过类属性
`use_waveform = True` 自动识别波形批分支。
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Sequence

import torch
from torch import nn

from models.config_utils import merge_model_kwargs


# int16 ADC 满量程常数：与 src/dl/models/acoustic_waveform_encoder.py 对齐，
# 用作"波形形状"的归一化分母——固定常量、与样本无关，BN 不会被它绕过。
_ADC_MAX_INT16: float = 32767.0


def _maybe_disable_autocast(device: torch.device):
    """波形数值跨度大，FP16 下 int16→fp 转换 + 对数运算易 underflow，编码阶段强制 FP32。"""
    if device.type == "cuda":
        return torch.amp.autocast(device_type=device.type, enabled=False)
    return nullcontext()


class DeepAcousticEncoder1D(nn.Module):
    """声学波形 → 高维嵌入的 1DCNN 编码器。

    输入：
        waveform_int16: (N, L) int16，N 通常为 batch × timesteps 展平
        scale_factor:   (N,) float32
    输出：
        embedding: (N, embedding_dim) float32

    归一化策略与 AcousticWaveformEncoder 主线一致：卷积分支只消费
    `waveform_int16 / 32767`，绝对幅值通过 log_scale 标量旁路引入。
    与 AcousticWaveformEncoder 的区别在于卷积栈本身：层数与通道宽度
    可配置（默认 4 层 16→32→64→64，kernel=7），便于把"高维提取"
    在配置层面做厚一些。
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        channels: Sequence[int] = (16, 32, 64, 64),
        kernel_size: int = 7,
        dropout: float = 0.1,
    ):
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim 必须为正整数，得到 {embedding_dim}")
        channels = tuple(int(c) for c in channels)
        if len(channels) < 2:
            raise ValueError("acoustic_channels 至少需要 2 层")
        if any(c <= 0 for c in channels):
            raise ValueError(f"acoustic_channels 必须为正整数序列，得到 {channels}")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size 必须为正奇数，得到 {kernel_size}")
        self.embedding_dim = embedding_dim

        layers: list[nn.Module] = []
        current = 1
        for i, c in enumerate(channels):
            # 前 N-1 层下采样 stride=2，最后一层 stride=1，保留中间分辨率给末端池化
            stride = 2 if i < len(channels) - 1 else 1
            layers.extend(
                [
                    nn.Conv1d(current, c, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2),
                    nn.BatchNorm1d(c),
                    nn.ReLU(),
                ]
            )
            current = c
        self.features = nn.Sequential(*layers)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.projection = nn.Sequential(
            nn.Linear(current * 2 + 1, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, waveform_int16: torch.Tensor, scale_factor: torch.Tensor) -> torch.Tensor:
        if waveform_int16.ndim != 2:
            raise ValueError(f"waveform_int16 必须为 2D [N, L]，得到 {tuple(waveform_int16.shape)}")
        if scale_factor.ndim != 1 or scale_factor.shape[0] != waveform_int16.shape[0]:
            raise ValueError("scale_factor 必须为 1D 且与 waveform_int16 的 batch 维一致")
        if torch.any(scale_factor <= 0).item():
            raise ValueError("scale_factor 必须全部为正数")
        with _maybe_disable_autocast(waveform_int16.device):
            # 方案 B：用固定常数 ADC 满量程归一化，把波形形状压到 [-1, 1]；
            # 卷积分支不再消费物理电压幅值，绝对幅值通过 log_scale 标量旁路保留。
            # 与 AcousticWaveformEncoder 主线策略一致，避免输入语义分裂。
            waveform = waveform_int16.to(torch.float32) / _ADC_MAX_INT16
            log_scale = torch.log(scale_factor.to(torch.float32)).unsqueeze(-1)
            feat = self.features(waveform.unsqueeze(1))
            avg = self.avg_pool(feat).squeeze(-1)
            mx = self.max_pool(feat).squeeze(-1)
            return self.projection(torch.cat([avg, mx, log_scale], dim=1))


class _CausalConv1d(nn.Module):
    """因果一维卷积：右侧 padding 后截断，避免未来信息泄漏。

    与 src/dl/models/tcn.py 同构，本模块在此独立维护以避免 backbone 路径耦合。
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.padding > 0:
            out = out[:, :, : -self.padding]
        return out


class _TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            _CausalConv1d(in_channels, out_channels, kernel_size, dilation),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            _CausalConv1d(out_channels, out_channels, kernel_size, dilation),
            nn.BatchNorm1d(out_channels),
        )
        self.proj = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.net(x) + self.proj(x))


class CNN1DTCNFusionRegressor(nn.Module):
    """1DCNN 声学编码 → 通道拼接 → TCN 时序融合 → 三池化回归头。

    前向签名：
        model(ultrasonic, ultrasonic_scale, fiber_mic, fiber_mic_scale, slow)

    输入张量形状：
        ultrasonic        (B, T, L_u) int16
        ultrasonic_scale  (B, T)      float32
        fiber_mic         (B, T, L_f) int16
        fiber_mic_scale   (B, T)      float32
        slow              (B, T, slow_dim) float32
    输出：
        (B, out_dim) float32
    """

    use_waveform: bool = True
    input_format: str = "NCT"

    def __init__(self, config: dict | None = None, **kwargs):
        super().__init__()
        settings = merge_model_kwargs(
            {
                "slow_dim": 8,
                "waveform_embedding_dim": 64,
                "acoustic_channels": [16, 32, 64, 64],
                "acoustic_kernel_size": 7,
                "acoustic_dropout": 0.1,
                "tcn_channels": [64, 64, 64],
                "tcn_kernel_size": 3,
                "tcn_dropout": 0.2,
                "head_dropout": 0.2,
                "out_dim": 4,
                "use_ultrasonic": True,
                "use_fiber_mic": True,
            },
            config,
            kwargs,
        )
        slow_dim = int(settings["slow_dim"])
        embedding_dim = int(settings["waveform_embedding_dim"])
        acoustic_channels = list(settings["acoustic_channels"])
        acoustic_kernel_size = int(settings["acoustic_kernel_size"])
        acoustic_dropout = float(settings["acoustic_dropout"])
        tcn_channels = list(settings["tcn_channels"])
        tcn_kernel_size = int(settings["tcn_kernel_size"])
        tcn_dropout = float(settings["tcn_dropout"])
        head_dropout = float(settings["head_dropout"])
        out_dim = int(settings["out_dim"])
        use_ultrasonic = bool(settings["use_ultrasonic"])
        use_fiber_mic = bool(settings["use_fiber_mic"])

        if slow_dim <= 0:
            raise ValueError(f"slow_dim 必须为正整数，得到 {slow_dim}")
        if not (use_ultrasonic or use_fiber_mic):
            raise ValueError("至少需要开启一个波形分支 (use_ultrasonic 或 use_fiber_mic)")
        if not tcn_channels:
            raise ValueError("tcn_channels 不能为空")

        self.slow_dim = slow_dim
        self.use_ultrasonic = use_ultrasonic
        self.use_fiber_mic = use_fiber_mic
        self.embedding_dim = embedding_dim

        self.ultrasonic_encoder = (
            DeepAcousticEncoder1D(embedding_dim, acoustic_channels, acoustic_kernel_size, acoustic_dropout)
            if use_ultrasonic
            else None
        )
        self.fiber_mic_encoder = (
            DeepAcousticEncoder1D(embedding_dim, acoustic_channels, acoustic_kernel_size, acoustic_dropout)
            if use_fiber_mic
            else None
        )

        enabled_branches = int(use_ultrasonic) + int(use_fiber_mic)
        fused_channels = slow_dim + embedding_dim * enabled_branches

        tcn_layers: list[nn.Module] = []
        current = fused_channels
        for i, ch in enumerate(tcn_channels):
            tcn_layers.append(
                _TemporalBlock(
                    in_channels=current,
                    out_channels=int(ch),
                    kernel_size=tcn_kernel_size,
                    dilation=2 ** i,
                    dropout=tcn_dropout,
                )
            )
            current = int(ch)
        self.tcn = nn.Sequential(*tcn_layers)
        self.tcn_out_channels = current

        # 末位 + 均值 + 最大三池化拼接 → MLP
        self.head = nn.Sequential(
            nn.Linear(current * 3, 128),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(64, out_dim),
        )

    def _encode_waveform_branch(
        self,
        encoder: DeepAcousticEncoder1D,
        waveform: torch.Tensor,
        scale: torch.Tensor,
        batch: int,
        timesteps: int,
    ) -> torch.Tensor:
        if waveform.shape[:2] != (batch, timesteps) or scale.shape != (batch, timesteps):
            raise ValueError("波形与 scale 的前两维必须匹配 slow 的 (B, T)")
        flat_wave = waveform.reshape(batch * timesteps, waveform.size(-1))
        flat_scale = scale.reshape(batch * timesteps)
        return encoder(flat_wave, flat_scale).reshape(batch, timesteps, -1)

    def forward(
        self,
        ultrasonic: torch.Tensor | None,
        ultrasonic_scale: torch.Tensor | None,
        fiber_mic: torch.Tensor | None,
        fiber_mic_scale: torch.Tensor | None,
        slow: torch.Tensor,
    ) -> torch.Tensor:
        if slow.ndim != 3:
            raise ValueError(f"slow 必须为 3D [B, T, C]，实际 {tuple(slow.shape)}")
        if slow.size(-1) != self.slow_dim:
            raise ValueError(f"slow 通道数应为 {self.slow_dim}，实际 {slow.size(-1)}")
        batch, timesteps, _ = slow.shape

        embeds: list[torch.Tensor] = []
        if self.ultrasonic_encoder is not None:
            if ultrasonic is None or ultrasonic_scale is None:
                raise ValueError("use_ultrasonic=True 但未传入 ultrasonic / ultrasonic_scale")
            embeds.append(
                self._encode_waveform_branch(self.ultrasonic_encoder, ultrasonic, ultrasonic_scale, batch, timesteps)
            )
        if self.fiber_mic_encoder is not None:
            if fiber_mic is None or fiber_mic_scale is None:
                raise ValueError("use_fiber_mic=True 但未传入 fiber_mic / fiber_mic_scale")
            embeds.append(
                self._encode_waveform_branch(self.fiber_mic_encoder, fiber_mic, fiber_mic_scale, batch, timesteps)
            )

        fused = torch.cat([*embeds, slow], dim=-1)  # (B, T, fused_channels)
        fused_nct = fused.transpose(1, 2)            # (B, fused_channels, T)
        feats = self.tcn(fused_nct)                  # (B, C_out, T)

        last = feats[:, :, -1]
        avg = feats.mean(dim=-1)
        mx = feats.amax(dim=-1)
        pooled = torch.cat([last, avg, mx], dim=-1)  # (B, 3 * C_out)
        return self.head(pooled)
