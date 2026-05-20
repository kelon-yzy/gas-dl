"""CNN1D-TCN 慢变量分支实验模型。

在原始 `cnn1d_tcn_fusion` 基础上做结构隔离实验：
1. 可选地先用 MLP 对慢变量逐时间步编码，再与声学 embedding 融合。
2. TCN 池化后先得到共享融合表示，再拆成 target-specific 小 head。

原始 `cnn1d_tcn_fusion` 保持不变，便于和本实验做干净对照。
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from models.cnn1d_tcn_fusion import DeepAcousticEncoder1D, _TemporalBlock
from models.config_utils import merge_model_kwargs


class SlowFeatureEncoder(nn.Module):
    """逐时间步慢变量编码器。"""

    def __init__(self, in_dim: int, hidden_dim: int, embedding_dim: int):
        super().__init__()
        if in_dim <= 0:
            raise ValueError(f"in_dim 必须为正整数，得到 {in_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim 必须为正整数，得到 {hidden_dim}")
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim 必须为正整数，得到 {embedding_dim}")
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, slow: torch.Tensor) -> torch.Tensor:
        if slow.ndim != 3:
            raise ValueError(f"slow 必须为 3D [B, T, C]，实际 {tuple(slow.shape)}")
        return self.net(slow)


class GasHeadNormalize(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 4, target_sum: float = 100.0, eps: float = 1e-8):
        super().__init__()
        if in_dim <= 0:
            raise ValueError(f"in_dim 必须为正整数，得到 {in_dim}")
        if out_dim <= 0:
            raise ValueError(f"out_dim 必须为正整数，得到 {out_dim}")
        self.eps = float(eps)
        self.output_dim = int(out_dim)
        self.target_sum = float(target_sum)
        self.head = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.head(x)
        positive = F.softplus(raw) + self.eps
        y_pred = positive / positive.sum(dim=-1, keepdim=True)
        return y_pred * self.target_sum


class CNN1DTCNSlowBranchRegressor(nn.Module):
    """慢变量分支增强版 CNN1D-TCN 多模态回归器。"""

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
                "slow_encoder": {
                    "enabled": False,
                    "hidden_dim": 32,
                    "embedding_dim": 64,
                },
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
        slow_encoder_cfg = {
            "enabled": False,
            "hidden_dim": 32,
            "embedding_dim": 64,
        }
        slow_encoder_cfg.update(dict(settings["slow_encoder"]))

        if slow_dim <= 0:
            raise ValueError(f"slow_dim 必须为正整数，得到 {slow_dim}")
        if out_dim <= 0:
            raise ValueError(f"out_dim 必须为正整数，得到 {out_dim}")
        if not (use_ultrasonic or use_fiber_mic):
            raise ValueError("至少需要开启一个波形分支 (use_ultrasonic 或 use_fiber_mic)")
        if not tcn_channels:
            raise ValueError("tcn_channels 不能为空")

        self.slow_dim = slow_dim
        self.out_dim = out_dim
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

        self.slow_encoder = None
        slow_feature_dim = slow_dim
        if bool(slow_encoder_cfg["enabled"]):
            self.slow_encoder = SlowFeatureEncoder(
                in_dim=slow_dim,
                hidden_dim=int(slow_encoder_cfg["hidden_dim"]),
                embedding_dim=int(slow_encoder_cfg["embedding_dim"]),
            )
            slow_feature_dim = int(slow_encoder_cfg["embedding_dim"])

        enabled_branches = int(use_ultrasonic) + int(use_fiber_mic)
        fused_channels = slow_feature_dim + embedding_dim * enabled_branches

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

        self.shared_head = nn.Sequential(
            nn.Linear(current * 3, 128),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(head_dropout),
        )
        self.head = GasHeadNormalize(64, out_dim=out_dim, target_sum=100.0)

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

    def _encode_slow(self, slow: torch.Tensor) -> torch.Tensor:
        if slow.ndim != 3:
            raise ValueError(f"slow 必须为 3D [B, T, C]，实际 {tuple(slow.shape)}")
        if slow.size(-1) != self.slow_dim:
            raise ValueError(f"slow 通道数应为 {self.slow_dim}，实际 {slow.size(-1)}")
        if self.slow_encoder is None:
            return slow
        return self.slow_encoder(slow)

    def forward(
        self,
        ultrasonic: torch.Tensor | None,
        ultrasonic_scale: torch.Tensor | None,
        fiber_mic: torch.Tensor | None,
        fiber_mic_scale: torch.Tensor | None,
        slow: torch.Tensor,
    ) -> torch.Tensor:
        batch, timesteps, _ = slow.shape
        slow_features = self._encode_slow(slow)

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

        fused = torch.cat([*embeds, slow_features], dim=-1)
        fused_nct = fused.transpose(1, 2)
        feats = self.tcn(fused_nct)

        last = feats[:, :, -1]
        avg = feats.mean(dim=-1)
        mx = feats.amax(dim=-1)
        pooled = torch.cat([last, avg, mx], dim=-1)
        fusion_repr = self.shared_head(pooled)
        return self.head(fusion_repr)
