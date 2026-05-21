"""CNN1D-LSTM 慢变量分支实验模型。"""

from __future__ import annotations

import torch
from torch import nn

from models.cnn1d_tcn_fusion import DeepAcousticEncoder1D
from models.cnn1d_tcn_fusion_slow_branch import GasHeadNormalize, SlowFeatureEncoder
from models.config_utils import merge_model_kwargs


class CNN1DLSTMSlowBranchRegressor(nn.Module):
    """慢变量分支增强版 CNN1D-LSTM 多模态回归器。"""

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
                "lstm_hidden_size": 64,
                "lstm_num_layers": 1,
                "lstm_bidirectional": False,
                "lstm_dropout": 0.0,
                "head_dropout": 0.2,
                "out_dim": 4,
                "use_ultrasonic": True,
                "use_fiber_mic": True,
                "derive_last": False,
                "derive_last_mode": "residual",
                "output_prior": None,
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
        lstm_hidden_size = int(settings["lstm_hidden_size"])
        lstm_num_layers = int(settings["lstm_num_layers"])
        lstm_bidirectional = bool(settings["lstm_bidirectional"])
        lstm_dropout = float(settings["lstm_dropout"])
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
        if lstm_hidden_size <= 0:
            raise ValueError(f"lstm_hidden_size 必须为正整数，得到 {lstm_hidden_size}")
        if lstm_num_layers <= 0:
            raise ValueError(f"lstm_num_layers 必须为正整数，得到 {lstm_num_layers}")
        if out_dim <= 0:
            raise ValueError(f"out_dim 必须为正整数，得到 {out_dim}")
        if not (use_ultrasonic or use_fiber_mic):
            raise ValueError("至少需要开启一个波形分支 (use_ultrasonic 或 use_fiber_mic)")

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
        recurrent_dropout = lstm_dropout if lstm_num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=fused_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
            bidirectional=lstm_bidirectional,
        )
        self.lstm_out_channels = lstm_hidden_size * (2 if lstm_bidirectional else 1)

        self.shared_head = nn.Sequential(
            nn.Linear(self.lstm_out_channels * 3, 128),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(head_dropout),
        )
        self.head = GasHeadNormalize(
            64,
            out_dim=out_dim,
            target_sum=100.0,
            derive_last=bool(settings["derive_last"]),
            derive_last_mode=str(settings["derive_last_mode"]),
            output_prior=settings.get("output_prior"),
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
        feats, _ = self.lstm(fused)

        last = feats[:, -1, :]
        avg = feats.mean(dim=1)
        mx = feats.amax(dim=1)
        pooled = torch.cat([last, avg, mx], dim=-1)
        fusion_repr = self.shared_head(pooled)
        return self.head(fusion_repr)
