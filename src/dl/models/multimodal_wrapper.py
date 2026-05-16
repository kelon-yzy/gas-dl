from __future__ import annotations

import torch
from torch import nn

from models.acoustic_waveform_encoder import AcousticWaveformEncoder
from models.cnn1d import CNN1DRegressor
from models.cnn_lstm import CNNLSTMRegressor
from models.gru import GRURegressor
from models.lstm import LSTMRegressor
from models.tcn import TCNRegressor
from models.transformer_encoder import TransformerRegressor

# 每个 backbone 可接受的参数白名单
_BACKBONE_KWARGS: dict[str, set[str]] = {
    "CNN1DRegressor": {"in_channels", "hidden_channels", "kernel_size", "dropout", "out_dim"},
    "GRURegressor": {"input_size", "hidden_size", "num_layers", "bidirectional", "dropout", "out_dim"},
    "LSTMRegressor": {"input_size", "hidden_size", "num_layers", "bidirectional", "dropout", "out_dim"},
    "TCNRegressor": {"in_channels", "channels", "kernel_size", "dropout", "out_dim"},
    "TransformerRegressor": {"input_dim", "d_model", "nhead", "num_layers", "dim_feedforward", "dropout", "out_dim"},
    "CNNLSTMRegressor": {"in_channels", "conv_channels", "hidden_size", "num_layers", "kernel_size", "dropout", "out_dim"},
}

# backbone 输入维度参数名 → 多模态版本会动态设为 fused_dim
_INPUT_DIM_PARAM: dict[str, str] = {
    "CNN1DRegressor": "in_channels",
    "GRURegressor": "input_size",
    "LSTMRegressor": "input_size",
    "TCNRegressor": "in_channels",
    "TransformerRegressor": "input_dim",
    "CNNLSTMRegressor": "in_channels",
}


class MultimodalWrapper(nn.Module):
    """通用波形融合包装器：给任意纯慢变量 backbone 加上超声/光纤麦克风编码分支。

    backbone 由工厂函数预先构造，其输入维度已扩展为 slow_dim + waveform_embedding_dim * enabled_branch_count。
    包装器负责：
      1. 将超声波形编码为 B×T×D 的嵌入
      2. 将光纤麦克风波形编码为 B×T×D 的嵌入
      3. 沿通道维拼接 → [B, T, fused_dim]
      4. 按 backbone 的 input_format (NCT / NTC) 调整形状后送入 backbone
    """

    use_waveform: bool = True

    def __init__(
        self,
        backbone: nn.Module,
        slow_dim: int = 8,
        use_ultrasonic: bool = True,
        use_fiber_mic: bool = True,
        waveform_embedding_dim: int = 64,
    ):
        super().__init__()
        if not (use_ultrasonic or use_fiber_mic):
            raise ValueError("至少需要开启一个波形分支 (use_ultrasonic 或 use_fiber_mic)")
        self.slow_dim = slow_dim
        self.use_ultrasonic = use_ultrasonic
        self.use_fiber_mic = use_fiber_mic
        self.waveform_embedding_dim = waveform_embedding_dim
        self.input_format = getattr(backbone, "input_format", "NTC").upper()
        self.backbone = backbone

        self.ultrasonic_encoder = AcousticWaveformEncoder(waveform_embedding_dim) if use_ultrasonic else None
        self.fiber_mic_encoder = AcousticWaveformEncoder(waveform_embedding_dim) if use_fiber_mic else None

    def forward(
        self,
        ultrasonic: torch.Tensor | None,
        ultrasonic_scale: torch.Tensor | None,
        fiber_mic: torch.Tensor | None,
        fiber_mic_scale: torch.Tensor | None,
        slow: torch.Tensor,
    ) -> torch.Tensor:
        if slow.ndim != 3:
            raise ValueError(f"slow 必须为 3D [B, T, C], 实际 {tuple(slow.shape)}")
        B, T, _ = slow.shape
        embeds: list[torch.Tensor] = []

        if self.ultrasonic_encoder is not None:
            if ultrasonic is None or ultrasonic_scale is None:
                raise ValueError("use_ultrasonic=True 但未传入 ultrasonic / ultrasonic_scale")
            flat_u = ultrasonic.reshape(B * T, ultrasonic.size(-1))
            flat_us = ultrasonic_scale.reshape(B * T)
            u_emb = self.ultrasonic_encoder(flat_u, flat_us).reshape(B, T, -1)
            embeds.append(u_emb)

        if self.fiber_mic_encoder is not None:
            if fiber_mic is None or fiber_mic_scale is None:
                raise ValueError("use_fiber_mic=True 但未传入 fiber_mic / fiber_mic_scale")
            flat_f = fiber_mic.reshape(B * T, fiber_mic.size(-1))
            flat_fs = fiber_mic_scale.reshape(B * T)
            f_emb = self.fiber_mic_encoder(flat_f, flat_fs).reshape(B, T, -1)
            embeds.append(f_emb)

        fused: torch.Tensor = torch.cat([*embeds, slow], dim=-1)  # [B, T, fused_dim]

        if self.input_format == "NCT":
            fused = fused.transpose(1, 2)  # [B, fused_dim, T]

        return self.backbone(fused)


# ── 工厂函数 ──

def _build_multimodal(
    backbone_class: type[nn.Module],
    **kwargs,
) -> MultimodalWrapper:
    """构造多模态包装器：将 backbone 的输入维度替换为 fused_dim，只传 backbone 认可的参数。

    wrapper 参数 (slow_dim, use_ultrasonic, use_fiber_mic, waveform_embedding_dim, out_dim)
    从 kwargs 中提取后不再传给 backbone。
    """
    backbone_name = backbone_class.__name__
    allowed = _BACKBONE_KWARGS.get(backbone_name, set())
    input_param = _INPUT_DIM_PARAM[backbone_name]

    # ── 提取 wrapper 自身消费的参数 ──
    slow_dim = int(kwargs.pop("slow_dim", 8))
    use_ultrasonic = bool(kwargs.pop("use_ultrasonic", True))
    use_fiber_mic = bool(kwargs.pop("use_fiber_mic", True))
    waveform_embedding_dim = int(kwargs.pop("waveform_embedding_dim", 64))

    # ── 计算合并后的输入维度 ──
    enabled_count = int(use_ultrasonic) + int(use_fiber_mic)
    if enabled_count == 0:
        raise ValueError("至少需要开启一个波形分支 (use_ultrasonic 或 use_fiber_mic)")
    fused_dim = slow_dim + waveform_embedding_dim * enabled_count

    # ── 只传白名单内的参数给 backbone ──
    backbone_kwargs: dict = {input_param: fused_dim}
    for key in list(kwargs.keys()):
        if key in allowed and key != input_param:
            backbone_kwargs[key] = kwargs.pop(key)

    backbone = backbone_class(**backbone_kwargs)
    return MultimodalWrapper(
        backbone=backbone,
        slow_dim=slow_dim,
        use_ultrasonic=use_ultrasonic,
        use_fiber_mic=use_fiber_mic,
        waveform_embedding_dim=waveform_embedding_dim,
    )


def build_cnn1d_multimodal(**kwargs) -> MultimodalWrapper:
    return _build_multimodal(CNN1DRegressor, **kwargs)


def build_gru_multimodal(**kwargs) -> MultimodalWrapper:
    return _build_multimodal(GRURegressor, **kwargs)


def build_lstm_multimodal(**kwargs) -> MultimodalWrapper:
    return _build_multimodal(LSTMRegressor, **kwargs)


def build_tcn_multimodal(**kwargs) -> MultimodalWrapper:
    return _build_multimodal(TCNRegressor, **kwargs)


def build_transformer_multimodal(**kwargs) -> MultimodalWrapper:
    return _build_multimodal(TransformerRegressor, **kwargs)


def build_cnn_lstm_multimodal(**kwargs) -> MultimodalWrapper:
    return _build_multimodal(CNNLSTMRegressor, **kwargs)