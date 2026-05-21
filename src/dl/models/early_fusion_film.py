"""FiLM 调制型早期融合回归模型（E2，对应 docs/早期融合_Early_Fusion_完整实验方案.md §7）。

设计目标：
    1. 超声 + 光纤麦克风波形在每个低频时间步内做长度对齐与通道拼接。
    2. 慢变量（8 维）+ 启用分支的 log(scale)（最多 2 维）拼成条件向量，
       通过 FiLM 在每个卷积块输出上按通道生成 γ / β：h ⊙ (1 + γ) + β。
       γ / β 末层零初始化，训练初期退化为恒等映射，不破坏卷积栈表达。
    3. 4 层 stride=2 卷积主干把 1000 点波形压成 d_model 维 token。
    4. 可学习位置嵌入 + TransformerEncoder + 注意力池化。
    5. 单一 [B, out_dim] 输出，与现有 MSE / MAE + SumConstraintLoss 损失兼容。

前向签名与 CNN1DTCNFusionRegressor 对齐，runtime 通过 ``use_waveform = True``
自动识别波形批分支并按 (ultrasonic, ultrasonic_scale, fiber_mic, fiber_mic_scale, slow)
顺序调用。
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import nn

from models.config_utils import merge_model_kwargs


# int16 ADC 满量程常数：与 src/dl/models/acoustic_waveform_encoder.py 对齐。
# 卷积分支只消费 ``waveform_int16 / ADC_MAX``（形状归一化），
# 绝对幅值通过 log(scale) 加入 FiLM 条件向量，避免成为额外波形通道。
_ADC_MAX_INT16: float = 32767.0


def _maybe_disable_autocast(device: torch.device):
    """波形 int16→float32 转换 + log 运算在 FP16 下数值范围易越界，编码阶段强制 FP32。"""
    if device.type == "cuda":
        return torch.amp.autocast(device_type=device.type, enabled=False)
    return nullcontext()


class _FiberDownsample(nn.Module):
    """光纤麦克风 2000 → 1000 点的可学习降采样。

    与文档 §7.3 LearnableFiberDownsample 一致：两层 1D 卷积，
    第一层 stride=2 完成下采样，第二层 stride=1 把通道映射回 1。
    """

    def __init__(self):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(8, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, fiber: torch.Tensor) -> torch.Tensor:
        # fiber: [N, 1, 2000] → [N, 1, 1000]
        return self.down(fiber)


class _ConvBlock1D(nn.Module):
    """两层 Conv1d + GroupNorm + GELU 的卷积块。

    第一层按给定 kernel_size 与 stride 做下采样，第二层 k=3 stride=1 做特征精化。
    与文档 §6.3 ConvBlock1D 一致，GroupNorm 对小 batch 友好且不引入 train/eval 偏差。
    """

    def __init__(self, c_in: int, c_out: int, kernel_size: int, stride: int, dropout: float = 0.0, groups: int = 8):
        super().__init__()
        ng = max(1, min(groups, c_out))
        layers: list[nn.Module] = [
            nn.Conv1d(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, bias=False),
            nn.GroupNorm(num_groups=ng, num_channels=c_out),
            nn.GELU(),
            nn.Conv1d(c_out, c_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(num_groups=ng, num_channels=c_out),
            nn.GELU(),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _FiLM(nn.Module):
    """Feature-wise Linear Modulation：h ⊙ (1 + γ(s)) + β(s)。

    γ / β 由两层 MLP 从条件向量 s 生成。最后一层 Linear 零初始化，保证训练
    初期 γ ≈ 0, β ≈ 0，FiLM 退化为恒等映射，不破坏卷积栈的初始表达。
    """

    def __init__(self, cond_dim: int, channels: int, zero_init: bool = True):
        super().__init__()
        self.channels = channels
        hidden = max(2 * channels, 16)
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * channels),
        )
        if zero_init:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # h: [N, C, L], cond: [N, cond_dim] → γ, β: [N, C, 1]
        gb = self.net(cond)
        gamma, beta = gb.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1)
        beta = beta.unsqueeze(-1)
        return h * (1.0 + gamma) + beta


class EarlyFusionFiLMRegressor(nn.Module):
    """E2：FiLM 调制型早期融合回归模型。

    前向签名：
        model(ultrasonic, ultrasonic_scale, fiber_mic, fiber_mic_scale, slow)

    输入张量形状：
        ultrasonic       (B, T, 1000) int16，use_ultrasonic=False 时允许 None
        ultrasonic_scale (B, T)       float32
        fiber_mic        (B, T, 2000) int16，use_fiber_mic=False 时允许 None
        fiber_mic_scale  (B, T)       float32
        slow             (B, T, slow_dim) float32
    输出：
        (B, out_dim) float32 —— 直接回归值，外层 SumConstraintLoss 负责组分和约束。

    FiLM 条件向量按 ``[slow, log(u_scale), log(f_scale)]`` 拼接，单分支时跳过对应分量。
    """

    use_waveform: bool = True
    input_format: str = "NCT"

    _ULTRA_LEN: int = 1000
    _FIBER_LEN: int = 2000

    def __init__(self, config: dict | None = None, **kwargs):
        super().__init__()
        settings = merge_model_kwargs(
            {
                "slow_dim": 8,
                "d_model": 128,
                "conv_channels": [32, 64, 96, 128],
                "conv_kernels": [15, 11, 7, 5],
                "conv_dropout": 0.0,
                "transformer_layers": 3,
                "transformer_heads": 4,
                "transformer_ff_mult": 4,
                "transformer_dropout": 0.1,
                "head_dropout": 0.1,
                "out_dim": 4,
                "max_timesteps": 120,
                "use_positional_embedding": True,
                "film_zero_init": True,
                "use_ultrasonic": True,
                "use_fiber_mic": True,
            },
            config,
            kwargs,
        )
        slow_dim = int(settings["slow_dim"])
        d_model = int(settings["d_model"])
        conv_channels = [int(c) for c in settings["conv_channels"]]
        conv_kernels = [int(k) for k in settings["conv_kernels"]]
        conv_dropout = float(settings["conv_dropout"])
        transformer_layers = int(settings["transformer_layers"])
        transformer_heads = int(settings["transformer_heads"])
        transformer_ff_mult = int(settings["transformer_ff_mult"])
        transformer_dropout = float(settings["transformer_dropout"])
        head_dropout = float(settings["head_dropout"])
        out_dim = int(settings["out_dim"])
        max_timesteps = int(settings["max_timesteps"])
        use_positional_embedding = bool(settings["use_positional_embedding"])
        film_zero_init = bool(settings["film_zero_init"])
        use_ultrasonic = bool(settings["use_ultrasonic"])
        use_fiber_mic = bool(settings["use_fiber_mic"])

        if slow_dim <= 0:
            raise ValueError(f"slow_dim 必须为正整数，得到 {slow_dim}")
        if not (use_ultrasonic or use_fiber_mic):
            raise ValueError("至少需要开启一个波形分支 (use_ultrasonic 或 use_fiber_mic)")
        if not conv_channels or not conv_kernels:
            raise ValueError("conv_channels 与 conv_kernels 不能为空")
        if len(conv_channels) != len(conv_kernels):
            raise ValueError(
                f"conv_channels 与 conv_kernels 长度必须一致：得到 {len(conv_channels)} vs {len(conv_kernels)}"
            )
        if conv_channels[-1] != d_model:
            raise ValueError(
                f"conv_channels 最后一层必须等于 d_model：得到 conv_channels[-1]={conv_channels[-1]}, d_model={d_model}"
            )
        if any(k % 2 == 0 for k in conv_kernels):
            raise ValueError(f"conv_kernels 必须为奇数序列，得到 {conv_kernels}")
        if d_model % transformer_heads != 0:
            raise ValueError(
                f"d_model={d_model} 必须能被 transformer_heads={transformer_heads} 整除"
            )
        if max_timesteps <= 0:
            raise ValueError(f"max_timesteps 必须为正整数，得到 {max_timesteps}")

        self.slow_dim = slow_dim
        self.d_model = d_model
        self.use_ultrasonic = use_ultrasonic
        self.use_fiber_mic = use_fiber_mic
        self.use_positional_embedding = use_positional_embedding
        self.max_timesteps = max_timesteps

        # 输入通道数 = 启用分支数；FiLM 条件维 = slow + 启用分支的 log_scale
        enabled_branches = int(use_ultrasonic) + int(use_fiber_mic)
        in_channels = enabled_branches
        cond_dim = slow_dim + enabled_branches

        self.fiber_down = _FiberDownsample() if use_fiber_mic else None

        # 4 层下采样卷积主干 + 每层后的 FiLM 调制
        self.conv_blocks = nn.ModuleList()
        self.film_blocks = nn.ModuleList()
        current = in_channels
        for c, k in zip(conv_channels, conv_kernels):
            self.conv_blocks.append(_ConvBlock1D(current, c, kernel_size=k, stride=2, dropout=conv_dropout))
            self.film_blocks.append(_FiLM(cond_dim=cond_dim, channels=c, zero_init=film_zero_init))
            current = c

        self.pool = nn.AdaptiveAvgPool1d(1)

        # 时序模块：可选位置嵌入 + TransformerEncoder
        if use_positional_embedding:
            self.positional_embedding = nn.Parameter(torch.zeros(1, max_timesteps, d_model))
            nn.init.trunc_normal_(self.positional_embedding, std=0.02)
        else:
            self.positional_embedding = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=transformer_heads,
            dim_feedforward=transformer_ff_mult * d_model,
            dropout=transformer_dropout,
            batch_first=True,
            activation="gelu",
        )
        self.temporal = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        # 注意力池化 over T
        self.pool_gate = nn.Linear(d_model, 1)

        # 回归头：LayerNorm → MLP → out_dim
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(64, out_dim),
        )

    @staticmethod
    def _prepare_waveform(
        waveform_int16: torch.Tensor,
        scale: torch.Tensor,
        expected_len: int,
        batch: int,
        timesteps: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """int16 + scale → [N, 1, L] FP32 波形 + [N] log_scale，N = batch * timesteps。"""
        if waveform_int16.shape[:2] != (batch, timesteps):
            raise ValueError(f"波形前两维应为 ({batch}, {timesteps})，实际 {tuple(waveform_int16.shape[:2])}")
        if scale.shape != (batch, timesteps):
            raise ValueError(f"scale 形状应为 ({batch}, {timesteps})，实际 {tuple(scale.shape)}")
        if waveform_int16.size(-1) != expected_len:
            raise ValueError(f"waveform 末维应为 {expected_len}，实际 {waveform_int16.size(-1)}")
        flat_wave = waveform_int16.reshape(batch * timesteps, expected_len).to(torch.float32) / _ADC_MAX_INT16
        flat_scale = scale.reshape(batch * timesteps).to(torch.float32).clamp_min(1e-12)
        log_scale = torch.log(flat_scale)
        return flat_wave.unsqueeze(1), log_scale

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
        B, T, _ = slow.shape
        if T > self.max_timesteps:
            raise ValueError(f"序列长度 T={T} 超过 max_timesteps={self.max_timesteps}")
        NT = B * T

        with _maybe_disable_autocast(slow.device):
            wave_channels: list[torch.Tensor] = []
            log_scales: list[torch.Tensor] = []

            if self.use_ultrasonic:
                if ultrasonic is None or ultrasonic_scale is None:
                    raise ValueError("use_ultrasonic=True 但未传入 ultrasonic / ultrasonic_scale")
                wave_u, log_us = self._prepare_waveform(
                    ultrasonic, ultrasonic_scale, self._ULTRA_LEN, B, T
                )
                wave_channels.append(wave_u)
                log_scales.append(log_us)

            if self.use_fiber_mic:
                if fiber_mic is None or fiber_mic_scale is None:
                    raise ValueError("use_fiber_mic=True 但未传入 fiber_mic / fiber_mic_scale")
                wave_f, log_fs = self._prepare_waveform(
                    fiber_mic, fiber_mic_scale, self._FIBER_LEN, B, T
                )
                wave_f = self.fiber_down(wave_f)  # [NT, 1, 2000] → [NT, 1, 1000]
                wave_channels.append(wave_f)
                log_scales.append(log_fs)

            x = torch.cat(wave_channels, dim=1)  # [NT, n_branch, 1000]

            slow_flat = slow.reshape(NT, self.slow_dim).to(torch.float32)
            log_stack = [s.unsqueeze(-1) for s in log_scales]
            cond = torch.cat([slow_flat, *log_stack], dim=-1)  # [NT, cond_dim]

            # 卷积主干：每层 conv 后接一次 FiLM 调制
            for conv, film in zip(self.conv_blocks, self.film_blocks):
                x = conv(x)
                x = film(x, cond)

            x = self.pool(x).squeeze(-1)         # [NT, d_model]
            tokens = x.reshape(B, T, self.d_model)  # [B, T, d_model]

        # Transformer / pooling / head 沿用外层 AMP 上下文
        if self.positional_embedding is not None:
            tokens = tokens + self.positional_embedding[:, :T, :]

        encoded = self.temporal(tokens)
        gate = torch.softmax(self.pool_gate(encoded), dim=1)  # [B, T, 1]
        pooled = (gate * encoded).sum(dim=1)                  # [B, d_model]
        return self.head(pooled)
