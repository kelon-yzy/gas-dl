"""测试 EarlyFusionFiLMRegressor（E2）：前向 shape、registry 解析、单分支、错误路径。"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "dl"))

import torch
from models.early_fusion_film import EarlyFusionFiLMRegressor, _FiLM
from models.registry import build_model


def _make_batch(B: int = 2, T: int = 12, slow_dim: int = 8):
    """构造与 waveform_v3 dataset 对齐的最小合成 batch。"""
    return {
        "ultrasonic": torch.randint(0, 32768, (B, T, 1000), dtype=torch.int16),
        "ultrasonic_scale": torch.rand(B, T) + 1e-3,
        "fiber_mic": torch.randint(0, 32768, (B, T, 2000), dtype=torch.int16),
        "fiber_mic_scale": torch.rand(B, T) + 1e-3,
        "slow": torch.randn(B, T, slow_dim),
    }


class EarlyFusionFiLMTests(unittest.TestCase):
    def test_forward_output_shape(self):
        """双分支前向输出 [B, out_dim]，且模型自动暴露 use_waveform=True。"""
        model = build_model(
            {
                "name": "early_fusion_film",
                "slow_dim": 8,
                "d_model": 32,
                "conv_channels": [8, 16, 24, 32],
                "conv_kernels": [15, 11, 7, 5],
                "transformer_layers": 1,
                "transformer_heads": 4,
                "transformer_ff_mult": 2,
                "out_dim": 4,
                "max_timesteps": 12,
            }
        )
        self.assertTrue(getattr(model, "use_waveform", False))
        model.eval()
        batch = _make_batch(B=2, T=12)
        with torch.no_grad():
            out = model(
                batch["ultrasonic"],
                batch["ultrasonic_scale"],
                batch["fiber_mic"],
                batch["fiber_mic_scale"],
                batch["slow"],
            )
        self.assertEqual(out.shape, (2, 4))
        self.assertEqual(out.dtype, torch.float32)
        self.assertFalse(torch.isnan(out).any())

    def test_ultrasonic_only_branch(self):
        """关闭 fiber 分支时，fiber 输入可为 None，输出形状仍为 [B, out_dim]。"""
        model = build_model(
            {
                "name": "early_fusion_film",
                "d_model": 32,
                "conv_channels": [8, 16, 24, 32],
                "conv_kernels": [15, 11, 7, 5],
                "transformer_layers": 1,
                "transformer_heads": 4,
                "transformer_ff_mult": 2,
                "out_dim": 4,
                "max_timesteps": 12,
                "use_fiber_mic": False,
            }
        )
        model.eval()
        batch = _make_batch(B=2, T=12)
        with torch.no_grad():
            out = model(batch["ultrasonic"], batch["ultrasonic_scale"], None, None, batch["slow"])
        self.assertEqual(out.shape, (2, 4))
        self.assertIsNone(model.fiber_down)

    def test_fiber_only_branch(self):
        """关闭 ultrasonic 分支时，超声输入可为 None。"""
        model = build_model(
            {
                "name": "early_fusion_film",
                "d_model": 32,
                "conv_channels": [8, 16, 24, 32],
                "conv_kernels": [15, 11, 7, 5],
                "transformer_layers": 1,
                "transformer_heads": 4,
                "transformer_ff_mult": 2,
                "out_dim": 4,
                "max_timesteps": 12,
                "use_ultrasonic": False,
            }
        )
        model.eval()
        batch = _make_batch(B=2, T=12)
        with torch.no_grad():
            out = model(None, None, batch["fiber_mic"], batch["fiber_mic_scale"], batch["slow"])
        self.assertEqual(out.shape, (2, 4))
        self.assertIsNotNone(model.fiber_down)

    def test_no_branch_raises(self):
        """两路波形都关闭必须抛 ValueError，避免静默退化为纯慢变量模型。"""
        with self.assertRaisesRegex(ValueError, "波形分支"):
            build_model(
                {
                    "name": "early_fusion_film",
                    "use_ultrasonic": False,
                    "use_fiber_mic": False,
                }
            )

    def test_unknown_kwargs_rejected(self):
        """未知配置项必须报错，避免静默回退默认值。"""
        with self.assertRaises(TypeError):
            build_model({"name": "early_fusion_film", "typo_param": 1})

    def test_conv_channels_must_match_d_model(self):
        """conv_channels[-1] 必须等于 d_model，否则池化后维度与 Transformer 不一致。"""
        with self.assertRaisesRegex(ValueError, "d_model"):
            build_model(
                {
                    "name": "early_fusion_film",
                    "d_model": 64,
                    "conv_channels": [8, 16, 24, 32],
                    "conv_kernels": [15, 11, 7, 5],
                }
            )

    def test_film_zero_init_identity_on_init(self):
        """zero_init=True 时，FiLM 的 γ/β 末层权重应全零，前向退化为恒等映射。"""
        film = _FiLM(cond_dim=10, channels=32, zero_init=True)
        last_linear = film.net[-1]
        self.assertTrue(torch.all(last_linear.weight == 0))
        self.assertTrue(torch.all(last_linear.bias == 0))
        h = torch.randn(4, 32, 50)
        cond = torch.randn(4, 10)
        out = film(h, cond)
        self.assertTrue(torch.allclose(out, h, atol=1e-6))

    def test_film_zero_init_disabled(self):
        """zero_init=False 时，γ/β 末层应为非零初始化，FiLM 不再恒等。"""
        film = _FiLM(cond_dim=10, channels=32, zero_init=False)
        last_linear = film.net[-1]
        self.assertFalse(torch.all(last_linear.weight == 0))

    def test_positional_embedding_optional(self):
        """use_positional_embedding=False 时不创建位置嵌入参数。"""
        model = build_model(
            {
                "name": "early_fusion_film",
                "d_model": 32,
                "conv_channels": [8, 16, 24, 32],
                "conv_kernels": [15, 11, 7, 5],
                "transformer_layers": 1,
                "transformer_heads": 4,
                "transformer_ff_mult": 2,
                "max_timesteps": 12,
                "use_positional_embedding": False,
            }
        )
        self.assertIsNone(model.positional_embedding)

    def test_timesteps_exceeding_max_raises(self):
        """序列长度超过 max_timesteps 时位置嵌入越界，应主动报错而非默默截断。"""
        model = build_model(
            {
                "name": "early_fusion_film",
                "d_model": 32,
                "conv_channels": [8, 16, 24, 32],
                "conv_kernels": [15, 11, 7, 5],
                "transformer_layers": 1,
                "transformer_heads": 4,
                "transformer_ff_mult": 2,
                "max_timesteps": 10,
            }
        )
        batch = _make_batch(B=2, T=12)
        with self.assertRaisesRegex(ValueError, "max_timesteps"):
            model(
                batch["ultrasonic"],
                batch["ultrasonic_scale"],
                batch["fiber_mic"],
                batch["fiber_mic_scale"],
                batch["slow"],
            )

    def test_input_format_attribute(self):
        """对外暴露 input_format='NCT'，符合 multimodal 模型约定。"""
        self.assertEqual(EarlyFusionFiLMRegressor.input_format, "NCT")
        self.assertTrue(EarlyFusionFiLMRegressor.use_waveform)

    def test_backward_runs(self):
        """前后向能正常完成一次梯度更新，慢变量 + 波形 + 卷积栈均能产生梯度。"""
        model = build_model(
            {
                "name": "early_fusion_film",
                "d_model": 32,
                "conv_channels": [8, 16, 24, 32],
                "conv_kernels": [15, 11, 7, 5],
                "transformer_layers": 1,
                "transformer_heads": 4,
                "transformer_ff_mult": 2,
                "max_timesteps": 12,
            }
        )
        batch = _make_batch(B=2, T=12)
        target = torch.randn(2, 4)
        out = model(
            batch["ultrasonic"],
            batch["ultrasonic_scale"],
            batch["fiber_mic"],
            batch["fiber_mic_scale"],
            batch["slow"],
        )
        loss = ((out - target) ** 2).mean()
        loss.backward()
        # 卷积栈第一层应有梯度
        first_conv = model.conv_blocks[0].net[0]
        self.assertIsNotNone(first_conv.weight.grad)
        self.assertFalse(torch.all(first_conv.weight.grad == 0))
        # FiLM 末层因为零初始化，初次反向梯度未必为零但应能产生（hidden 非零 → bias 梯度非零）
        last_film_bias = model.film_blocks[-1].net[-1].bias
        self.assertIsNotNone(last_film_bias.grad)


if __name__ == "__main__":
    unittest.main()
