"""CNN1DTCNFusionRegressor 前向、配置、单分支与异常分支测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "dl"))

from models.cnn1d_tcn_fusion import (  # noqa: E402  (sys.path 调整后再 import)
    CNN1DTCNFusionRegressor,
    DeepAcousticEncoder1D,
)
from models.registry import build_model  # noqa: E402


def _make_batch(B: int = 4, T: int = 10, slow_dim: int = 8, ultrasonic_len: int = 1000, fiber_mic_len: int = 2000):
    return {
        "ultrasonic": torch.randint(0, 32768, (B, T, ultrasonic_len), dtype=torch.int16),
        "ultrasonic_scale": torch.rand(B, T),
        "fiber_mic": torch.randint(0, 32768, (B, T, fiber_mic_len), dtype=torch.int16),
        "fiber_mic_scale": torch.rand(B, T),
        "slow": torch.randn(B, T, slow_dim),
    }


class CNN1DTCNFusionForwardTests(unittest.TestCase):
    def test_registry_exposes_model(self):
        from models.registry import MODEL_REGISTRY
        self.assertIn("cnn1d_tcn_fusion", MODEL_REGISTRY)
        self.assertIs(MODEL_REGISTRY["cnn1d_tcn_fusion"], CNN1DTCNFusionRegressor)

    def test_default_forward_shape(self):
        batch = _make_batch()
        model = build_model({"name": "cnn1d_tcn_fusion"})
        model.eval()
        with torch.no_grad():
            out = model(
                batch["ultrasonic"],
                batch["ultrasonic_scale"],
                batch["fiber_mic"],
                batch["fiber_mic_scale"],
                batch["slow"],
            )
        self.assertEqual(out.shape, (4, 4))

    def test_use_waveform_attribute_true(self):
        model = build_model({"name": "cnn1d_tcn_fusion"})
        # runtime 通过 getattr(model, "use_waveform", False) 决定是否走多模态前向
        self.assertTrue(getattr(model, "use_waveform", False))
        self.assertEqual(getattr(model, "input_format", ""), "NCT")

    def test_fused_channels_dual_branch(self):
        """双分支 fused_channels = slow_dim + emb_dim * 2，进入 TCN 的第一个 Conv 通道数应一致。"""
        model = CNN1DTCNFusionRegressor(slow_dim=8, waveform_embedding_dim=64)
        first_block = model.tcn[0]
        # _TemporalBlock 内第一个子模块是 _CausalConv1d，其 conv.in_channels 即融合后的通道
        self.assertEqual(first_block.net[0].conv.in_channels, 8 + 64 * 2)

    def test_fused_channels_single_branch(self):
        model = CNN1DTCNFusionRegressor(slow_dim=8, waveform_embedding_dim=64, use_fiber_mic=False)
        self.assertEqual(model.tcn[0].net[0].conv.in_channels, 8 + 64)

    def test_head_input_is_three_times_tcn_channels(self):
        """末位 + 均值 + 最大三池化拼接 → MLP 入口宽度应为 tcn_out_channels * 3。"""
        model = CNN1DTCNFusionRegressor(tcn_channels=[64, 64, 64])
        self.assertEqual(model.head[0].in_features, 64 * 3)

    def test_ultrasonic_only_branch_forward(self):
        batch = _make_batch()
        model = build_model({"name": "cnn1d_tcn_fusion", "use_fiber_mic": False})
        model.eval()
        with torch.no_grad():
            out = model(batch["ultrasonic"], batch["ultrasonic_scale"], None, None, batch["slow"])
        self.assertEqual(out.shape, (4, 4))

    def test_fiber_mic_only_branch_forward(self):
        batch = _make_batch()
        model = build_model({"name": "cnn1d_tcn_fusion", "use_ultrasonic": False})
        model.eval()
        with torch.no_grad():
            out = model(None, None, batch["fiber_mic"], batch["fiber_mic_scale"], batch["slow"])
        self.assertEqual(out.shape, (4, 4))

    def test_no_branch_raises_value_error(self):
        with self.assertRaises(ValueError):
            build_model({"name": "cnn1d_tcn_fusion", "use_ultrasonic": False, "use_fiber_mic": False})

    def test_unknown_config_key_raises(self):
        # merge_model_kwargs 会把未知键抛 TypeError
        with self.assertRaises(TypeError):
            build_model({"name": "cnn1d_tcn_fusion", "typo_channels": [16, 32]})

    def test_missing_ultrasonic_when_enabled_raises(self):
        model = build_model({"name": "cnn1d_tcn_fusion"})
        model.eval()
        batch = _make_batch()
        with self.assertRaises(ValueError):
            model(None, None, batch["fiber_mic"], batch["fiber_mic_scale"], batch["slow"])

    def test_slow_dim_mismatch_raises(self):
        model = build_model({"name": "cnn1d_tcn_fusion", "slow_dim": 8})
        batch = _make_batch(slow_dim=6)
        model.eval()
        with self.assertRaises(ValueError):
            model(
                batch["ultrasonic"],
                batch["ultrasonic_scale"],
                batch["fiber_mic"],
                batch["fiber_mic_scale"],
                batch["slow"],
            )

    def test_out_dim_configurable(self):
        batch = _make_batch()
        model = build_model({"name": "cnn1d_tcn_fusion", "out_dim": 3})
        model.eval()
        with torch.no_grad():
            out = model(
                batch["ultrasonic"],
                batch["ultrasonic_scale"],
                batch["fiber_mic"],
                batch["fiber_mic_scale"],
                batch["slow"],
            )
        self.assertEqual(out.shape, (4, 3))


class DeepAcousticEncoder1DTests(unittest.TestCase):
    def test_output_shape(self):
        encoder = DeepAcousticEncoder1D(embedding_dim=64)
        wave = torch.randint(0, 32768, (8, 1000), dtype=torch.int16)
        scale = torch.rand(8)
        out = encoder(wave, scale)
        self.assertEqual(out.shape, (8, 64))
        # projection 入口宽度 = avg(64) || max(64) || log_scale(1) = 129
        self.assertEqual(encoder.projection[0].in_features, 64 * 2 + 1)

    def test_different_scale_factors_change_embedding(self):
        encoder = DeepAcousticEncoder1D(embedding_dim=64)
        encoder.eval()
        base_wave = torch.randint(-32768, 32767, (1, 1000), dtype=torch.int16)
        waves = base_wave.repeat(2, 1)
        scales = torch.tensor([0.25, 2.0], dtype=torch.float32)
        with torch.no_grad():
            out = encoder(waves, scales)
        # 新语义下卷积分支只吃 waveform_int16/32767，与 scale_factor 无关；
        # 输出差异完全来自 log_scale 旁路进入 projection 的第一层 Linear。
        self.assertGreater(torch.max(torch.abs(out[0] - out[1])).item(), 1e-4)

    def test_invalid_embedding_dim(self):
        with self.assertRaises(ValueError):
            DeepAcousticEncoder1D(embedding_dim=0)

    def test_invalid_channels(self):
        with self.assertRaises(ValueError):
            DeepAcousticEncoder1D(channels=[32])

    def test_invalid_kernel_size_even(self):
        with self.assertRaises(ValueError):
            DeepAcousticEncoder1D(kernel_size=6)

    def test_scale_factor_shape_mismatch(self):
        encoder = DeepAcousticEncoder1D()
        wave = torch.randint(0, 32768, (8, 1000), dtype=torch.int16)
        bad_scale = torch.rand(8, 1)  # 2D 而非 1D
        with self.assertRaises(ValueError):
            encoder(wave, bad_scale)


if __name__ == "__main__":
    unittest.main()
