"""CNN1D-LSTM 慢变量分支实验模型测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "dl"))

from models.registry import MODEL_REGISTRY, build_model  # noqa: E402


def _make_batch(B: int = 4, T: int = 10, slow_dim: int = 8, ultrasonic_len: int = 1000, fiber_mic_len: int = 2000):
    return {
        "ultrasonic": torch.randint(0, 32768, (B, T, ultrasonic_len), dtype=torch.int16),
        "ultrasonic_scale": torch.rand(B, T),
        "fiber_mic": torch.randint(0, 32768, (B, T, fiber_mic_len), dtype=torch.int16),
        "fiber_mic_scale": torch.rand(B, T),
        "slow": torch.randn(B, T, slow_dim),
    }


class CNN1DLSTMSlowBranchTests(unittest.TestCase):
    def test_registry_exposes_model(self):
        self.assertIn("cnn1d_lstm_fusion_slow_branch", MODEL_REGISTRY)

    def test_slow_encoder_enabled_uses_192_input_channels(self):
        model = build_model(
            {
                "name": "cnn1d_lstm_fusion_slow_branch",
                "slow_encoder": {
                    "enabled": True,
                    "hidden_dim": 32,
                    "embedding_dim": 64,
                },
            }
        )
        self.assertEqual(model.lstm.input_size, 64 + 64 + 64)

    def test_slow_encoder_disabled_keeps_legacy_input_channels(self):
        model = build_model(
            {
                "name": "cnn1d_lstm_fusion_slow_branch",
                "slow_encoder": {
                    "enabled": False,
                    "hidden_dim": 32,
                    "embedding_dim": 64,
                },
            }
        )
        self.assertEqual(model.lstm.input_size, 64 + 64 + 8)

    def test_target_specific_heads_support_out_dim_4(self):
        batch = _make_batch()
        model = build_model(
            {
                "name": "cnn1d_lstm_fusion_slow_branch",
                "slow_encoder": {
                    "enabled": True,
                    "hidden_dim": 32,
                    "embedding_dim": 64,
                },
                "derive_last": True,
                "derive_last_mode": "bounded_simplex",
            }
        )
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
        self.assertTrue(torch.all(out >= 0))
        self.assertTrue(torch.allclose(out.sum(dim=-1), torch.full((4,), 100.0), atol=1e-4))

    def test_target_specific_heads_support_out_dim_3(self):
        batch = _make_batch()
        model = build_model(
            {
                "name": "cnn1d_lstm_fusion_slow_branch",
                "out_dim": 3,
                "slow_encoder": {
                    "enabled": True,
                    "hidden_dim": 32,
                    "embedding_dim": 64,
                },
            }
        )
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
        self.assertTrue(torch.all(out >= 0))
        self.assertTrue(torch.allclose(out.sum(dim=-1), torch.full((4,), 100.0), atol=1e-4))

    def test_ultrasonic_only_branch_with_slow_encoder_uses_128_input_channels(self):
        model = build_model(
            {
                "name": "cnn1d_lstm_fusion_slow_branch",
                "use_fiber_mic": False,
                "slow_encoder": {
                    "enabled": True,
                    "hidden_dim": 32,
                    "embedding_dim": 64,
                },
            }
        )
        self.assertEqual(model.lstm.input_size, 64 + 64)

    def test_fiber_only_branch_with_slow_encoder_uses_128_input_channels(self):
        model = build_model(
            {
                "name": "cnn1d_lstm_fusion_slow_branch",
                "use_ultrasonic": False,
                "slow_encoder": {
                    "enabled": True,
                    "hidden_dim": 32,
                    "embedding_dim": 64,
                },
            }
        )
        self.assertEqual(model.lstm.input_size, 64 + 64)

    def test_yaml_config_smoke_builds_model(self):
        config_path = ROOT / "configs" / "deep" / "bounded_simplex_equal_mse_slow_branch_lstm.yaml"
        self.assertTrue(config_path.exists(), f"配置文件不存在: {config_path}")
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        self.assertEqual(config["model"]["name"], "cnn1d_lstm_fusion_slow_branch")
        model = build_model(config["model"])
        self.assertEqual(model.head.output_dim, 4)
        self.assertTrue(model.head.derive_last)
        self.assertEqual(model.head.derive_last_mode, "bounded_simplex")


if __name__ == "__main__":
    unittest.main()
