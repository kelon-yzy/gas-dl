"""CNN1D-TCN 慢变量分支实验模型测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "dl"))

from models.cnn1d_tcn_fusion_slow_branch import CNN1DTCNSlowBranchRegressor  # noqa: E402
from models.registry import build_model  # noqa: E402


def _make_batch(B: int = 4, T: int = 10, slow_dim: int = 8, ultrasonic_len: int = 1000, fiber_mic_len: int = 2000):
    return {
        "ultrasonic": torch.randint(0, 32768, (B, T, ultrasonic_len), dtype=torch.int16),
        "ultrasonic_scale": torch.rand(B, T),
        "fiber_mic": torch.randint(0, 32768, (B, T, fiber_mic_len), dtype=torch.int16),
        "fiber_mic_scale": torch.rand(B, T),
        "slow": torch.randn(B, T, slow_dim),
    }


class CNN1DTCNSlowBranchTests(unittest.TestCase):
    def test_registry_exposes_model(self):
        from models.registry import MODEL_REGISTRY

        self.assertIn("cnn1d_tcn_fusion_slow_branch", MODEL_REGISTRY)
        self.assertIs(MODEL_REGISTRY["cnn1d_tcn_fusion_slow_branch"], CNN1DTCNSlowBranchRegressor)

    def test_slow_encoder_enabled_uses_192_input_channels(self):
        model = build_model(
            {
                "name": "cnn1d_tcn_fusion_slow_branch",
                "slow_encoder": {
                    "enabled": True,
                    "hidden_dim": 32,
                    "embedding_dim": 64,
                },
            }
        )
        self.assertEqual(model.tcn[0].net[0].conv.in_channels, 64 + 64 + 64)

    def test_slow_encoder_disabled_keeps_legacy_input_channels(self):
        model = build_model(
            {
                "name": "cnn1d_tcn_fusion_slow_branch",
                "slow_encoder": {
                    "enabled": False,
                    "hidden_dim": 32,
                    "embedding_dim": 64,
                },
            }
        )
        self.assertEqual(model.tcn[0].net[0].conv.in_channels, 64 + 64 + 8)

    def test_target_specific_heads_support_out_dim_4(self):
        batch = _make_batch()
        model = build_model(
            {
                "name": "cnn1d_tcn_fusion_slow_branch",
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
        self.assertEqual(out.shape, (4, 4))
        self.assertTrue(torch.all(out >= 0))
        self.assertTrue(torch.allclose(out.sum(dim=-1), torch.full((4,), 100.0), atol=1e-4))

    def test_target_specific_heads_support_out_dim_3(self):
        batch = _make_batch()
        model = build_model(
            {
                "name": "cnn1d_tcn_fusion_slow_branch",
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
                "name": "cnn1d_tcn_fusion_slow_branch",
                "use_fiber_mic": False,
                "slow_encoder": {
                    "enabled": True,
                    "hidden_dim": 32,
                    "embedding_dim": 64,
                },
            }
        )
        self.assertEqual(model.tcn[0].net[0].conv.in_channels, 64 + 64)

    def test_fiber_only_branch_with_slow_encoder_uses_128_input_channels(self):
        model = build_model(
            {
                "name": "cnn1d_tcn_fusion_slow_branch",
                "use_ultrasonic": False,
                "slow_encoder": {
                    "enabled": True,
                    "hidden_dim": 32,
                    "embedding_dim": 64,
                },
            }
        )
        self.assertEqual(model.tcn[0].net[0].conv.in_channels, 64 + 64)

    def test_head_is_softplus_normalize_head(self):
        model = build_model(
            {
                "name": "cnn1d_tcn_fusion_slow_branch",
                "slow_encoder": {
                    "enabled": True,
                    "hidden_dim": 32,
                    "embedding_dim": 64,
                },
            }
        )
        self.assertEqual(model.head.output_dim, 4)
        self.assertEqual(model.head.target_sum, 100.0)


if __name__ == "__main__":
    unittest.main()
