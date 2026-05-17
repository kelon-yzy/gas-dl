"""测试 6 个 MultimodalWrapper 融合模型的前向传播和参数分离。"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "dl"))

import torch
from models.registry import build_model
from models.multimodal_wrapper import _build_multimodal, _BACKBONE_KWARGS
from models.cnn1d import CNN1DRegressor
from models.cnn_lstm import CNNLSTMRegressor
from models.gru import GRURegressor
from models.lstm import LSTMRegressor
from models.tcn import TCNRegressor
from models.transformer_encoder import TransformerRegressor


class MultimodalWrapperModelTests(unittest.TestCase):
    """6 个 _multimodal 模型用同一个合成 batch 做 forward，断言输出都是 (B, 4)。"""

    def _make_batch(self, B: int = 4, T: int = 10, slow_dim: int = 8, ultrasonic_len: int = 1000, fiber_mic_len: int = 2000):
        """构造合成 waveform_v3 批次数据。"""
        return {
            "ultrasonic": torch.randint(0, 32768, (B, T, ultrasonic_len), dtype=torch.int16),
            "ultrasonic_scale": torch.rand(B, T),
            "fiber_mic": torch.randint(0, 32768, (B, T, fiber_mic_len), dtype=torch.int16),
            "fiber_mic_scale": torch.rand(B, T),
            "slow": torch.randn(B, T, slow_dim),
        }

    def _forward_model(self, model_name: str, batch: dict) -> torch.Tensor:
        cfg = {
            "name": model_name,
            "slow_dim": 8,
            "out_dim": 4,
        }
        if model_name == "cnn1d_multimodal":
            cfg["hidden_channels"] = [32, 64, 64]
        elif model_name == "tcn_multimodal":
            cfg["channels"] = [64, 64, 64]
        elif model_name == "transformer_multimodal":
            cfg["d_model"] = 64
            cfg["nhead"] = 4
            cfg["num_layers"] = 2
            cfg["dim_feedforward"] = 128
        elif model_name == "cnn_lstm_multimodal":
            cfg["conv_channels"] = 64
            cfg["hidden_size"] = 64
        elif model_name in ("gru_multimodal", "lstm_multimodal"):
            cfg["hidden_size"] = 64

        model = build_model(cfg)
        model.eval()
        with torch.no_grad():
            return model(
                batch["ultrasonic"],
                batch["ultrasonic_scale"],
                batch["fiber_mic"],
                batch["fiber_mic_scale"],
                batch["slow"],
            )

    def test_cnn1d_multimodal_output_shape(self):
        """CNN1D multimodal 输出维度为 (B, 4)。"""
        batch = self._make_batch()
        out = self._forward_model("cnn1d_multimodal", batch)
        self.assertEqual(out.shape, (4, 4), f"cnn1d_multimodal 输出形状错误: {out.shape}")

    def test_gru_multimodal_output_shape(self):
        batch = self._make_batch()
        out = self._forward_model("gru_multimodal", batch)
        self.assertEqual(out.shape, (4, 4), f"gru_multimodal 输出形状错误: {out.shape}")

    def test_lstm_multimodal_output_shape(self):
        batch = self._make_batch()
        out = self._forward_model("lstm_multimodal", batch)
        self.assertEqual(out.shape, (4, 4), f"lstm_multimodal 输出形状错误: {out.shape}")

    def test_tcn_multimodal_output_shape(self):
        batch = self._make_batch()
        out = self._forward_model("tcn_multimodal", batch)
        self.assertEqual(out.shape, (4, 4), f"tcn_multimodal 输出形状错误: {out.shape}")

    def test_transformer_multimodal_output_shape(self):
        batch = self._make_batch()
        out = self._forward_model("transformer_multimodal", batch)
        self.assertEqual(out.shape, (4, 4), f"transformer_multimodal 输出形状错误: {out.shape}")

    def test_cnn_lstm_multimodal_output_shape(self):
        """CNN-LSTM multimodal 输出维度为 (B, 4)。"""
        batch = self._make_batch()
        out = self._forward_model("cnn_lstm_multimodal", batch)
        self.assertEqual(out.shape, (4, 4), f"cnn_lstm_multimodal 输出形状错误: {out.shape}")

    def test_cnn1d_multimodal_input_format_is_NCT(self):
        """CNN1D backbone 的 input_format 必须是 NCT，确保卷积类不出现维度错位。"""
        self.assertEqual(CNN1DRegressor.input_format, "NCT")

    def test_cnn_lstm_multimodal_input_format_is_NCT(self):
        self.assertEqual(CNNLSTMRegressor.input_format, "NCT")

    def test_gru_input_format_is_NTC(self):
        self.assertEqual(GRURegressor.input_format, "NTC")

    def test_lstm_input_format_is_NTC(self):
        self.assertEqual(LSTMRegressor.input_format, "NTC")

    def test_transformer_input_format_is_NTC(self):
        self.assertEqual(TransformerRegressor.input_format, "NTC")

    def test_tcn_input_format_is_NCT(self):
        self.assertEqual(TCNRegressor.input_format, "NCT")

    def test_ultrasonic_only_branch(self):
        """use_ultrasonic=True, use_fiber_mic=False 能正常前向。"""
        cfg = {"name": "gru_multimodal", "slow_dim": 8, "hidden_size": 64, "out_dim": 4, "use_fiber_mic": False}
        model = build_model(cfg)
        model.eval()
        batch = self._make_batch()
        with torch.no_grad():
            out = model(
                batch["ultrasonic"], batch["ultrasonic_scale"],
                None, None,
                batch["slow"],
            )
        self.assertEqual(out.shape, (4, 4))

    def test_fiber_mic_only_branch(self):
        """use_ultrasonic=False, use_fiber_mic=True 能正常前向。"""
        cfg = {"name": "gru_multimodal", "slow_dim": 8, "hidden_size": 64, "out_dim": 4, "use_ultrasonic": False}
        model = build_model(cfg)
        model.eval()
        batch = self._make_batch()
        with torch.no_grad():
            out = model(
                None, None,
                batch["fiber_mic"], batch["fiber_mic_scale"],
                batch["slow"],
            )
        self.assertEqual(out.shape, (4, 4))

    def test_no_branch_raises_value_error(self):
        """use_ultrasonic=False 且 use_fiber_mic=False 必须抛 ValueError。"""
        with self.assertRaises(ValueError):
            build_model({"name": "gru_multimodal", "slow_dim": 8, "hidden_size": 64, "out_dim": 4, "use_ultrasonic": False, "use_fiber_mic": False})

    def test_wrapper_params_not_leaked_to_backbone(self):
        """确保 wrapper 参数不会透传到 backbone 构造函数。"""
        model = _build_multimodal(GRURegressor, slow_dim=8, hidden_size=64, out_dim=4, use_ultrasonic=True, use_fiber_mic=True)
        # fused_dim = 8 + 64*2 = 136 被设置为 input_size
        self.assertEqual(model.backbone.gru.input_size, 136)
        # slow_dim 被正确记录
        self.assertEqual(model.slow_dim, 8)

    def test_fused_dim_single_branch(self):
        """单分支时 fused_dim = slow_dim + waveform_embedding_dim * 1。"""
        model = _build_multimodal(CNN1DRegressor, slow_dim=8, hidden_channels=[32, 64], out_dim=4, use_fiber_mic=False)
        # fused_dim = 8 + 64*1 = 72
        self.assertEqual(model.backbone.encoder[0].in_channels, 72)

    def test_unknown_wrapper_or_backbone_kwargs_raise_value_error(self):
        """未知配置项不能被静默忽略，否则实验配置会悄悄退回默认值。"""
        with self.assertRaisesRegex(ValueError, "dropouut"):
            build_model(
                {
                    "name": "cnn1d_multimodal",
                    "slow_dim": 8,
                    "hidden_channels": [32, 64, 64],
                    "out_dim": 4,
                    "dropouut": 0.1,
                }
            )


if __name__ == "__main__":
    unittest.main()
