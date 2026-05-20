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

    def test_cnn1d_multimodal_defaults_to_mean_max_temporal_pooling(self):
        """cnn1d_multimodal 默认开启 mean_max 末端池化，保留跨时间步峰值响应。"""
        model = build_model({"name": "cnn1d_multimodal", "slow_dim": 8, "hidden_channels": [32, 64, 64], "out_dim": 4})
        self.assertEqual(model.backbone.temporal_pooling, "mean_max")

    def test_plain_cnn1d_defaults_to_avg_temporal_pooling(self):
        """纯慢变量 cnn1d 保持原有 avg 默认值，不受多模态改造影响。"""
        model = build_model({"name": "cnn1d", "in_channels": 12, "hidden_channels": [32, 64, 64], "out_dim": 4})
        self.assertEqual(model.temporal_pooling, "avg")

    def test_cnn1d_multimodal_explicit_avg_temporal_pooling_forward(self):
        """显式覆盖为 avg 时，多模态 CNN1D 仍可正常前向。"""
        batch = self._make_batch()
        model = build_model(
            {
                "name": "cnn1d_multimodal",
                "slow_dim": 8,
                "hidden_channels": [32, 64, 64],
                "out_dim": 4,
                "temporal_pooling": "avg",
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

    def test_invalid_temporal_pooling_raises_value_error(self):
        """非法池化策略必须报错，避免实验配置被静默接受。"""
        with self.assertRaisesRegex(ValueError, "median"):
            build_model(
                {
                    "name": "cnn1d_multimodal",
                    "slow_dim": 8,
                    "hidden_channels": [32, 64, 64],
                    "out_dim": 4,
                    "temporal_pooling": "median",
                }
            )

    def test_mean_max_temporal_pooling_doubles_head_input_dim(self):
        """mean_max 时 head 输入维度应翻倍，避免 pooled feature 静默错配。"""
        model = build_model(
            {
                "name": "cnn1d_multimodal",
                "slow_dim": 8,
                "hidden_channels": [32, 64],
                "out_dim": 4,
            }
        )
        self.assertEqual(model.backbone.head[0].in_features, 128)

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

    def test_cnn1d_multimodal_fused_dim_without_stage_one_hot(self):
        """CNN1D 双分支 fused_dim = slow_dim + waveform_embedding_dim * 2 = 136。"""
        model = _build_multimodal(CNN1DRegressor, slow_dim=8, hidden_channels=[32, 64], out_dim=4)
        # fused_dim = 8 + 64*2 = 136
        self.assertEqual(model.backbone.encoder[0].in_channels, 136)

    def test_waveform_batch_no_stage_one_hot_key(self):
        """waveform 正式数据流不再包含 stage_one_hot 字段。"""
        batch = self._make_batch()
        self.assertNotIn("stage_one_hot", batch)
        # 确认前向不需要 stage_one_hot
        model = build_model({"name": "cnn1d_multimodal", "slow_dim": 8, "hidden_channels": [32, 64, 64], "out_dim": 4})
        model.eval()
        with torch.no_grad():
            out = model(
                batch["ultrasonic"], batch["ultrasonic_scale"],
                batch["fiber_mic"], batch["fiber_mic_scale"],
                batch["slow"],
            )
        self.assertEqual(out.shape, (4, 4))

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

    def test_stage_one_hot_config_raises_value_error(self):
        """use_stage_one_hot 配置项已移除，传入应报未知配置项错误。"""
        with self.assertRaisesRegex(ValueError, "use_stage_one_hot"):
            build_model({
                "name": "cnn1d_multimodal",
                "slow_dim": 8,
                "hidden_channels": [32, 64, 64],
                "out_dim": 4,
                "use_stage_one_hot": True,
            })

    def test_waveform_dropout_propagated_to_encoder(self):
        """waveform_dropout 应传入 AcousticWaveformEncoder，控制编码器 Dropout 概率。"""
        cfg_dropout = {"name": "gru_multimodal", "slow_dim": 8, "hidden_size": 64, "out_dim": 4, "waveform_dropout": 0.5}
        model = build_model(cfg_dropout)
        u_enc = model.ultrasonic_encoder
        f_enc = model.fiber_mic_encoder
        self.assertIsNotNone(u_enc)
        self.assertIsNotNone(f_enc)
        # 编码器 projection 层包含 Dropout，检查概率值
        for name, module in u_enc.named_modules():
            if isinstance(module, torch.nn.Dropout):
                self.assertEqual(module.p, 0.5, f"ultrasonic_encoder {name} 期望 dropout=0.5，实际 {module.p}")
        for name, module in f_enc.named_modules():
            if isinstance(module, torch.nn.Dropout):
                self.assertEqual(module.p, 0.5, f"fiber_mic_encoder {name} 期望 dropout=0.5，实际 {module.p}")

    def test_waveform_dropout_default_is_0pt1(self):
        """不指定 waveform_dropout 时默认为 0.1（与 AcousticWaveformEncoder 默认值一致）。"""
        model = build_model({"name": "gru_multimodal", "slow_dim": 8, "hidden_size": 64, "out_dim": 4})
        u_enc = model.ultrasonic_encoder
        for name, module in u_enc.named_modules():
            if isinstance(module, torch.nn.Dropout):
                self.assertAlmostEqual(module.p, 0.1, places=4)

    def test_waveform_dropout_zero_disables_encoder_dropout(self):
        """waveform_dropout=0.0 时编码器不应有 Dropout 效果。"""
        cfg = {"name": "gru_multimodal", "slow_dim": 8, "hidden_size": 64, "out_dim": 4, "waveform_dropout": 0.0}
        model = build_model(cfg)
        for name, module in model.ultrasonic_encoder.named_modules():
            if isinstance(module, torch.nn.Dropout):
                self.assertEqual(module.p, 0.0, f"期望 dropout=0.0，实际 {module.p}")


if __name__ == "__main__":
    unittest.main()
