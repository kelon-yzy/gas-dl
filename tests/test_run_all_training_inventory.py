"""校验 run_all_training.ps1 训练名单包含 6 个 _multimodal 配置、不包含已移除的条目。"""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PS1_PATH = ROOT / "run_all_training.ps1"


class RunAllTrainingInventoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.content = PS1_PATH.read_text(encoding="utf-8")

    def _extract_configs(self):
        return re.findall(r'Config\s*=\s*"([^"]+)"', self.content)

    def test_contains_six_multimodal_configs(self):
        """训练名单必须包含 6 个 _multimodal 配置。"""
        configs = self._extract_configs()
        multimodal = [c for c in configs if "_multimodal" in c]
        self.assertEqual(len(multimodal), 6, f"应有 6 个 _multimodal 配置，实际找到: {multimodal}")

    def test_multimodal_configs_in_order(self):
        """6 个 _multimodal 配置的顺序固定：cnn1d, gru, lstm, tcn, transformer, cnn_lstm。"""
        configs = self._extract_configs()
        multimodal = [c for c in configs if "_multimodal" in c]
        expected_order = [
            "configs/deep/slow_only_cnn1d_multimodal_formal.yaml",
            "configs/deep/slow_only_gru_multimodal_formal.yaml",
            "configs/deep/slow_only_lstm_multimodal_formal.yaml",
            "configs/deep/slow_only_tcn_multimodal_formal.yaml",
            "configs/deep/slow_only_transformer_multimodal_formal.yaml",
            "configs/deep/slow_only_cnn_lstm_multimodal_formal.yaml",
        ]
        self.assertEqual(multimodal, expected_order, f"multimodal 配置顺序不符: {multimodal}")

    def test_does_not_contain_removed_cnn1d_slow_only(self):
        """训练名单不应包含 slow_only_cnn1d_formal.yaml。"""
        configs = self._extract_configs()
        self.assertNotIn("configs/deep/slow_only_cnn1d_formal.yaml", configs, "slow_only_cnn1d_formal.yaml 应已移除")

    def test_does_not_contain_removed_cnn_lstm_slow_only(self):
        """训练名单不应包含 slow_only_cnn_lstm_formal.yaml。"""
        configs = self._extract_configs()
        self.assertNotIn("configs/deep/slow_only_cnn_lstm_formal.yaml", configs, "slow_only_cnn_lstm_formal.yaml 应已移除")

    def test_does_not_contain_removed_fusion_formal(self):
        """训练名单不应包含 fusion_formal.yaml。"""
        configs = self._extract_configs()
        self.assertNotIn("configs/deep/fusion_formal.yaml", configs, "fusion_formal.yaml 应已从训练名单移除")

    def test_contains_gru_slow_only(self):
        configs = self._extract_configs()
        self.assertIn("configs/deep/slow_only_gru_formal.yaml", configs, "GRU 纯慢变量配置缺失")

    def test_contains_lstm_slow_only(self):
        configs = self._extract_configs()
        self.assertIn("configs/deep/slow_only_lstm_formal.yaml", configs, "LSTM 纯慢变量配置缺失")

    def test_contains_tcn_slow_only(self):
        configs = self._extract_configs()
        self.assertIn("configs/deep/slow_only_tcn_formal.yaml", configs, "TCN 纯慢变量配置缺失")

    def test_contains_transformer_slow_only(self):
        configs = self._extract_configs()
        self.assertIn("configs/deep/slow_only_transformer_formal.yaml", configs, "Transformer 纯慢变量配置缺失")

    def test_contains_branch_fusion_slow_only(self):
        configs = self._extract_configs()
        self.assertIn("configs/deep/slow_only_branch_fusion_formal.yaml", configs, "BranchFusion 纯慢变量配置缺失")

    def test_contains_waveform_only(self):
        configs = self._extract_configs()
        self.assertIn("configs/deep/waveform_only_formal.yaml", configs, "MultimodalFusionV3 纯多模态配置缺失")


if __name__ == "__main__":
    unittest.main()