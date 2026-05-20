import pathlib
import sys
import unittest
from unittest import mock

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipeline import train_deep


class TrainDeepPathTests(unittest.TestCase):
    def test_waveform_only_config_uses_project_data_root(self) -> None:
        config_path = ROOT / "configs" / "deep" / "waveform_only_formal.yaml"
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)

        self.assertEqual(config["data"]["npz_path"], "../../data/waveform_v3")
        self.assertEqual(config["data"]["index_path"], "../../data/waveform_v3/sequence_index.csv")
        self.assertEqual(config["data"]["split_dir"], "../../data/waveform_v3/splits")
        self.assertEqual(config["data"]["scaler_path"], "../../data/waveform_v3/scalers/scaler_slow_sequence.json")

    def test_train_deep_normalizes_default_output_dir_to_project_root(self) -> None:
        captured = {}
        config_path = ROOT / "configs" / "deep" / "fusion_formal.yaml"
        expected_output_dir = str((ROOT / "outputs" / "exp02_deep_e2e" / "v3_multimodal_fusion_seed42").resolve())

        def fake_train_config(config, epochs_override=None, resume_path=None,
                              checkpoint_every=0, restore_rng=True, stop_after_epoch=None):
            captured["config"] = config
            captured["epochs_override"] = epochs_override
            return {"macro_RMSE": 0.0}

        argv = [
            "train_deep.py",
            "--config",
            str(config_path),
            "--epochs",
            "1",
            "--no-ui",
        ]

        with mock.patch.object(train_deep, "build_cli_progress", return_value=None), mock.patch.object(
            train_deep, "train_config", side_effect=fake_train_config
        ), mock.patch.object(sys, "argv", argv):
            train_deep.main()

        self.assertEqual(captured["epochs_override"], 1)
        self.assertEqual(captured["config"]["run"]["output_dir"], expected_output_dir)

    def test_multimodal_single_run_scripts_exist_and_point_to_formal_configs(self) -> None:
        expected = {
            "run_gru_multimodal_formal.ps1": "configs/deep/slow_only_gru_multimodal_formal.yaml",
            "run_lstm_multimodal_formal.ps1": "configs/deep/slow_only_lstm_multimodal_formal.yaml",
            "run_tcn_multimodal_formal.ps1": "configs/deep/slow_only_tcn_multimodal_formal.yaml",
        }
        for script_name, config_path in expected.items():
            script_path = ROOT / script_name
            self.assertTrue(script_path.exists(), f"脚本不存在: {script_name}")
            content = script_path.read_text(encoding="utf-8")
            self.assertIn(config_path, content, f"{script_name} 未绑定到期望配置 {config_path}")
            self.assertIn("--ui", content, f"{script_name} 应默认请求显示 UI")
            self.assertNotIn("--no-ui", content, f"{script_name} 不应再硬编码禁用 UI")


class MultimodalConfigTests(unittest.TestCase):
    """校验 6 个 _multimodal_formal.yaml 存在且字段符合约定。"""

    MULTIMODAL_CONFIGS = [
        {
            "file": "slow_only_cnn1d_multimodal_formal.yaml",
            "model_name": "cnn1d_multimodal",
            "run_name": "v3_cnn1d_multimodal_seed42",
            "output_dir": "outputs/exp02_deep_e2e/v3_cnn1d_multimodal_seed42",
            "batch_size": 32,
        },
        {
            "file": "slow_only_gru_multimodal_formal.yaml",
            "model_name": "gru_multimodal",
            "run_name": "v3_gru_multimodal_seed42",
            "output_dir": "outputs/exp02_deep_e2e/v3_gru_multimodal_seed42",
            "batch_size": 32,
        },
        {
            "file": "slow_only_lstm_multimodal_formal.yaml",
            "model_name": "lstm_multimodal",
            "run_name": "v3_lstm_multimodal_seed42",
            "output_dir": "outputs/exp02_deep_e2e/v3_lstm_multimodal_seed42",
            "batch_size": 32,
        },
        {
            "file": "slow_only_tcn_multimodal_formal.yaml",
            "model_name": "tcn_multimodal",
            "run_name": "v3_tcn_multimodal_seed42",
            "output_dir": "outputs/exp02_deep_e2e/v3_tcn_multimodal_seed42",
            "batch_size": 32,
        },
        {
            "file": "slow_only_transformer_multimodal_formal.yaml",
            "model_name": "transformer_multimodal",
            "run_name": "v3_transformer_multimodal_seed42",
            "output_dir": "outputs/exp02_deep_e2e/v3_transformer_multimodal_seed42",
            "batch_size": 8,
        },
        {
            "file": "slow_only_cnn_lstm_multimodal_formal.yaml",
            "model_name": "cnn_lstm_multimodal",
            "run_name": "v3_cnn_lstm_multimodal_seed42",
            "output_dir": "outputs/exp02_deep_e2e/v3_cnn_lstm_multimodal_seed42",
            "batch_size": 8,
        },
    ]

    def test_multimodal_configs_exist(self):
        for cfg in self.MULTIMODAL_CONFIGS:
            path = ROOT / "configs" / "deep" / cfg["file"]
            self.assertTrue(path.exists(), f"配置文件不存在: {cfg['file']}")

    def test_multimodal_configs_have_correct_model_name(self):
        for entry in self.MULTIMODAL_CONFIGS:
            path = ROOT / "configs" / "deep" / entry["file"]
            with path.open("r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            self.assertEqual(config["model"]["name"], entry["model_name"], f"{entry['file']} model.name 错误")

    def test_multimodal_configs_have_correct_run_name(self):
        for entry in self.MULTIMODAL_CONFIGS:
            path = ROOT / "configs" / "deep" / entry["file"]
            with path.open("r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            self.assertEqual(config["run"]["name"], entry["run_name"], f"{entry['file']} run.name 错误")

    def test_multimodal_configs_have_correct_output_dir(self):
        for entry in self.MULTIMODAL_CONFIGS:
            path = ROOT / "configs" / "deep" / entry["file"]
            with path.open("r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            self.assertEqual(config["run"]["output_dir"], entry["output_dir"], f"{entry['file']} run.output_dir 错误")

    def test_multimodal_configs_have_slow_dim_and_out_dim(self):
        for entry in self.MULTIMODAL_CONFIGS:
            path = ROOT / "configs" / "deep" / entry["file"]
            with path.open("r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            self.assertEqual(config["model"]["slow_dim"], 8, f"{entry['file']} slow_dim 应为 8")
            self.assertEqual(config["model"]["out_dim"], 4, f"{entry['file']} out_dim 应为 4")

    def test_multimodal_configs_have_expected_batch_size(self):
        for entry in self.MULTIMODAL_CONFIGS:
            path = ROOT / "configs" / "deep" / entry["file"]
            with path.open("r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            self.assertEqual(config["training"]["batch_size"], entry["batch_size"], f"{entry['file']} batch_size 应为 {entry['batch_size']}")

    def test_multimodal_configs_data_root_consistent(self):
        for entry in self.MULTIMODAL_CONFIGS:
            path = ROOT / "configs" / "deep" / entry["file"]
            with path.open("r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            self.assertEqual(config["data"]["dataset_type"], "waveform_v3")
            self.assertEqual(config["data"]["npz_path"], "../../data/waveform_v3")
            self.assertEqual(config["data"]["index_path"], "../../data/waveform_v3/sequence_index.csv")
            self.assertEqual(config["data"]["split_dir"], "../../data/waveform_v3/splits")


if __name__ == "__main__":
    unittest.main()
