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


if __name__ == "__main__":
    unittest.main()
