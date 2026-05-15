import pathlib
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "dl"))
sys.path.insert(0, str(ROOT / "src" / "ml"))

from data.dataset_v2 import V2SequenceDataset
from data.dataset_waveform import WaveformSequenceDataset
from patent_model.data_loader import load_patent_dataset
from patent_model.dataset import PatentDataset
from patent_model.fault_labels import inject_faults
from patent_model.modeling import ModelConfig, SingleComponentPatentModel
from patent_model.robustness import add_environment_noise, select_pressure_slice
from training.seed import set_seed
from training.train import _ensure_scaler_path, _forward_batch


def _dataset() -> PatentDataset:
    sample_ids = np.array(["S1", "S2", "S3"], dtype=object)
    metadata = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "mixture_id": ["M1", "M1", "M2"],
        }
    )
    return PatentDataset(
        sample_ids=sample_ids,
        acoustic=np.array([[1.0, 0.5], [2.0, 0.4], [3.0, 0.3]]),
        optical=np.array([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]]),
        thermal=np.array([[0.7], [0.8], [0.9]]),
        environment=np.array([[25.0, 0.10, 40.0], [25.5, 0.20, 45.0], [26.0, 0.30, 50.0]]),
        targets=np.array([[10.0, 80.0, 5.0, 5.0], [11.0, 79.0, 5.0, 5.0], [12.0, 78.0, 5.0, 5.0]]),
        component_names=("H2", "CH4", "CO2", "N2"),
        metadata=metadata,
        acoustic_columns=("TOF", "Amp"),
        optical_columns=("V_NDIR_CH4", "V_NDIR_CO2"),
        thermal_columns=("V_TCS",),
        environment_columns=("T_C", "P_MPa", "H_RH"),
        provenance={"source": "unit-test"},
        filter_report={"metadata_filter": {"kept": 3}},
    )


class CodeReviewFixTests(unittest.TestCase):
    def test_dataset_transformations_preserve_provenance_and_filter_report(self) -> None:
        dataset = _dataset()

        for transformed in (
            inject_faults(dataset, "acoustic_bias", severity="mild", seed=1),
            add_environment_noise(dataset, sigma_t=0.1, sigma_p=0.01, sigma_h=0.2, seed=2),
            select_pressure_slice(dataset, target_pressure_mpa=0.2, max_samples=2),
        ):
            self.assertEqual(transformed.provenance, dataset.provenance)
            self.assertEqual(transformed.filter_report, dataset.filter_report)

    def test_oof_degenerate_path_emits_warning(self) -> None:
        config = ModelConfig(stacking_folds=5, n_perturbations=1, random_state=7)
        model = SingleComponentPatentModel(config)
        inputs = {
            "acoustic": np.arange(15, dtype=float).reshape(5, 3),
            "optical": np.arange(15, 30, dtype=float).reshape(5, 3),
            "thermal": np.arange(30, 45, dtype=float).reshape(5, 3),
        }
        target = np.linspace(0.0, 1.0, 5)
        groups = np.array(["M1", "M1", "M1", "M1", "M1"], dtype=object)

        with self.assertLogs("patent_model.modeling", level="WARNING") as captured:
            model._oof_meta_inputs(inputs, target, groups)

        self.assertTrue(any("OOF fallback" in message for message in captured.output))

    def test_model_forward_uses_declared_input_format(self) -> None:
        class NCTModel(torch.nn.Module):
            input_format = "NCT"

            def __init__(self) -> None:
                super().__init__()
                self.seen_shape = None

            def forward(self, x):
                self.seen_shape = tuple(x.shape)
                return x.mean(dim=(1, 2), keepdim=False).unsqueeze(1)

        model = NCTModel()
        batch = {
            "ultrasonic": torch.zeros((2, 5, 8), dtype=torch.int16),
            "ultrasonic_scale": torch.ones((2, 5), dtype=torch.float32),
            "fiber_mic": torch.zeros((2, 5, 8), dtype=torch.int16),
            "fiber_mic_scale": torch.ones((2, 5), dtype=torch.float32),
            "slow": torch.zeros((2, 5, 3), dtype=torch.float32),
            "target": torch.zeros((2, 1), dtype=torch.float32),
            "meta": {"sample_id": ["S1", "S2"]},
        }
        _forward_batch(model, batch, torch.device("cpu"))

        self.assertEqual(model.seen_shape, (2, 3, 5))

    def test_training_config_gets_default_scaler_path_when_missing(self) -> None:
        data_config = {"dataset_type": "waveform_v3"}
        output_dir = pathlib.Path("outputs") / "unit_run"

        _ensure_scaler_path(data_config, output_dir)

        self.assertEqual(data_config["scaler_path"], str(output_dir / "scaler_slow_sequence.json"))

    def test_v2_dataset_uses_preloaded_data_without_reloading_npz(self) -> None:
        data = {
            "X": np.arange(2 * 4 * 12, dtype=np.float32).reshape(2, 4, 12),
            "y": np.arange(2 * 4, dtype=np.float32).reshape(2, 4),
            "sequence_ids": np.array(["Q1", "Q2"], dtype=object),
            "channel_names": np.array(
                [
                    "V_NDIR_CH4",
                    "V_NDIR_CO2",
                    "V_TCS",
                    "T_C",
                    "P_MPa",
                    "H_RH",
                    "L_m",
                    "piston_position_m",
                    "TOF",
                    "Amp",
                    "f_peak",
                    "A_fft_max",
                ],
                dtype=object,
            ),
            "label_names": np.array(["x_H2", "x_CH4", "x_CO2", "x_N2"], dtype=object),
        }
        dataset = V2SequenceDataset("missing.npz", indices=[0], index_path=None, preloaded_data=data)

        with patch("data.dataset_v2.load_v2_npz", side_effect=AssertionError("unexpected reload")):
            sample = dataset[0]

        self.assertEqual(tuple(sample[0].shape), (4, 12))
        self.assertEqual(sample[2]["sample_id"], "Q1")

    def test_waveform_dataset_uses_preloaded_data_without_reloading_package(self) -> None:
        data = {
            "ultrasonic": np.zeros((1, 2, 8), dtype=np.int16),
            "ultrasonic_scale": np.ones((1, 2), dtype=np.float32),
            "fiber_mic": np.zeros((1, 2, 10), dtype=np.int16),
            "fiber_mic_scale": np.ones((1, 2), dtype=np.float32),
            "slow": np.zeros((1, 2, 8), dtype=np.float32),
            "y": np.zeros((1, 4), dtype=np.float32),
            "sequence_ids": np.array(["Q1"], dtype=object),
            "slow_channel_names": np.array(
                ["V_NDIR_CH4", "V_NDIR_CO2", "V_TCS", "T_C", "P_MPa", "H_RH", "L_m", "piston_position_m"],
                dtype=object,
            ),
            "label_names": np.array(["x_H2", "x_CH4", "x_CO2", "x_N2"], dtype=object),
        }
        dataset = WaveformSequenceDataset("missing", indices=[0], index_path=None, preloaded_data=data)

        with patch("data.dataset_waveform.load_waveform_package", side_effect=AssertionError("unexpected reload")):
            sample = dataset[0]

        self.assertEqual(tuple(sample["slow"].shape), (2, 8))
        self.assertEqual(sample["meta"]["sample_id"], "Q1")

    def test_four_component_loader_reports_missing_condition_n2_column(self) -> None:
        tmp = tempfile.mkdtemp()
        try:
            base = pathlib.Path(tmp)
            (base / "training").mkdir()
            (base / "labels").mkdir()
            (base / "features").mkdir()
            pd.DataFrame(
                {
                    "sample_id": ["S1"],
                    "TOF": [1.0],
                    "Amp": [1.0],
                    "f_peak": [1.0],
                    "A_fft_max": [1.0],
                    "L_m": [1.0],
                    "T_C": [25.0],
                    "P_MPa": [0.1],
                    "H_RH": [40.0],
                }
            ).to_csv(base / "training" / "train_acoustic.csv", index=False)
            pd.DataFrame({"sample_id": ["S1"], "V_NDIR_CH4": [1.0], "V_NDIR_CO2": [1.0], "delta_I_CH4": [0.1], "delta_I_CO2": [0.1]}).to_csv(
                base / "training" / "train_optical.csv",
                index=False,
            )
            pd.DataFrame({"sample_id": ["S1"], "V_TCS": [1.0]}).to_csv(base / "training" / "train_thermal.csv", index=False)
            pd.DataFrame({"sample_id": ["S1"], "x_H2": [10.0], "x_CH4": [80.0], "x_CO2": [5.0]}).to_csv(
                base / "labels" / "labels.csv",
                index=False,
            )
            pd.DataFrame(
                {
                    "sample_id": ["S1"],
                    "mixture_id": ["M1"],
                    "stage_id": ["detection"],
                    "repeat_id": [1],
                    "status": ["synthetic_measurement"],
                    "pressure_stage": ["low"],
                    "distance_stage": ["short"],
                    "piston_position_m": [0.0],
                    "T_C": [25.0],
                    "P_MPa": [0.1],
                    "H_RH": [40.0],
                }
            ).to_csv(base / "condition_grid_v1.csv", index=False)
            pd.DataFrame(
                {
                    "sample_id": ["S1"],
                    "sound_speed": [340.0],
                    "attenuation_alpha": [0.1],
                    "ndir_ch4_saturated": [0],
                    "ndir_co2_saturated": [0],
                    "optical_baseline_drift_ch4": [0.0],
                    "optical_baseline_drift_co2": [0.0],
                    "thermal_baseline_drift": [0.0],
                    "lambda_mix_calibrated": [0.1],
                    "calibration_status": ["ok"],
                }
            ).to_csv(base / "features" / "feature_table.csv", index=False)

            with self.assertRaisesRegex(ValueError, "condition_grid_v1.csv.*x_N2"):
                load_patent_dataset(base, profile="raw_no_env_four")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_seed_sets_cudnn_determinism_flags(self) -> None:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

        set_seed(123)

        self.assertTrue(torch.backends.cudnn.deterministic)
        self.assertFalse(torch.backends.cudnn.benchmark)


if __name__ == "__main__":
    unittest.main()
