import pathlib
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "ml"))
sys.modules.pop("scripts", None)
sys.modules.pop("scripts.report", None)

from patent_model.dataset import PatentDataset
from patent_model.modeling import ModelConfig, TraditionalFusionModel
from scripts.train_patent_model import _predictions_frame, _weights_frame, prepare_training_data, run_training


def _dataset() -> PatentDataset:
    sample_ids = np.array([f"S{i:02d}" for i in range(12)], dtype=object)
    mixture_ids = np.array([f"M{i // 3:02d}" for i in range(12)], dtype=object)
    acoustic = np.column_stack(
        [
            np.linspace(0.1, 1.2, 12),
            np.linspace(1.0, 2.1, 12),
            np.linspace(2.0, 3.1, 12),
        ]
    )
    optical = np.column_stack(
        [
            np.linspace(0.2, 1.3, 12),
            np.linspace(0.8, 1.9, 12),
        ]
    )
    thermal = np.linspace(0.5, 1.6, 12).reshape(-1, 1)
    environment = np.column_stack(
        [
            np.linspace(20.0, 30.0, 12),
            np.linspace(0.1, 0.6, 12),
            np.linspace(30.0, 70.0, 12),
        ]
    )
    targets = np.column_stack(
        [
            np.linspace(5.0, 20.0, 12),
            np.linspace(60.0, 75.0, 12),
            np.linspace(2.0, 8.0, 12),
            np.linspace(33.0, -3.0, 12),
        ]
    )
    metadata = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "mixture_id": mixture_ids,
            "stage_id": ["distance_stage"] * 12,
            "status": ["synthetic_measurement"] * 12,
            "pressure_stage": ["mid"] * 12,
            "distance_stage": ["mid"] * 12,
            "x_N2": targets[:, 3],
            "optical_baseline_drift_ch4": np.zeros(12),
            "optical_baseline_drift_co2": np.zeros(12),
            "thermal_baseline_drift": np.zeros(12),
            "attenuation_alpha": np.full(12, 0.5),
            "ndir_ch4_saturated": np.zeros(12, dtype=int),
            "ndir_co2_saturated": np.zeros(12, dtype=int),
        }
    )
    return PatentDataset(
        sample_ids=sample_ids,
        acoustic=acoustic,
        optical=optical,
        thermal=thermal,
        environment=environment,
        targets=targets,
        component_names=("H2", "CH4", "CO2", "N2"),
        metadata=metadata,
        acoustic_columns=("TOF", "Amp", "f_peak"),
        optical_columns=("V_NDIR_CH4", "V_NDIR_CO2"),
        thermal_columns=("V_TCS",),
        environment_columns=("T_C", "P_MPa", "H_RH"),
    )


class TraditionalModalFusionTests(unittest.TestCase):
    def test_traditional_fusion_model_predicts_modalities_and_fused_outputs(self) -> None:
        dataset = _dataset()
        config = ModelConfig(
            branch_model_type="pls",
            meta_model_type="ridge",
            pls_n_components=2,
            stacking_folds=3,
            n_perturbations=2,
            n_jobs=1,
            include_environment=False,
            random_state=7,
        )
        model = TraditionalFusionModel(config=config, component_names=dataset.component_names).fit(dataset)

        metrics, prediction = model.evaluate(dataset)

        self.assertEqual(prediction.raw.shape, (dataset.n_samples, 4))
        self.assertEqual(prediction.dynamic_weights.shape, (dataset.n_samples, 4, 3))
        self.assertEqual(set(prediction.by_model), {"acoustic", "optical", "thermal", "fused"})
        np.testing.assert_allclose(prediction.dynamic_weights.sum(axis=2), np.ones((dataset.n_samples, 4)))
        self.assertEqual(set(metrics["model"]), {"acoustic", "optical", "thermal", "fused"})
        self.assertEqual(sorted(metrics["component"].unique().tolist()), ["CH4", "CO2", "H2", "N2"])

    def test_prediction_and_weight_frames_use_new_modal_schema(self) -> None:
        dataset = _dataset()
        config = ModelConfig(
            branch_model_type="pls",
            meta_model_type="ridge",
            pls_n_components=2,
            stacking_folds=3,
            n_perturbations=2,
            n_jobs=1,
            include_environment=False,
            random_state=11,
        )
        model = TraditionalFusionModel(config=config, component_names=dataset.component_names).fit(dataset)
        _, prediction = model.evaluate(dataset)

        predictions = _predictions_frame(dataset, prediction)
        weights = _weights_frame(dataset, prediction)

        self.assertIn("acoustic_pred_H2", predictions.columns)
        self.assertIn("optical_pred_CH4", predictions.columns)
        self.assertIn("thermal_pred_CO2", predictions.columns)
        self.assertIn("fused_pred_N2", predictions.columns)
        self.assertNotIn("raw_pred_H2", predictions.columns)
        self.assertEqual(sorted(weights.columns.tolist()), ["acoustic_weight", "component", "optical_weight", "sample_id", "thermal_weight"])

    def test_run_training_writes_four_model_family_metrics(self) -> None:
        dataset = _dataset()
        train_idx = np.arange(8)
        test_idx = np.arange(8, 12)
        prepared = SimpleNamespace(
            data_dir=ROOT,
            feature_profile_name="v3_waveform_dual_channel_four",
            train_original_samples=len(train_idx),
            train=dataset.subset(train_idx),
            test=dataset.subset(test_idx),
        )
        args = SimpleNamespace(
            output_dir=tempfile.mkdtemp(),
            feature_profile="v3_waveform_dual_channel_four",
            component_mode="four",
            mc_env_samples=0,
            mc_env_sigma_t=0.5,
            mc_env_sigma_p=0.005,
            mc_env_sigma_h=1.0,
            metadata_filter="none",
            stage_filter="stable",
            physical_range_filter="none",
            label_closure_filter="none",
            duplicate_filter="none",
            duplicate_per_mixture_limit=3,
            duplicate_filter_seed=42,
            seed=42,
            n_perturbations=2,
            stacking_folds=2,
            perturbation_scale=0.04,
            branch_model_type="xgboost",
            meta_model_type="ridge",
            pls_n_components=2,
            xgb_n_estimators=10,
            xgb_max_depth=2,
            xgb_learning_rate=0.1,
            xgb_device="cpu",
            xgb_n_jobs=1,
            n_jobs=1,
        )
        try:
            summary = run_training(args, prepared_data=prepared)
            metrics = pd.read_csv(pathlib.Path(args.output_dir) / "component_metrics.csv")
            predictions = pd.read_csv(pathlib.Path(args.output_dir) / "predictions.csv")

            self.assertEqual(set(metrics["model"]), {"acoustic", "optical", "thermal", "fused"})
            self.assertIn("fused_macro_RMSE_pp", summary)
            self.assertIn("best_model_by_macro_RMSE_pp", summary)
            self.assertEqual(set(summary["modal_effective_pls_components"]), {"acoustic", "optical", "thermal", "fused"})
            self.assertIn("fused_pred_H2", predictions.columns)
        finally:
            for path in pathlib.Path(args.output_dir).glob("*"):
                if path.is_file():
                    path.unlink()
            pathlib.Path(args.output_dir).rmdir()


if __name__ == "__main__":
    unittest.main()
