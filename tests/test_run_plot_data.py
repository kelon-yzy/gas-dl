import json
import pathlib
import sys
import tempfile
import unittest

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipeline import run_plot_data


class RunPlotDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp_path = pathlib.Path(self._tmp.name)

    def _write_run(self) -> pathlib.Path:
        run_dir = self.tmp_path / "outputs" / "exp02" / "demo_run"
        run_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "epoch": [1, 2, 3],
                "train_loss": [5.0, 3.0, 2.0],
                "val_loss": [4.5, 2.8, 2.2],
                "val_macro_RMSE": [2.6, 1.8, 1.9],
                "val_macro_MAE": [2.1, 1.4, 1.5],
                "val_mean_pred_sum": [100.2, 100.0, 99.9],
                "val_mean_abs_sum_error": [0.4, 0.1, 0.2],
                "val_std_pred_sum": [0.3, 0.05, 0.09],
            }
        ).to_csv(run_dir / "train_log.csv", index=False)
        pd.DataFrame(
            {
                "sample_id": ["a", "b", "c"],
                "mixture_id": ["a", "b", "c"],
                "split": ["test", "test", "test"],
                "y_true_CO2": [5.0, 6.0, 7.0],
                "y_true_H2": [1.0, 2.0, 3.0],
                "y_true_N2": [14.0, 14.0, 14.0],
                "y_true_CH4": [80.0, 78.0, 76.0],
                "y_pred_CO2": [5.1, 5.8, 7.3],
                "y_pred_H2": [1.2, 1.8, 3.1],
                "y_pred_N2": [14.7, 14.9, 14.4],
                "y_pred_CH4": [79.0, 77.5, 75.2],
                "sum_true": [100.0, 100.0, 100.0],
                "sum_pred": [100.0, 100.0, 100.0],
                "abs_sum_error": [0.0, 0.0, 0.0],
            }
        ).to_csv(run_dir / "predictions.csv", index=False)
        pd.DataFrame(
            {
                "component": ["CO2", "H2", "N2", "CH4"],
                "RMSE": [0.1, 0.2, 0.3, 0.4],
                "MAE": [0.1, 0.2, 0.3, 0.4],
                "R2": [0.99, 0.98, 0.97, 0.96],
            }
        ).to_csv(run_dir / "component_metrics.csv", index=False)
        (run_dir / "summary.json").write_text(
            json.dumps(
                {
                    "run_name": "demo_run",
                    "model": "cnn1d_tcn_fusion_slow_branch",
                    "macro_RMSE": 1.8,
                    "macro_MAE": 1.4,
                    "mean_abs_sum_error": 0.1,
                    "label_names": ["x_CO2", "x_H2", "x_N2", "x_CH4"],
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "config.json").write_text(
            json.dumps({"model": {"name": "cnn1d_tcn_fusion_slow_branch"}}),
            encoding="utf-8",
        )
        return run_dir

    def test_load_run_analysis_bundle_uses_prediction_column_order(self) -> None:
        bundle = run_plot_data.load_run_analysis_bundle(self._write_run(), require_report_files=True)

        self.assertEqual(bundle.components, ("CO2", "H2", "N2", "CH4"))
        self.assertEqual(list(bundle.component_metrics["component"]), ["CO2", "H2", "N2", "CH4"])
        self.assertEqual(bundle.best_epoch, 2)
        self.assertAlmostEqual(bundle.best_val_macro_rmse, 1.8)


if __name__ == "__main__":
    unittest.main()
