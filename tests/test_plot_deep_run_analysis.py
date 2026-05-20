import pathlib
import sys
import tempfile
import unittest

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipeline import plot_deep_run_analysis


class PlotDeepRunAnalysisTests(unittest.TestCase):
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
                "lr": [2e-4, 1.5e-4, 1e-4],
            }
        ).to_csv(run_dir / "train_log.csv", index=False)
        pd.DataFrame(
            {
                "sample_id": ["a", "b", "c"],
                "mixture_id": ["a", "b", "c"],
                "split": ["test", "test", "test"],
                "y_true_H2": [1.0, 2.0, 3.0],
                "y_true_CH4": [80.0, 78.0, 76.0],
                "y_true_CO2": [5.0, 6.0, 7.0],
                "y_true_N2": [14.0, 14.0, 14.0],
                "y_pred_H2": [1.2, 1.8, 3.1],
                "y_pred_CH4": [79.0, 77.5, 75.2],
                "y_pred_CO2": [5.1, 5.8, 7.3],
                "y_pred_N2": [14.7, 14.9, 14.4],
                "sum_true": [100.0, 100.0, 100.0],
                "sum_pred": [100.0, 100.0, 100.0],
                "abs_sum_error": [0.0, 0.0, 0.0],
            }
        ).to_csv(run_dir / "predictions.csv", index=False)
        return run_dir

    def test_generate_run_analysis_artifacts_outputs_expected_files(self) -> None:
        run_dir = self._write_run()
        output_dir = self.tmp_path / "figures"

        result = plot_deep_run_analysis.generate_run_analysis_artifacts(
            run_dir=run_dir,
            output_dir=output_dir,
            formats=("png", "svg"),
            dpi=120,
        )

        self.assertEqual(result["run_dir"], str(run_dir))
        self.assertEqual(result["formats"], ["png", "svg"])
        self.assertEqual(len(result["generated_files"]), 6)
        self.assertTrue((output_dir / "demo_run_training_curves.png").exists())
        self.assertTrue((output_dir / "demo_run_training_curves.svg").exists())
        self.assertTrue((output_dir / "demo_run_prediction_scatter.png").exists())
        self.assertTrue((output_dir / "demo_run_prediction_scatter.svg").exists())
        self.assertTrue((output_dir / "demo_run_error_distributions.png").exists())
        self.assertTrue((output_dir / "demo_run_error_distributions.svg").exists())


if __name__ == "__main__":
    unittest.main()
