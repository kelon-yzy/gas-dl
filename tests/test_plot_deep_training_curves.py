import json
import pathlib
import sys
import tempfile
import unittest

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipeline import plot_deep_training_curves


class PlotDeepTrainingCurvesTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp_path = pathlib.Path(self._tmp.name)

    def _write_run(
        self,
        root: pathlib.Path,
        relative_dir: str,
        *,
        include_lr: bool = True,
        include_sum_metrics: bool = False,
    ) -> pathlib.Path:
        run_dir = root / relative_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        frame_data = {
            "epoch": [1, 2, 3],
            "train_loss": [1.4, 1.0, 0.8],
            "val_loss": [1.2, 0.9, 0.7],
            "val_macro_RMSE": [1.1, 0.8, 0.9],
            "val_macro_MAE": [0.9, 0.7, 0.8],
            "improved": [True, True, False],
            "lr": [3e-4, 2e-4, 1e-4],
        }
        if include_sum_metrics:
            frame_data.update(
                {
                    "val_mean_pred_sum": [99.8, 100.0, 100.1],
                    "val_mean_abs_sum_error": [0.30, 0.10, 0.15],
                    "val_std_pred_sum": [0.20, 0.05, 0.08],
                }
            )
        frame = pd.DataFrame(frame_data)
        if not include_lr:
            frame = frame.drop(columns=["lr"])
        frame.to_csv(run_dir / "train_log.csv", index=False)
        (run_dir / "summary.json").write_text(
            json.dumps({"run_name": run_dir.name, "model": "tcn_multimodal", "macro_RMSE": 0.8}),
            encoding="utf-8",
        )
        (run_dir / "config.json").write_text(
            json.dumps({"run": {"name": run_dir.name}, "model": {"name": "tcn_multimodal"}}),
            encoding="utf-8",
        )
        return run_dir

    def test_find_training_run_dirs_discovers_all_nested_logs(self) -> None:
        root = self.tmp_path / "outputs"
        first = self._write_run(root, "exp02/run_a")
        second = self._write_run(root, "exp03/run_b")
        (root / "notes").mkdir(parents=True, exist_ok=True)
        (root / "notes" / "train_log.txt").write_text("not a csv", encoding="utf-8")

        discovered = plot_deep_training_curves.find_training_run_dirs(root)

        self.assertEqual(discovered, [first, second])

    def test_generate_training_curve_artifacts_outputs_png_and_svg(self) -> None:
        root = self.tmp_path / "outputs"
        self._write_run(root, "exp02/run_a")
        output_dir = self.tmp_path / "figures"

        result = plot_deep_training_curves.generate_training_curve_artifacts(
            root=root,
            output_dir=output_dir,
            formats=("png", "svg"),
            dpi=120,
        )

        self.assertEqual(result["processed_runs"], 1)
        self.assertEqual(result["skipped_runs"], 0)
        self.assertTrue((output_dir / "exp02__run_a_training_curves.png").exists())
        self.assertTrue((output_dir / "exp02__run_a_training_curves.svg").exists())
        self.assertTrue((output_dir / "all_runs_val_macro_RMSE.png").exists())
        self.assertTrue((output_dir / "all_runs_val_macro_RMSE.svg").exists())

    def test_generate_training_curve_artifacts_allows_missing_lr_column(self) -> None:
        root = self.tmp_path / "outputs"
        self._write_run(root, "exp02/run_without_lr", include_lr=False)
        output_dir = self.tmp_path / "figures"

        result = plot_deep_training_curves.generate_training_curve_artifacts(
            root=root,
            output_dir=output_dir,
            formats=("png", "svg"),
            dpi=120,
        )

        self.assertEqual(result["processed_runs"], 1)
        self.assertEqual(result["skipped_runs"], 0)
        self.assertEqual(result["warnings"], [])
        self.assertTrue((output_dir / "exp02__run_without_lr_training_curves.png").exists())

    def test_generate_training_curve_artifacts_prefers_sum_diagnostics_panel(self) -> None:
        root = self.tmp_path / "outputs"
        self._write_run(root, "exp02/run_with_sum_metrics", include_sum_metrics=True)
        output_dir = self.tmp_path / "figures"

        result = plot_deep_training_curves.generate_training_curve_artifacts(
            root=root,
            output_dir=output_dir,
            formats=("svg",),
            dpi=120,
        )

        self.assertEqual(result["processed_runs"], 1)
        svg_text = (output_dir / "exp02__run_with_sum_metrics_training_curves.svg").read_text(encoding="utf-8")
        self.assertIn("Validation sum diagnostics", svg_text)
        self.assertIn("mean_abs_sum_error", svg_text)
        self.assertIn("mean_pred_sum", svg_text)

    def test_generate_training_curve_artifacts_skips_empty_or_invalid_logs(self) -> None:
        root = self.tmp_path / "outputs"
        self._write_run(root, "exp02/run_valid")

        empty_dir = root / "exp02" / "run_empty"
        empty_dir.mkdir(parents=True, exist_ok=True)
        (empty_dir / "train_log.csv").write_text("", encoding="utf-8")

        invalid_dir = root / "exp02" / "run_invalid"
        invalid_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"epoch": [1], "train_loss": [1.0]}).to_csv(invalid_dir / "train_log.csv", index=False)

        output_dir = self.tmp_path / "figures"
        result = plot_deep_training_curves.generate_training_curve_artifacts(
            root=root,
            output_dir=output_dir,
            formats=("png",),
            dpi=120,
        )

        self.assertEqual(result["processed_runs"], 1)
        self.assertEqual(result["skipped_runs"], 2)
        self.assertEqual(len(result["warnings"]), 2)
        self.assertTrue(any("run_empty" in warning for warning in result["warnings"]))
        self.assertTrue(any("run_invalid" in warning for warning in result["warnings"]))


if __name__ == "__main__":
    unittest.main()
