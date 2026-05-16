import json
import pathlib
import shutil
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "ml"))
sys.modules.pop("scripts", None)
sys.modules.pop("scripts.report", None)

from scripts.report.loaders import _load_main_runs


class ReportLoaderModalSchemaTests(unittest.TestCase):
    def test_load_main_runs_expands_summary_to_four_model_rows(self) -> None:
        tmp = pathlib.Path(tempfile.mkdtemp())
        try:
            run_dir = tmp / "four_component_v3sync_model_grid_raw_tph" / "xgboost_ridge"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "branch_model_type": "xgboost",
                        "meta_model_type": "ridge",
                        "train_samples": 100,
                        "test_samples": 20,
                        "acoustic_macro_RMSE_pp": 5.0,
                        "acoustic_macro_MRE_pct": 100.0,
                        "acoustic_macro_R2": 0.2,
                        "acoustic_macro_MaxRE_pct": 1000.0,
                        "optical_macro_RMSE_pp": 4.5,
                        "optical_macro_MRE_pct": 90.0,
                        "optical_macro_R2": 0.3,
                        "optical_macro_MaxRE_pct": 900.0,
                        "thermal_macro_RMSE_pp": 6.0,
                        "thermal_macro_MRE_pct": 130.0,
                        "thermal_macro_R2": 0.1,
                        "thermal_macro_MaxRE_pct": 1100.0,
                        "fused_macro_RMSE_pp": 4.0,
                        "fused_macro_MRE_pct": 80.0,
                        "fused_macro_R2": 0.5,
                        "fused_macro_MaxRE_pct": 800.0,
                    }
                ),
                encoding="utf-8",
            )

            frame = _load_main_runs(tmp)

            self.assertEqual(set(frame["model_name"]), {"acoustic", "optical", "thermal", "fused"})
            self.assertEqual(len(frame), 4)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
