import pathlib
import sys
import tempfile
import unittest
from types import SimpleNamespace

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "ml"))
sys.modules.pop("scripts", None)
sys.modules.pop("scripts.report", None)

from scripts.run_four_component_model_grid import _rows_from_summary


class TraditionalGridSummarySchemaTests(unittest.TestCase):
    def test_rows_from_summary_expands_one_combo_to_four_model_rows(self) -> None:
        summary = {
            "data_dir": str(ROOT),
            "resolved_feature_profile": "v3_waveform_dual_channel_four",
            "train_samples": 13026,
            "test_samples": 3255,
            "fit_seconds": 1.2,
            "evaluate_seconds": 0.3,
            "write_seconds": 0.1,
            "total_seconds": 1.6,
            "n_jobs": 2,
            "xgb_n_jobs": 4,
            "prediction_cache_reused": True,
            "metadata_filter": "none",
            "stage_filter": "stable",
            "filter_report": {
                "stage_filter": {
                    "before_samples": 40000,
                    "after_samples": 20000,
                    "removed_samples": 20000,
                    "before_unique_mixtures": 5427,
                    "after_unique_mixtures": 5427,
                }
            },
            "physical_range_filter": "none",
            "label_closure_filter": "none",
            "duplicate_filter": "per_mixture_limit",
            "branch_model_type": "xgboost",
            "meta_model_type": "ridge",
            "acoustic_macro_RMSE_pp": 5.1,
            "acoustic_macro_MRE_pct": 100.0,
            "acoustic_macro_R2": 0.2,
            "acoustic_macro_MaxRE_pct": 900.0,
            "optical_macro_RMSE_pp": 4.7,
            "optical_macro_MRE_pct": 90.0,
            "optical_macro_R2": 0.3,
            "optical_macro_MaxRE_pct": 800.0,
            "thermal_macro_RMSE_pp": 6.4,
            "thermal_macro_MRE_pct": 140.0,
            "thermal_macro_R2": 0.1,
            "thermal_macro_MaxRE_pct": 950.0,
            "fused_macro_RMSE_pp": 4.0,
            "fused_macro_MRE_pct": 80.0,
            "fused_macro_R2": 0.6,
            "fused_macro_MaxRE_pct": 700.0,
            "best_model_by_macro_RMSE_pp": "fused",
            "modal_effective_pls_components": {
                "acoustic": None,
                "optical": None,
                "thermal": None,
                "fused": None,
            },
        }
        combo_dir = pathlib.Path(tempfile.mkdtemp()) / "xgboost_ridge"
        combo_dir.mkdir(parents=True, exist_ok=True)
        try:
            rows = _rows_from_summary("v3_raw_no_env", "xgboost_ridge", summary, combo_dir)

            self.assertEqual(len(rows), 4)
            self.assertEqual({row["model_name"] for row in rows}, {"acoustic", "optical", "thermal", "fused"})
            fused_row = next(row for row in rows if row["model_name"] == "fused")
            self.assertEqual(fused_row["macro_RMSE_pp"], 4.0)
            self.assertEqual(fused_row["best_model_by_macro_RMSE_pp"], "fused")
            self.assertEqual(fused_row["stage_filter"], "stable")
            self.assertEqual(fused_row["stage_filter_before_samples"], 40000)
            self.assertEqual(fused_row["stage_filter_after_samples"], 20000)
            self.assertEqual(fused_row["stage_filter_removed_samples"], 20000)
        finally:
            combo_dir.rmdir()
            combo_dir.parent.rmdir()


if __name__ == "__main__":
    unittest.main()
