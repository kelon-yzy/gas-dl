import json
import pathlib
import shutil
import sys
import tempfile
import unittest

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipeline import aggregate
from pipeline import status_store


class AggregateAndStatusStoreTests(unittest.TestCase):
    def test_collect_grid_summaries_does_not_duplicate_main_exp01_as_exp06_without_repro_outputs(self) -> None:
        tmp = pathlib.Path(tempfile.mkdtemp())
        try:
            summary_path = tmp / "outputs" / "exp01_traditional" / "runs" / "archive" / "four_component_formal_seed42_core_grid_summary.csv"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [
                    {
                        "profile": "v3_raw_no_env",
                        "combo": "xgboost_xgboost",
                        "model_name": "fused",
                        "macro_RMSE_pp": 4.099393358944436,
                    }
                ]
            ).to_csv(summary_path, index=False)

            frame = aggregate._collect_grid_summaries(tmp)

            self.assertEqual(frame["exp_id"].tolist(), ["exp01_traditional"])
            self.assertEqual(
                [value.replace("\\", "/") for value in frame["source_file"].tolist()],
                ["outputs/exp01_traditional/runs/archive/four_component_formal_seed42_core_grid_summary.csv"],
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_collect_deep_summaries_does_not_duplicate_main_seed42_as_exp06_without_repro_outputs(self) -> None:
        tmp = pathlib.Path(tempfile.mkdtemp())
        try:
            summary_path = tmp / "src" / "dl" / "outputs" / "exp02_deep_e2e" / "v3_multimodal_fusion_seed42" / "summary.json"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(
                json.dumps(
                    {
                        "run_name": "v3_multimodal_fusion_seed42",
                        "macro_RMSE": 6.79,
                    }
                ),
                encoding="utf-8",
            )

            frame = aggregate._collect_deep_summaries(tmp)

            self.assertEqual(frame["exp_id"].tolist(), ["exp02_deep_e2e"])
            self.assertEqual(
                [value.replace("\\", "/") for value in frame["source_file"].tolist()],
                ["src/dl/outputs/exp02_deep_e2e/v3_multimodal_fusion_seed42/summary.json"],
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_status_store_finish_reuses_row_and_picks_best_macro_rmse(self) -> None:
        tmp = pathlib.Path(tempfile.mkdtemp())
        try:
            status_path = tmp / "STATUS.tsv"
            summary_path = tmp / "outputs" / "exp01_traditional" / "runs" / "archive" / "four_component_formal_seed42_core_grid_summary.csv"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [
                    {"profile": "v3_raw_no_env", "combo": "pls_ridge", "model_name": "fused", "macro_RMSE_pp": 5.7932},
                    {"profile": "v3_raw_no_env", "combo": "xgboost_xgboost", "model_name": "fused", "macro_RMSE_pp": 4.0994},
                ]
            ).to_csv(summary_path, index=False)

            status_store.mark_running(
                exp_id="exp01_traditional",
                model="grid",
                seed=42,
                notes="core+diag running",
                status_path=status_path,
                started_at="2026-05-15 17:00:00",
            )
            status_store.mark_finished(
                exp_id="exp01_traditional",
                model="grid",
                seed=42,
                notes="core+diag finished",
                summary_csv=summary_path,
                status_path=status_path,
                finished_at="2026-05-15 17:06:31",
            )

            rows = status_store.load_rows(status_path)

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["status"], "success")
            self.assertEqual(rows[0]["started"], "2026-05-15 17:00:00")
            self.assertEqual(rows[0]["finished"], "2026-05-15 17:06:31")
            self.assertEqual(rows[0]["macro_RMSE"], "4.0994")
            self.assertEqual(rows[0]["notes"], "core+diag finished")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_collect_grid_summaries_uses_model_name_in_result_group(self) -> None:
        tmp = pathlib.Path(tempfile.mkdtemp())
        try:
            summary_path = tmp / "outputs" / "exp01_traditional" / "runs" / "archive" / "four_component_formal_seed42_core_grid_summary.csv"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [
                    {
                        "profile": "v3_raw_no_env",
                        "combo": "xgboost_ridge",
                        "model_name": "acoustic",
                        "macro_RMSE_pp": 5.0,
                    },
                    {
                        "profile": "v3_raw_no_env",
                        "combo": "xgboost_ridge",
                        "model_name": "fused",
                        "macro_RMSE_pp": 4.0,
                    },
                ]
            ).to_csv(summary_path, index=False)

            frame = aggregate._collect_grid_summaries(tmp)

            groups = set(frame["result_group"].tolist())
            self.assertEqual(groups, {"v3_raw_no_env/xgboost_ridge/acoustic", "v3_raw_no_env/xgboost_ridge/fused"})
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
