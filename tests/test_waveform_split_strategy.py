import pathlib
import sys
import tempfile
import unittest

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "sim"))
sys.path.insert(0, str(ROOT / "src" / "dl"))

from sim_common.splits import STRATIFIED_GROUP_SPLIT_POLICY, build_stratified_group_splits_with_extrapolation
from data.split_utils import load_existing_splits


def _build_conditions():
    rows = []
    l_values = (0.2, 0.6, 1.0, 1.4)
    h2_values = (0.2, 1.5, 4.0, 8.0, 14.0, 24.0)
    co2_values = (0.2, 1.2, 2.5, 4.5, 7.5, 12.5)
    n2_values = (0.5, 2.0, 4.5, 8.5, 13.5, 19.0)
    pressure_values = (0.11, 0.18, 0.28, 0.42, 0.58, 0.69)
    humidity_values = (25.0, 35.0, 45.0, 55.0, 65.0, 75.0)
    temperature_values = (16.0, 19.0, 22.0, 26.0, 30.0, 34.0)

    sequence_index = 1
    for group_index in range(24):
        family = group_index % 6
        for repeat_index, l_m in enumerate(l_values, start=1):
            x_h2 = h2_values[family]
            x_co2 = co2_values[(family + repeat_index - 1) % 6]
            x_n2 = n2_values[(family + repeat_index - 1) % 6]
            x_ch4 = 100.0 - x_h2 - x_co2 - x_n2
            rows.append(
                {
                    "sequence_id": f"Q{sequence_index:06d}",
                    "base_condition_id": f"B{group_index + 1:06d}",
                    "mixture_id": f"MD{group_index + 1:05d}",
                    "x_H2": f"{x_h2:.6f}",
                    "x_CH4": f"{x_ch4:.6f}",
                    "x_CO2": f"{x_co2:.6f}",
                    "x_N2": f"{x_n2:.6f}",
                    "T_C_base": f"{temperature_values[family]:.4f}",
                    "P_MPa_base": f"{pressure_values[family]:.4f}",
                    "H_RH_base": f"{humidity_values[family]:.4f}",
                    "L_m_base": f"{l_m:.4f}",
                    "status": "synthetic_measurement",
                }
            )
            sequence_index += 1
    return rows


class WaveformSplitStrategyTests(unittest.TestCase):
    def test_group_isolation_and_determinism(self):
        conditions = _build_conditions()
        split_rows_a, summary_a = build_stratified_group_splits_with_extrapolation(conditions, seed=20260514)
        split_rows_b, summary_b = build_stratified_group_splits_with_extrapolation(conditions, seed=20260514)

        self.assertEqual(summary_a["split_policy"], STRATIFIED_GROUP_SPLIT_POLICY)
        self.assertEqual(split_rows_a, split_rows_b)
        self.assertEqual(summary_a["splits"], summary_b["splits"])

        sequence_to_split = {}
        mixture_to_split = {}
        for split_name, rows in split_rows_a.items():
            for row in rows:
                sequence_to_split[row["sequence_id"]] = split_name
                mixture_id = row["mixture_id"]
                if mixture_id in mixture_to_split:
                    self.assertEqual(mixture_to_split[mixture_id], split_name)
                else:
                    mixture_to_split[mixture_id] = split_name

        self.assertEqual(len(sequence_to_split), len(conditions))
        self.assertEqual(set(sequence_to_split.values()), {"train", "val", "test", "extrapolation"})

    def test_extrapolation_size_is_close_to_target(self):
        conditions = _build_conditions()
        split_rows, summary = build_stratified_group_splits_with_extrapolation(conditions, seed=20260514)

        target = int(round(len(conditions) * summary["extrapolation_ratio_target"]))
        actual = len(split_rows["extrapolation"])
        max_group_size = 4

        self.assertGreater(actual, 0)
        self.assertLessEqual(abs(actual - target), max_group_size)
        self.assertGreater(summary["selected_extrapolation_groups"], 0)
        self.assertGreater(summary["candidate_extrapolation_groups"], 0)
        self.assertGreater(len(split_rows["train"]), len(split_rows["val"]))
        self.assertGreater(len(split_rows["train"]), len(split_rows["test"]))

    def test_load_existing_splits_accepts_optional_extrapolation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            split_dir = pathlib.Path(tmpdir)
            pd.DataFrame([{"sequence_id": "Q1", "mixture_id": "M1"}]).to_csv(split_dir / "train_sequence_ids.csv", index=False)
            pd.DataFrame([{"sequence_id": "Q2", "mixture_id": "M2"}]).to_csv(split_dir / "val_sequence_ids.csv", index=False)
            pd.DataFrame([{"sequence_id": "Q3", "mixture_id": "M3"}]).to_csv(split_dir / "test_sequence_ids.csv", index=False)
            pd.DataFrame([{"sequence_id": "Q4", "mixture_id": "M4"}]).to_csv(split_dir / "extrapolation_sequence_ids.csv", index=False)

            splits = load_existing_splits(split_dir)

        self.assertIsNotNone(splits)
        self.assertIn("extrapolation", splits)
        self.assertEqual(list(splits["extrapolation"]["sequence_id"]), ["Q4"])


if __name__ == "__main__":
    unittest.main()
