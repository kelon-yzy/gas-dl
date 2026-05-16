import pathlib
import shutil
import sys
import tempfile
import unittest

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "ml"))

from patent_model.data_loader import load_patent_dataset


def _write_stage_fixture(base: pathlib.Path) -> None:
    (base / "training").mkdir()
    (base / "labels").mkdir()
    (base / "features").mkdir()
    sample_ids = ["S1", "S2", "S3", "S4"]
    stages = ["baseline_stage", "distance_stage", "pressure_stage", "purge_stage"]

    pd.DataFrame(
        {
            "sample_id": sample_ids,
            "TOF": [0.1, 0.2, 0.3, 0.4],
            "Amp": [1.0, 1.1, 1.2, 1.3],
            "f_peak": [40000.0, 40000.0, 40000.0, 40000.0],
            "A_fft_max": [10.0, 11.0, 12.0, 13.0],
            "L_m": [0.5, 0.6, 0.7, 0.8],
            "T_C": [25.0, 25.0, 25.0, 25.0],
            "P_MPa": [0.1, 0.2, 0.3, 0.4],
            "H_RH": [40.0, 40.0, 40.0, 40.0],
        }
    ).to_csv(base / "training" / "train_acoustic.csv", index=False)
    pd.DataFrame(
        {
            "sample_id": sample_ids,
            "V_NDIR_CH4": [1.0, 1.1, 1.2, 1.3],
            "V_NDIR_CO2": [2.0, 2.1, 2.2, 2.3],
            "delta_I_CH4": [0.0, 0.1, 0.2, 0.3],
            "delta_I_CO2": [0.0, 0.1, 0.2, 0.3],
        }
    ).to_csv(base / "training" / "train_optical.csv", index=False)
    pd.DataFrame({"sample_id": sample_ids, "V_TCS": [1.0, 1.1, 1.2, 1.3]}).to_csv(
        base / "training" / "train_thermal.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "sample_id": sample_ids,
            "x_H2": [10.0, 11.0, 12.0, 13.0],
            "x_CH4": [80.0, 79.0, 78.0, 77.0],
            "x_CO2": [5.0, 5.0, 5.0, 5.0],
        }
    ).to_csv(base / "labels" / "labels.csv", index=False)
    pd.DataFrame(
        {
            "sample_id": sample_ids,
            "mixture_id": ["M1", "M2", "M3", "M4"],
            "stage_id": stages,
            "repeat_id": [1, 1, 1, 1],
            "status": ["calibration_control", "synthetic_measurement", "synthetic_measurement", "synthetic_measurement"],
            "pressure_stage": ["low", "mid", "high", "mid"],
            "distance_stage": ["short", "mid", "long", "mid"],
            "piston_position_m": [0.1, 0.2, 0.3, 0.4],
            "x_N2": [5.0, 5.0, 5.0, 5.0],
            "T_C": [25.0, 25.0, 25.0, 25.0],
            "P_MPa": [0.1, 0.2, 0.3, 0.4],
            "H_RH": [40.0, 40.0, 40.0, 40.0],
        }
    ).to_csv(base / "condition_grid_v1.csv", index=False)
    pd.DataFrame(
        {
            "sample_id": sample_ids,
            "sound_speed": [340.0, 350.0, 360.0, 370.0],
            "attenuation_alpha": [0.1, 0.2, 0.3, 0.4],
            "ndir_ch4_saturated": [0, 0, 0, 0],
            "ndir_co2_saturated": [0, 0, 0, 0],
            "optical_baseline_drift_ch4": [0.0, 0.0, 0.0, 0.0],
            "optical_baseline_drift_co2": [0.0, 0.0, 0.0, 0.0],
            "thermal_baseline_drift": [0.0, 0.0, 0.0, 0.0],
            "lambda_mix_calibrated": [0.1, 0.2, 0.3, 0.4],
            "calibration_status": ["ok", "ok", "ok", "ok"],
        }
    ).to_csv(base / "features" / "feature_table.csv", index=False)


class StageFilterTests(unittest.TestCase):
    def test_stable_stage_filter_keeps_only_distance_and_pressure_stages(self) -> None:
        tmp = tempfile.mkdtemp()
        try:
            base = pathlib.Path(tmp)
            _write_stage_fixture(base)

            dataset = load_patent_dataset(base, profile="raw_no_env_four", stage_filter="stable", duplicate_filter="none")

            self.assertEqual(dataset.sample_ids.tolist(), ["S2", "S3"])
            self.assertEqual(dataset.metadata["stage_id"].tolist(), ["distance_stage", "pressure_stage"])
            report = dataset.filter_report["stage_filter"]
            self.assertEqual(report["mode"], "stable")
            self.assertEqual(report["before_samples"], 4)
            self.assertEqual(report["after_samples"], 2)
            self.assertEqual(report["removed_samples"], 2)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
