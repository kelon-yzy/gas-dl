import json
import pathlib
import shutil
import sys
import tempfile
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "ml"))
sys.path.insert(0, str(ROOT / "src" / "sim"))

from patent_model.feature_profiles import (
    DUAL_WAVEFORM_ACOUSTIC_COLUMNS,
    DUAL_WAVEFORM_ACOUSTIC_ENV_COLUMNS,
    DUAL_WAVEFORM_OPTICAL_ENV_COLUMNS,
    get_feature_profile,
)
from scripts.extract_dual_waveform_features import generate_traditional_from_waveform_v3


class PatentFeatureProfileTests(unittest.TestCase):
    def test_dual_waveform_profiles_match_patent_core_columns(self) -> None:
        self.assertEqual(
            DUAL_WAVEFORM_ACOUSTIC_COLUMNS,
            ("TOF", "Amp", "f_peak", "A_fft_max"),
        )
        self.assertEqual(
            get_feature_profile("v3_waveform_dual_channel_four")["optical_columns"],
            ("V_NDIR_CH4", "V_NDIR_CO2", "delta_I_CH4", "delta_I_CO2", "A_NDIR_CH4", "A_NDIR_CO2"),
        )
        self.assertEqual(
            DUAL_WAVEFORM_ACOUSTIC_ENV_COLUMNS,
            (
                "TOF",
                "Amp",
                "f_peak",
                "A_fft_max",
                "T_C",
                "P_MPa",
                "H_RH",
                "T_K",
                "P_kPa",
                "p_H2O_kPa",
                "x_H2O",
                "AH_g_m3",
                "P_dry_kPa",
                "sound_speed",
                "attenuation_alpha",
                "c_sound",
                "c_T_norm",
                "delta_Amp",
            ),
        )
        self.assertEqual(
            DUAL_WAVEFORM_OPTICAL_ENV_COLUMNS,
            (
                "V_NDIR_CH4",
                "V_NDIR_CO2",
                "delta_I_CH4",
                "delta_I_CO2",
                "A_NDIR_CH4",
                "A_NDIR_CO2",
                "T_C",
                "P_MPa",
                "H_RH",
                "T_K",
                "P_kPa",
                "p_H2O_kPa",
                "x_H2O",
                "AH_g_m3",
                "P_dry_kPa",
            ),
        )

    def test_feature_export_writes_patent_core_training_columns(self) -> None:
        tmp = pathlib.Path(tempfile.mkdtemp())
        try:
            source = tmp / "data"
            (source / "sequences").mkdir(parents=True)
            (source / "labels").mkdir()
            (source / "metadata").mkdir()
            np.save(source / "sequences" / "ultrasonic_int16.npy", np.zeros((1, 120, 8), dtype=np.int16))
            np.save(source / "sequences" / "ultrasonic_scale.npy", np.ones((1, 120), dtype=np.float32))
            np.save(source / "sequences" / "fiber_mic_int16.npy", np.zeros((1, 120, 10), dtype=np.int16))
            np.save(source / "sequences" / "fiber_mic_scale.npy", np.ones((1, 120), dtype=np.float32))
            slow = np.zeros((1, 120, 8), dtype=np.float32)
            slow[:, :, 0] = 1.0
            slow[:, :, 1] = 2.0
            slow[:, :, 2] = 3.0
            slow[:, :, 3] = 25.0
            slow[:, :, 4] = 0.1
            slow[:, :, 5] = 40.0
            slow[:, :, 6] = 0.8
            slow[:, :, 7] = 0.8
            np.save(source / "sequences" / "slow.npy", slow)
            np.save(source / "labels" / "y.npy", np.array([[10.0, 70.0, 5.0, 15.0]], dtype=np.float32))
            np.save(source / "metadata" / "sequence_ids.npy", np.array(["Q000001"], dtype=object))

            import pandas as pd

            pd.DataFrame([{"sequence_id": "Q000001", "mixture_id": "M0001"}]).to_csv(source / "sequence_index.csv", index=False)
            pd.DataFrame([{"sequence_id": "Q000001"}]).to_csv(source / "condition_grid_sequence.csv", index=False)

            output = tmp / "out"
            generate_traditional_from_waveform_v3(source, output, sequence_limit=1, timesteps=[0])

            acoustic_columns = list(pd.read_csv(output / "training" / "train_acoustic.csv").columns)
            optical_columns = list(pd.read_csv(output / "training" / "train_optical.csv").columns)
            manifest = json.loads((output / "features" / "feature_manifest.json").read_text(encoding="utf-8"))

            self.assertEqual(acoustic_columns, ["sample_id", "TOF", "Amp", "f_peak", "A_fft_max"])
            self.assertEqual(
                optical_columns,
                ["sample_id", "V_NDIR_CH4", "V_NDIR_CO2", "delta_I_CH4", "delta_I_CO2", "A_NDIR_CH4", "A_NDIR_CO2"],
            )
            self.assertEqual(
                manifest["training_acoustic_columns"],
                ["sample_id", "TOF", "Amp", "f_peak", "A_fft_max"],
            )
            self.assertEqual(
                manifest["training_optical_columns"],
                ["sample_id", "V_NDIR_CH4", "V_NDIR_CO2", "delta_I_CH4", "delta_I_CO2", "A_NDIR_CH4", "A_NDIR_CO2"],
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
