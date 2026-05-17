import pathlib
import sys
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "dl"))

from data.dataset_waveform import WaveformSequenceDataset


def _preloaded_waveform_data(timesteps: int = 120) -> dict:
    return {
        "ultrasonic": np.zeros((1, timesteps, 8), dtype=np.int16),
        "ultrasonic_scale": np.ones((1, timesteps), dtype=np.float32),
        "fiber_mic": np.zeros((1, timesteps, 10), dtype=np.int16),
        "fiber_mic_scale": np.ones((1, timesteps), dtype=np.float32),
        "slow": np.zeros((1, timesteps, 8), dtype=np.float32),
        "y": np.zeros((1, 4), dtype=np.float32),
        "sequence_ids": np.array(["Q1"], dtype=object),
        "slow_channel_names": np.array(
            ["V_NDIR_CH4", "V_NDIR_CO2", "V_TCS", "T_C", "P_MPa", "H_RH", "L_m", "piston_position_m"],
            dtype=object,
        ),
        "label_names": np.array(["x_H2", "x_CH4", "x_CO2", "x_N2"], dtype=object),
    }


class WaveformStageOneHotTests(unittest.TestCase):
    def test_dataset_returns_stage_one_hot_for_all_timesteps(self) -> None:
        dataset = WaveformSequenceDataset("missing", indices=[0], index_path=None, preloaded_data=_preloaded_waveform_data())

        sample = dataset[0]

        self.assertIn("stage_one_hot", sample)
        self.assertEqual(tuple(sample["stage_one_hot"].shape), (120, 4))
        self.assertTrue(np.allclose(sample["stage_one_hot"].numpy().sum(axis=1), 1.0))
        self.assertTrue(np.array_equal(sample["stage_one_hot"][0].numpy(), np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)))
        self.assertTrue(np.array_equal(sample["stage_one_hot"][20].numpy(), np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)))
        self.assertTrue(np.array_equal(sample["stage_one_hot"][70].numpy(), np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)))
        self.assertTrue(np.array_equal(sample["stage_one_hot"][100].numpy(), np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)))

    def test_dataset_stage_one_hot_respects_time_window_subset(self) -> None:
        dataset = WaveformSequenceDataset(
            "missing",
            indices=[0],
            index_path=None,
            time_indices=[0, 20, 70, 100],
            preloaded_data=_preloaded_waveform_data(),
        )

        sample = dataset[0]
        expected = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        self.assertEqual(tuple(sample["slow"].shape), (4, 8))
        self.assertEqual(tuple(sample["stage_one_hot"].shape), (4, 4))
        self.assertTrue(np.array_equal(sample["stage_one_hot"].numpy(), expected))


if __name__ == "__main__":
    unittest.main()
