import csv
import pathlib
import shutil
import sys
import tempfile
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "sim"))
sys.modules.pop("scripts", None)

from scripts.generate_waveform_dataset import generate_waveform_dataset
from sim_common.phases import phase_boundaries


class GenerateWaveformDatasetTests(unittest.TestCase):
    def test_noise_seed_copies_expand_sequence_count_and_diversify_waveforms(self) -> None:
        tmp = tempfile.mkdtemp()
        try:
            base = pathlib.Path(tmp)
            generate_waveform_dataset(
                base,
                sequence_count=2,
                timesteps=24,
                seed=20260517,
                storage="memmap",
                noise_seed_count=3,
                multi_path_phase="off",
            )

            rows = self._read_csv_rows(base / "condition_grid_sequence.csv")
            self.assertEqual(len(rows), 6)

            base_ids = {row["base_condition_id"] for row in rows}
            self.assertEqual(len(base_ids), 2)

            grouped = {}
            for row in rows:
                grouped.setdefault(row["base_condition_id"], []).append(row)

            for group_rows in grouped.values():
                self.assertEqual(len(group_rows), 3)
                self.assertEqual({row["noise_seed_index"] for row in group_rows}, {"0", "1", "2"})
                self.assertEqual(len({row["x_H2"] for row in group_rows}), 1)
                self.assertEqual(len({row["x_CH4"] for row in group_rows}), 1)
                self.assertEqual(len({row["x_CO2"] for row in group_rows}), 1)
                self.assertEqual(len({row["x_N2"] for row in group_rows}), 1)

            sequence_ids = np.load(base / "metadata" / "sequence_ids.npy", allow_pickle=True)
            self.assertEqual(len(sequence_ids), 6)

            ultrasonic = np.load(base / "sequences" / "ultrasonic_int16.npy", mmap_mode="r")
            self.assertFalse(np.array_equal(ultrasonic[0, 0], ultrasonic[1, 0]))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_steady_multi_path_updates_l_m_inside_sequence(self) -> None:
        tmp = tempfile.mkdtemp()
        try:
            base = pathlib.Path(tmp)
            generate_waveform_dataset(
                base,
                sequence_count=1,
                timesteps=24,
                seed=20260517,
                storage="memmap",
                noise_seed_count=1,
                multi_path_phase="steady",
            )

            slow = np.load(base / "sequences" / "slow.npy", mmap_mode="r")
            q1, q2, q3 = phase_boundaries(24)
            del q1
            steady_lms = sorted({round(float(value), 1) for value in slow[0, q2:q3, 6]})
            self.assertEqual(steady_lms, [0.2, 0.6, 1.0, 1.4])

            rows = self._read_csv_rows(base / "condition_grid_sequence.csv")
            self.assertEqual(rows[0]["multi_path_phase"], "steady")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    @staticmethod
    def _read_csv_rows(path: pathlib.Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))


if __name__ == "__main__":
    unittest.main()
