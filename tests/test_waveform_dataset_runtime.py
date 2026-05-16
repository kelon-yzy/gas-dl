import json
import pathlib
import pickle
import shutil
import sys
import tempfile
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "dl"))

from data.dataset_waveform import WaveformSequenceDataset


class WaveformDatasetRuntimeTests(unittest.TestCase):
    def test_waveform_dataset_keeps_memmap_and_survives_pickle(self) -> None:
        tmp = tempfile.mkdtemp()
        dataset = None
        restored = None
        sample = None
        try:
            base = pathlib.Path(tmp)
            (base / "sequences").mkdir(parents=True)
            (base / "labels").mkdir(parents=True)
            (base / "metadata").mkdir(parents=True)

            ultrasonic = np.arange(2 * 3 * 16, dtype=np.int16).reshape(2, 3, 16)
            ultrasonic_scale = np.ones((2, 3), dtype=np.float32)
            fiber_mic = np.arange(2 * 3 * 20, dtype=np.int16).reshape(2, 3, 20)
            fiber_mic_scale = np.ones((2, 3), dtype=np.float32)
            slow = np.arange(2 * 3 * 8, dtype=np.float32).reshape(2, 3, 8)
            targets = np.arange(2 * 4, dtype=np.float32).reshape(2, 4)

            np.save(base / "sequences" / "ultrasonic_int16.npy", ultrasonic)
            np.save(base / "sequences" / "ultrasonic_scale.npy", ultrasonic_scale)
            np.save(base / "sequences" / "fiber_mic_int16.npy", fiber_mic)
            np.save(base / "sequences" / "fiber_mic_scale.npy", fiber_mic_scale)
            np.save(base / "sequences" / "slow.npy", slow)
            np.save(base / "labels" / "y.npy", targets)
            np.save(base / "metadata" / "sequence_ids.npy", np.array(["Q000001", "Q000002"], dtype=object))
            np.save(
                base / "metadata" / "slow_channel_names.npy",
                np.array(["V_NDIR_CH4", "V_NDIR_CO2", "V_TCS", "T_C", "P_MPa", "H_RH", "L_m", "piston_position_m"], dtype=object),
            )
            np.save(
                base / "metadata" / "label_names.npy",
                np.array(["x_H2", "x_CH4", "x_CO2", "x_N2"], dtype=object),
            )
            (base / "metadata" / "waveform_v3_spec.json").write_text(
                json.dumps(
                    {
                        "channels": {
                            "ultrasonic": {"waveform_samples": 16},
                            "fiber_mic": {"waveform_samples": 20},
                        }
                    }
                ),
                encoding="utf-8",
            )

            dataset = WaveformSequenceDataset(npz_path=base, indices=[0, 1], index_path=None)
            sample = dataset[0]
            # _ensure_loaded 将 mmap 数据预转为可写 ndarray，消除逐样本 copy
            self.assertIsInstance(dataset.slow, np.ndarray)
            self.assertIsInstance(dataset.y, np.ndarray)
            self.assertFalse(np.shares_memory(dataset.slow, np.load(base / "sequences" / "slow.npy", mmap_mode="r")))

            payload = pickle.dumps(dataset, protocol=pickle.HIGHEST_PROTOCOL)
            self.assertLess(len(payload), 100_000)

            restored = pickle.loads(payload)
            self.assertIsNone(restored.slow)
            self.assertIsNone(restored.y)

            sample = restored[0]
            # 恢复后重新加载，数据仍是可写 ndarray
            self.assertIsInstance(restored.slow, np.ndarray)
            self.assertIsInstance(restored.y, np.ndarray)
            self.assertEqual(tuple(sample["slow"].shape), (3, 8))
            self.assertEqual(tuple(sample["target"].shape), (4,))
            self.assertEqual(sample["meta"]["sample_id"], "Q000001")
        finally:
            del sample
            del restored
            del dataset
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
