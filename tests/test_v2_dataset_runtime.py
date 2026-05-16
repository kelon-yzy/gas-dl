import pathlib
import pickle
import shutil
import sys
import tempfile
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "dl"))

from data.dataset_v2 import V2SequenceDataset


class V2DatasetRuntimeTests(unittest.TestCase):
    def test_v2_dataset_survives_pickle_without_embedding_arrays(self) -> None:
        tmp = tempfile.mkdtemp()
        dataset = None
        restored = None
        sample = None
        try:
            base = pathlib.Path(tmp)
            X = np.arange(2 * 4 * 12, dtype=np.float32).reshape(2, 4, 12)
            y = np.arange(2 * 4, dtype=np.float32).reshape(2, 4)
            np.savez(
                base / "v2_sequence.npz",
                X=X,
                y=y,
                channel_names=np.array(
                    [
                        "V_NDIR_CH4",
                        "V_NDIR_CO2",
                        "V_TCS",
                        "T_C",
                        "P_MPa",
                        "H_RH",
                        "L_m",
                        "piston_position_m",
                        "TOF",
                        "Amp",
                        "f_peak",
                        "A_fft_max",
                    ],
                    dtype=object,
                ),
                label_names=np.array(["x_H2", "x_CH4", "x_CO2", "x_N2"], dtype=object),
                sequence_ids=np.array(["Q000001", "Q000002"], dtype=object),
            )

            dataset = V2SequenceDataset(npz_path=base / "v2_sequence.npz", indices=[0, 1], index_path=None)
            sample = dataset[0]
            self.assertIsInstance(dataset.X, np.ndarray)
            self.assertIsInstance(dataset.y, np.ndarray)

            payload = pickle.dumps(dataset, protocol=pickle.HIGHEST_PROTOCOL)
            self.assertLess(len(payload), 100_000)

            restored = pickle.loads(payload)
            self.assertIsNone(restored.X)
            self.assertIsNone(restored.y)

            sample = restored[0]
            self.assertIsInstance(restored.X, np.ndarray)
            self.assertIsInstance(restored.y, np.ndarray)
            self.assertEqual(tuple(sample[0].shape), (4, 12))
            self.assertEqual(tuple(sample[1].shape), (4,))
            self.assertEqual(sample[2]["sample_id"], "Q000001")
        finally:
            del sample
            del restored
            del dataset
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
