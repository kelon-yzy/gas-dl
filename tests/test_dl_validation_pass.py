import pathlib
import sys
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "dl"))

from training.train import evaluate_with_predictions


class CountingRegressionDataset(Dataset):
    def __init__(self) -> None:
        self.x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        self.y = self.x + 1.0
        self.access_count = 0

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index):
        self.access_count += 1
        meta = {"sample_id": f"S{index}", "mixture_id": f"M{index // 2}"}
        return self.x[index], self.y[index], meta


class ValidationPassTests(unittest.TestCase):
    def test_evaluate_with_predictions_uses_one_loader_pass(self) -> None:
        dataset = CountingRegressionDataset()
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        model = torch.nn.Identity()
        loss_fn = torch.nn.MSELoss()

        loss, bundle = evaluate_with_predictions(model, loader, loss_fn, torch.device("cpu"))

        self.assertEqual(dataset.access_count, len(dataset))
        self.assertAlmostEqual(loss, 1.0)
        np.testing.assert_allclose(bundle.y_pred, dataset.x.numpy())
        np.testing.assert_allclose(bundle.y_true, dataset.y.numpy())
        self.assertEqual(bundle.meta["sample_id"].tolist(), ["S0", "S1", "S2", "S3"])


if __name__ == "__main__":
    unittest.main()
