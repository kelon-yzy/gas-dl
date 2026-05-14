import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "ml"))

from scripts.train_patent_model import build_parser, prepare_training_data


class DuplicateFilterDefaultTests(unittest.TestCase):
    def test_default_duplicate_filter_reduces_train_samples(self) -> None:
        parser = build_parser()
        args = parser.parse_args([])
        args.data_dir = str(ROOT / "outputs" / "exp01_traditional")
        args.feature_profile = "v3_waveform_dual_channel_four"
        args.component_mode = "four"
        data = prepare_training_data(args)
        self.assertEqual(args.duplicate_filter, "per_mixture_limit")
        self.assertEqual(args.duplicate_per_mixture_limit, 3)
        self.assertEqual(data.train.n_samples, 13026)
        self.assertEqual(data.test.n_samples, 3255)


if __name__ == "__main__":
    unittest.main()
