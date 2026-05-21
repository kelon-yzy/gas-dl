import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "ml"))

from scripts.run_four_component_model_grid import build_parser as build_grid_parser
from scripts.train_patent_model import build_parser, prepare_training_data


class DuplicateFilterDefaultTests(unittest.TestCase):
    def test_default_duplicate_filter_reduces_train_samples(self) -> None:
        parser = build_parser()
        args = parser.parse_args([])
        args.data_dir = str(ROOT / "outputs" / "exp01_traditional" / "data")
        args.feature_profile = "v3_waveform_dual_channel_four"
        args.component_mode = "four"
        data = prepare_training_data(args)
        self.assertEqual(args.stage_filter, "stable")
        self.assertEqual(args.duplicate_filter, "per_mixture_limit")
        self.assertEqual(args.duplicate_per_mixture_limit, 3)
        self.assertEqual(data.train.n_samples, 9587)
        self.assertEqual(data.test.n_samples, 2411)

    def test_grid_parser_defaults_to_stable_stage_filter(self) -> None:
        args = build_grid_parser().parse_args([])

        self.assertEqual(args.stage_filter, "stable")


if __name__ == "__main__":
    unittest.main()
