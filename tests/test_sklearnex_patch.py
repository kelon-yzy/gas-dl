import pathlib
import sys
import types
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "ml"))


class SklearnexPatchTests(unittest.TestCase):
    def test_patch_sklearn_for_traditional_ml_calls_sklearnex_patch(self) -> None:
        calls = []
        fake_sklearnex = types.ModuleType("sklearnex")

        def patch_sklearn() -> None:
            calls.append("patched")

        fake_sklearnex.patch_sklearn = patch_sklearn
        original_sklearnex = sys.modules.get("sklearnex")
        sys.modules["sklearnex"] = fake_sklearnex
        try:
            from patent_model.sklearnex_patch import patch_sklearn_for_traditional_ml

            patch_sklearn_for_traditional_ml()
        finally:
            if original_sklearnex is None:
                sys.modules.pop("sklearnex", None)
            else:
                sys.modules["sklearnex"] = original_sklearnex

        self.assertEqual(calls, ["patched"])


if __name__ == "__main__":
    unittest.main()
