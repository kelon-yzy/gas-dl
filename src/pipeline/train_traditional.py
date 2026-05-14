from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ML_ROOT = ROOT / "src" / "ml"
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from scripts.run_four_component_model_grid import main as run_four_component_model_grid


def main() -> None:
    parser = argparse.ArgumentParser(description="Run G1 traditional-model grid for V3.1 dual-channel data.")
    parser.add_argument("--data-dir", default=str(ROOT / "outputs" / "exp01_traditional"))
    parser.add_argument("--output-root", default=str(ROOT / "outputs" / "exp01_traditional"))
    parser.add_argument("--tag", default="formal")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=1)
    args = parser.parse_args()

    argv = [
        "--raw-data-dir", args.data_dir,
        "--env-data-dir", args.data_dir,
        "--output-root", args.output_root,
        "--tag", args.tag,
        "--seed", str(args.seed),
        "--component-mode", "four",
        "--max-workers", str(args.max_workers),
    ]
    if args.train_limit is not None:
        argv.extend(["--train-limit", str(args.train_limit)])
    if args.test_limit is not None:
        argv.extend(["--test-limit", str(args.test_limit)])

    result = run_four_component_model_grid(argv)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(json.dumps({"finished_at": stamp, "result": result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
