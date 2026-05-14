from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
DL_ROOT = SRC_ROOT / "dl"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(DL_ROOT) not in sys.path:
    sys.path.insert(0, str(DL_ROOT))

from pipeline.cli_progress import build_cli_progress
from training.train import train_one


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a deep-learning config for V3.1 dual-channel data.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--ui", action="store_true")
    parser.add_argument("--no-ui", action="store_true")
    args = parser.parse_args()
    progress = build_cli_progress(force=args.ui, disable=args.no_ui)
    summary = train_one(args.config, epochs_override=args.epochs, progress=progress)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(json.dumps({"finished_at": stamp, "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

