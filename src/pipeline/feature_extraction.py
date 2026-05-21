from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SIM_ROOT = ROOT / "src" / "sim"
if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))

from scripts.extract_dual_waveform_features import generate_traditional_from_waveform_v3


def main() -> None:
    parser = argparse.ArgumentParser(description="Export traditional-model feature tables from V3.1 dual-channel waveform data.")
    parser.add_argument("--source-dir", default=str(ROOT / "data" / "waveform_v3"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "exp01_traditional" / "data"))
    parser.add_argument("--sequence-limit", type=int, default=None)
    args = parser.parse_args()
    generate_traditional_from_waveform_v3(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        sequence_limit=args.sequence_limit,
        timesteps=None,
    )


if __name__ == "__main__":
    main()
