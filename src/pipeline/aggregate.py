from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SUMMARY_DIR = ROOT / "outputs" / "summary"
RESULTS_TSV = SUMMARY_DIR / "results.tsv"
RESULTS_MULTI_TSV = SUMMARY_DIR / "results_multiseed.tsv"


def _collect_grid_summaries() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    patterns = [
        ("outputs/exp01_traditional/four_component_*_grid_summary.csv", "exp01_traditional"),
        ("outputs/exp01_traditional_smoke/four_component_*_grid_summary.csv", "exp01_traditional_smoke"),
        ("outputs/exp03_fusion/four_component_*_grid_summary.csv", "exp03_fusion"),
        ("outputs/exp05_robust/**/robustness_summary.csv", "exp05_robust"),
    ]
    for pattern, exp_id in patterns:
        for path in ROOT.glob(pattern):
            frame = pd.read_csv(path)
            frame.insert(0, "source_file", str(path.relative_to(ROOT)))
            frame.insert(1, "exp_id", exp_id)
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _aggregate_multiseed(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    group_cols = [c for c in ["exp_id", "profile", "combo"] if c in frame.columns]
    metric_cols = [c for c in ["macro_RMSE_pp", "macro_MRE_pct", "macro_R2", "macro_MaxRE_pct"] if c in frame.columns]
    if not group_cols or not metric_cols:
        return pd.DataFrame()
    grouped = frame.groupby(group_cols, as_index=False)[metric_cols].agg(["mean", "std", "min", "max"])
    grouped.columns = ["_".join([x for x in col if x]).strip("_") for col in grouped.columns.to_flat_index()]
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate traditional experiment outputs into summary tables.")
    parser.parse_args()
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    frame = _collect_grid_summaries()
    frame.to_csv(RESULTS_TSV, sep="\t", index=False)
    multi = _aggregate_multiseed(frame)
    multi.to_csv(RESULTS_MULTI_TSV, sep="\t", index=False)
    print(f"Wrote {RESULTS_TSV}")
    print(f"Wrote {RESULTS_MULTI_TSV}")


if __name__ == "__main__":
    main()
