from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SUMMARY_DIR = ROOT / "outputs" / "summary"
RESULTS_TSV = SUMMARY_DIR / "results.tsv"
RESULTS_MULTI_TSV = SUMMARY_DIR / "results_multiseed.tsv"


def _collect_grid_summaries(root: Path = ROOT) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    patterns = [
        ("outputs/exp04_domain/domain_holdout_summary.csv", "exp04_domain"),
        ("outputs/exp01_traditional/four_component_*_grid_summary.csv", "exp01_traditional"),
        ("outputs/exp01_traditional_smoke/four_component_*_grid_summary.csv", "exp01_traditional_smoke"),
        ("outputs/exp03_fusion/four_component_*_grid_summary.csv", "exp03_fusion"),
        ("outputs/exp05_robust/**/robustness_summary.csv", "exp05_robust"),
    ]
    for pattern, exp_id in patterns:
        for path in root.glob(pattern):
            frame = pd.read_csv(path)
            frame.insert(0, "source_file", str(path.relative_to(root)))
            frame.insert(1, "exp_id", exp_id)
            frame["result_family"] = "traditional"
            if "profile" in frame.columns and "combo" in frame.columns and "model_name" in frame.columns:
                frame["result_group"] = (
                    frame["profile"].astype(str) + "/" + frame["combo"].astype(str) + "/" + frame["model_name"].astype(str)
                )
            elif "profile" in frame.columns and "combo" in frame.columns:
                frame["result_group"] = frame["profile"].astype(str) + "/" + frame["combo"].astype(str)
            elif "domain_id" in frame.columns and "combo" in frame.columns:
                frame["result_group"] = "domain/" + frame["domain_id"].astype(str) + "/" + frame["combo"].astype(str)
            elif "combo" in frame.columns:
                frame["result_group"] = frame["combo"].astype(str)
            else:
                frame["result_group"] = exp_id
            frames.append(frame)
    repro_files = list(root.glob("outputs/exp06_reproducibility/traditional/four_component_*_grid_summary.csv"))
    if repro_files:
        seed42_main = root / "outputs" / "exp01_traditional" / "four_component_formal_seed42_core_grid_summary.csv"
        if seed42_main.exists():
            repro_files.insert(0, seed42_main)
        for path in repro_files:
            frame = pd.read_csv(path)
            frame.insert(0, "source_file", str(path.relative_to(root)))
            frame.insert(1, "exp_id", "exp06_reproducibility")
            frame["result_family"] = "traditional"
            if "model_name" in frame.columns:
                frame["result_group"] = (
                    frame["profile"].astype(str) + "/" + frame["combo"].astype(str) + "/" + frame["model_name"].astype(str)
                )
            else:
                frame["result_group"] = frame["profile"].astype(str) + "/" + frame["combo"].astype(str)
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _collect_deep_summaries(root: Path = ROOT) -> pd.DataFrame:
    rows: list[dict] = []
    patterns = [
        ("src/dl/outputs/exp02_deep_e2e/*/summary.json", "exp02_deep_e2e"),
        ("outputs/exp02_deep_e2e/*/summary.json", "exp02_deep_e2e"),
    ]
    for pattern, exp_id in patterns:
        for path in root.glob(pattern):
            with path.open("r", encoding="utf-8") as handle:
                row = json.load(handle)
            row["source_file"] = str(path.relative_to(root))
            row["exp_id"] = exp_id
            row["result_family"] = "deep"
            row["result_group"] = "deep/" + re.sub(r"_seed\d+$", "", str(row["run_name"]))
            rows.append(row)
    repro_files = list(root.glob("outputs/exp06_reproducibility/deep/*/summary.json"))
    if repro_files:
        seed42_main_patterns = [
            "outputs/exp02_deep_e2e/*_seed42/summary.json",
            "src/dl/outputs/exp02_deep_e2e/*_seed42/summary.json",
        ]
        seed42_main_files: list[Path] = []
        for pattern in seed42_main_patterns:
            seed42_main_files.extend(root.glob(pattern))
        for path in seed42_main_files + repro_files:
            with path.open("r", encoding="utf-8") as handle:
                row = json.load(handle)
            row["source_file"] = str(path.relative_to(root))
            row["exp_id"] = "exp06_reproducibility"
            row["result_family"] = "deep"
            row["result_group"] = "deep/" + re.sub(r"_seed\d+$", "", str(row["run_name"]))
            rows.append(row)
    return pd.DataFrame(rows)


def _collect_results() -> pd.DataFrame:
    frames = [_collect_grid_summaries(), _collect_deep_summaries()]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _aggregate_multiseed(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    group_cols = [c for c in ["exp_id", "result_family", "result_group"] if c in frame.columns]
    metric_cols = [
        c
        for c in [
            "macro_RMSE_pp",
            "macro_MRE_pct",
            "macro_R2",
            "macro_MaxRE_pct",
            "macro_RMSE",
            "macro_MAE",
            "mean_abs_sum_error",
            "max_abs_sum_error",
        ]
        if c in frame.columns
    ]
    if not group_cols or not metric_cols:
        return pd.DataFrame()
    grouped = frame.groupby(group_cols, as_index=False)[metric_cols].agg(["mean", "std", "min", "max"])
    grouped.columns = ["_".join([x for x in col if x]).strip("_") for col in grouped.columns.to_flat_index()]
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate experiment outputs into summary tables.")
    parser.parse_args()
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    frame = _collect_results()
    frame.to_csv(RESULTS_TSV, sep="\t", index=False)
    multi = _aggregate_multiseed(frame)
    multi.to_csv(RESULTS_MULTI_TSV, sep="\t", index=False)
    print(f"Wrote {RESULTS_TSV}")
    print(f"Wrote {RESULTS_MULTI_TSV}")


if __name__ == "__main__":
    main()
