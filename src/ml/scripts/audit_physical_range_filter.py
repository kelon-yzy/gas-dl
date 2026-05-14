"""Audit strict physical range filtering before changing thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from patent_model.data_loader import PHYSICAL_RANGE_LIMITS, load_patent_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit physical-range strict filter impact.")
    parser.add_argument("--data-dir", default="../output")
    parser.add_argument("--feature-profile", default="derived_env_four")
    parser.add_argument("--metadata-filter", default="none")
    parser.add_argument("--label-closure-filter", default="none")
    parser.add_argument("--output-path")
    return parser


def _finite_values(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values)]


def _summary(series: pd.Series) -> dict[str, float | int | None]:
    values = _finite_values(series)
    if values.size == 0:
        return {"count": 0, "mean": None, "std": None, "min": None, "p50": None, "max": None}
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p50": float(np.percentile(values, 50)),
        "max": float(np.max(values)),
    }


def _histogram(series: pd.Series, bins: int = 10) -> dict[str, list[float] | list[int]]:
    values = _finite_values(series)
    if values.size == 0:
        return {"bin_edges": [], "counts": []}
    counts, edges = np.histogram(values, bins=bins)
    return {
        "bin_edges": [float(value) for value in edges],
        "counts": [int(value) for value in counts],
    }


def _stats_frame(dataset) -> pd.DataFrame:
    frame = dataset.metadata.reset_index(drop=True).copy()
    for component_index, component_name in enumerate(dataset.component_names):
        column = f"x_{component_name}"
        if column not in frame.columns:
            frame[column] = dataset.targets[:, component_index]
    frame["target_sum"] = dataset.targets.sum(axis=1)
    return frame


def build_physical_range_audit(
    data_dir: str | Path,
    feature_profile: str,
    metadata_filter: str = "none",
    label_closure_filter: str = "none",
) -> dict[str, object]:
    dataset = load_patent_dataset(
        data_dir,
        profile=feature_profile,
        metadata_filter=metadata_filter,
        physical_range_filter="none",
        label_closure_filter=label_closure_filter,
        duplicate_filter="none",
    )
    frame = _stats_frame(dataset)
    removed_mask = pd.Series(False, index=frame.index, dtype=bool)
    feature_reports: dict[str, object] = {}

    for column, (lower, upper) in PHYSICAL_RANGE_LIMITS.items():
        values = pd.to_numeric(frame[column], errors="coerce")
        finite_mask = pd.Series(np.isfinite(values.to_numpy(dtype=float)), index=frame.index)
        range_mask = pd.Series(values.between(lower, upper, inclusive="both"), index=frame.index).fillna(False)
        non_finite_mask = ~finite_mask
        out_of_range_mask = finite_mask & ~range_mask
        feature_removed_mask = non_finite_mask | out_of_range_mask
        removed_mask |= feature_removed_mask
        feature_reports[column] = {
            "limits": {"min": float(lower), "max": float(upper)},
            "non_finite_samples": int(non_finite_mask.sum()),
            "out_of_range_samples": int(out_of_range_mask.sum()),
            "removed_samples": int(feature_removed_mask.sum()),
            "all": _summary(frame[column]),
            "kept": _summary(frame.loc[~feature_removed_mask, column]),
            "removed": _summary(frame.loc[feature_removed_mask, column]),
        }

    removed = frame.loc[removed_mask]
    kept = frame.loc[~removed_mask]
    return {
        "data_dir": str(Path(data_dir).resolve()),
        "feature_profile": feature_profile,
        "metadata_filter": metadata_filter,
        "label_closure_filter": label_closure_filter,
        "physical_range_limits": {
            column: {"min": float(bounds[0]), "max": float(bounds[1])}
            for column, bounds in PHYSICAL_RANGE_LIMITS.items()
        },
        "samples": {
            "before": int(len(frame)),
            "would_keep": int(len(kept)),
            "would_remove": int(len(removed)),
            "remove_ratio": float(len(removed) / len(frame)) if len(frame) else 0.0,
        },
        "features": feature_reports,
        "h2_distribution": {
            "all": _summary(frame["x_H2"]),
            "kept": _summary(kept["x_H2"]),
            "removed": _summary(removed["x_H2"]),
            "removed_histogram": _histogram(removed["x_H2"]),
        },
        "sound_speed_distribution": {
            "all": _summary(frame["sound_speed"]),
            "kept": _summary(kept["sound_speed"]),
            "removed": _summary(removed["sound_speed"]),
            "removed_histogram": _histogram(removed["sound_speed"]),
        },
    }


def main(argv: list[str] | None = None) -> dict[str, object]:
    args = build_parser().parse_args(argv)
    audit = build_physical_range_audit(
        args.data_dir,
        feature_profile=args.feature_profile,
        metadata_filter=args.metadata_filter,
        label_closure_filter=args.label_closure_filter,
    )
    output_path = Path(args.output_path) if args.output_path else Path(args.data_dir) / "quality" / "physical_range_filter_audit.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    return audit


if __name__ == "__main__":
    main()
