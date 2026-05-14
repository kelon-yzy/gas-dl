"""Data loaders for report figure generation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from patent_model.plotting_style import PROFILE_ORDER
from scripts.report.constants import COMBO_ALIASES
from scripts.report.decision import _candidate_label, _normalize_combo_name, _split_combo_name


def _load_main_runs(outputs_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for profile_dir in sorted(outputs_root.glob("four_component_v3sync_model_grid_*")):
        profile = profile_dir.name.removeprefix("four_component_v3sync_model_grid_")
        if profile not in PROFILE_ORDER:
            continue
        for run_dir in sorted(profile_dir.iterdir()):
            summary_path = run_dir / "summary.json"
            if not run_dir.is_dir() or not summary_path.exists():
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            meta_model_type = str(summary["meta_model_type"])
            key = f"dynamic_{meta_model_type}"
            rows.append(
                {
                    "profile": profile,
                    "combo": run_dir.name,
                    "branch": summary["branch_model_type"],
                    "meta": meta_model_type,
                    "macro_RMSE_pp": float(summary[f"{key}_macro_RMSE_pp"]),
                    "macro_MRE_pct": float(summary[f"{key}_macro_MRE_pct"]),
                    "macro_R2": float(summary[f"{key}_macro_R2"]),
                    "macro_MaxRE_pct": float(summary[f"{key}_macro_MaxRE_pct"]),
                    "train_samples": int(summary["train_samples"]),
                    "test_samples": int(summary["test_samples"]),
                    "run_dir": str(run_dir),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise FileNotFoundError("No four-component experiment summaries were found.")
    return frame


def _load_best_component_frame(run_dir: Path) -> pd.DataFrame:
    metrics = pd.read_csv(run_dir / "component_metrics.csv")
    meta_model_type = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))["meta_model_type"]
    key = f"dynamic_{meta_model_type}"
    return metrics[metrics["model"] == key].copy()


def _load_environment_comparison(outputs_root: Path) -> pd.DataFrame:
    path = outputs_root / "environment_compensation_model_grid_four" / "best_profile_by_model.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing environment compensation comparison file: {path}")
    frame = pd.read_csv(path)
    frame["run"] = frame["run"].replace(COMBO_ALIASES)
    return frame


def _load_robustness_tables(robustness_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(robustness_root / "robustness_summary.csv")
    noise = pd.read_csv(robustness_root / "profile_environment_noise_metrics.csv")
    pressure = pd.read_csv(robustness_root / "profile_pressure_bin_metrics.csv")
    return summary, noise, pressure


def _load_robustness_model_summary(outputs_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for summary_path in sorted(outputs_root.glob("environment_compensation_robustness_four_main_*/robustness_summary.csv")):
        combo = _normalize_combo_name(summary_path.parent.name.removeprefix("environment_compensation_robustness_four_main_"))
        branch, meta = _split_combo_name(combo)
        summary = pd.read_csv(summary_path)
        summary.insert(1, "combo", combo)
        summary.insert(2, "branch", branch)
        summary.insert(3, "meta", meta)
        rows.extend(summary.to_dict("records"))
    if not rows:
        return pd.DataFrame(
            columns=[
                "profile",
                "combo",
                "branch",
                "meta",
                "detection_macro_RMSE_pp",
                "noise_macro_RMSE_pp_level_0",
                "noise_macro_RMSE_pp_worst",
                "noise_macro_RMSE_pp_increase",
                "pressure_macro_RMSE_pp_worst",
            ]
        )
    frame = pd.DataFrame(rows)
    frame["robustness_auc_proxy"] = frame[
        ["detection_macro_RMSE_pp", "noise_macro_RMSE_pp_worst", "pressure_macro_RMSE_pp_worst"]
    ].mean(axis=1)
    return frame


def _load_component_candidates(candidate_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for row in candidate_rows.itertuples():
        component_frame = _load_best_component_frame(Path(str(row.run_dir))).copy()
        component_frame.insert(0, "profile", row.profile)
        component_frame.insert(1, "combo", row.combo)
        component_frame.insert(2, "candidate_label", _candidate_label(row.profile, row.combo))
        rows.append(component_frame)
    return pd.concat(rows, ignore_index=True)

