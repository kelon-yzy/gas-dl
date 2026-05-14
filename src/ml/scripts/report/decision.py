"""Decision-table helpers for report figure generation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from patent_model.plotting_style import PROFILE_LABELS
from scripts.report.constants import COMBO_ALIASES, COMBO_LABELS, PREFERRED_COMBO, PREFERRED_PROFILE


def _normalize_combo_name(value: str) -> str:
    return COMBO_ALIASES.get(value, value)


def _split_combo_name(combo: str) -> tuple[str, str]:
    branch, meta = combo.split("_", 1)
    return branch, meta


def _candidate_label(profile: str, combo: str) -> str:
    profile_label = PROFILE_LABELS.get(profile, profile)
    combo_label = COMBO_LABELS.get(combo, combo)
    return f"{profile_label}\n{combo_label}"


def _highlight_mask(frame: pd.DataFrame) -> pd.Series:
    return (frame["profile"] == PREFERRED_PROFILE) & (frame["combo"] == PREFERRED_COMBO)


def _fallback_preferred_row(frame: pd.DataFrame) -> pd.Series:
    preferred = frame[_highlight_mask(frame)]
    if not preferred.empty:
        return preferred.iloc[0]
    return frame.sort_values(["macro_RMSE_pp", "macro_MRE_pct", "macro_R2"], ascending=[True, True, False]).iloc[0]


def _build_decision_summary(main_runs: pd.DataFrame, robustness_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = main_runs.copy()
    frame["combo"] = frame["combo"].map(_normalize_combo_name)
    if not robustness_summary.empty:
        merged = frame.merge(
            robustness_summary,
            on=["profile", "combo", "branch", "meta"],
            how="left",
        )
    else:
        merged = frame.copy()
        merged["detection_macro_RMSE_pp"] = np.nan
        merged["noise_macro_RMSE_pp_level_0"] = np.nan
        merged["noise_macro_RMSE_pp_worst"] = np.nan
        merged["noise_macro_RMSE_pp_increase"] = np.nan
        merged["pressure_macro_RMSE_pp_worst"] = np.nan
        merged["robustness_auc_proxy"] = np.nan

    merged["rmse_rank"] = merged["macro_RMSE_pp"].rank(method="min")
    merged["r2_rank"] = merged["macro_R2"].rank(method="min", ascending=False)
    merged["mre_rank"] = merged["macro_MRE_pct"].rank(method="min")
    merged["maxre_rank"] = merged["macro_MaxRE_pct"].rank(method="min")
    if merged["detection_macro_RMSE_pp"].notna().any():
        merged["robustness_rank"] = merged["detection_macro_RMSE_pp"].fillna(merged["macro_RMSE_pp"] * 10.0).rank(method="min")
    else:
        merged["robustness_rank"] = merged["macro_RMSE_pp"].rank(method="min")
    merged["overall_rank_score"] = merged[["rmse_rank", "r2_rank", "mre_rank", "maxre_rank", "robustness_rank"]].sum(axis=1)
    merged = merged.sort_values(["overall_rank_score", "macro_RMSE_pp", "macro_R2"], ascending=[True, True, False]).reset_index(drop=True)
    merged["recommendation"] = "other"
    merged.loc[_highlight_mask(merged), "recommendation"] = "recommended"

    top_candidates = merged.head(10).copy()
    top_candidates["recommendation"] = "candidate"
    preferred = top_candidates[_highlight_mask(top_candidates)]
    if preferred.empty:
        preferred_row = _fallback_preferred_row(merged)
        top_candidates = pd.concat([top_candidates, preferred_row.to_frame().T], ignore_index=True)
        top_candidates = top_candidates.drop_duplicates(subset=["profile", "combo"]).reset_index(drop=True)
    top_candidates.loc[_highlight_mask(top_candidates), "recommendation"] = "recommended"

    baseline_mask = (top_candidates["profile"] == "raw_tph") & (top_candidates["combo"] == "svr_ridge")
    top_candidates.loc[baseline_mask & (top_candidates["recommendation"] != "recommended"), "recommendation"] = "baseline"
    return merged, top_candidates

