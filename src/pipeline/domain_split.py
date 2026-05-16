from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DETECTION_STATUSES = ("synthetic_measurement",)
DETECTION_STAGES = ("distance_stage", "pressure_stage")
TERNARY_LABELS = {0: "low", 1: "mid", 2: "high"}
BINARY_LABELS = {0: "dry", 1: "wet"}


def _require_columns(frame: pd.DataFrame, columns: list[str], source: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def _load_detection_rows(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "condition_grid_v1.csv"
    frame = pd.read_csv(path)
    _require_columns(
        frame,
        ["sample_id", "mixture_id", "stage_id", "status", "T_C", "P_MPa", "H_RH"],
        str(path),
    )
    keep_mask = frame["status"].isin(DETECTION_STATUSES) & frame["stage_id"].isin(DETECTION_STAGES)
    filtered = frame.loc[keep_mask].reset_index(drop=True).copy()
    if filtered.empty:
        raise ValueError("condition_grid_v1.csv did not contain any detection samples for domain split.")
    return filtered


def _thresholds(mixture_frame: pd.DataFrame) -> dict[str, float]:
    t_q1, t_q2 = mixture_frame["T_C"].quantile([1.0 / 3.0, 2.0 / 3.0]).tolist()
    p_q1, p_q2 = mixture_frame["P_MPa"].quantile([1.0 / 3.0, 2.0 / 3.0]).tolist()
    rh_mid = float(mixture_frame["H_RH"].median())
    return {
        "T_C_q1": float(t_q1),
        "T_C_q2": float(t_q2),
        "P_MPa_q1": float(p_q1),
        "P_MPa_q2": float(p_q2),
        "H_RH_mid": rh_mid,
    }


def _assign_ternary(values: pd.Series, q1: float, q2: float) -> np.ndarray:
    numeric = values.to_numpy(dtype=float)
    return np.where(numeric <= q1, 0, np.where(numeric <= q2, 1, 2)).astype(int)


def _assign_binary(values: pd.Series, midpoint: float) -> np.ndarray:
    numeric = values.to_numpy(dtype=float)
    return np.where(numeric <= midpoint, 0, 1).astype(int)


def _initial_mixture_domains(detection_rows: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    mixture_frame = (
        detection_rows[["mixture_id", "T_C", "P_MPa", "H_RH"]]
        .drop_duplicates(subset=["mixture_id"])
        .reset_index(drop=True)
        .copy()
    )
    thresholds = _thresholds(mixture_frame)
    mixture_frame["t_bin"] = _assign_ternary(mixture_frame["T_C"], thresholds["T_C_q1"], thresholds["T_C_q2"])
    mixture_frame["p_bin"] = _assign_ternary(mixture_frame["P_MPa"], thresholds["P_MPa_q1"], thresholds["P_MPa_q2"])
    mixture_frame["rh_bin"] = _assign_binary(mixture_frame["H_RH"], thresholds["H_RH_mid"])
    mixture_frame["initial_domain"] = [
        f"T{t_bin}_P{p_bin}_RH{rh_bin}"
        for t_bin, p_bin, rh_bin in zip(mixture_frame["t_bin"], mixture_frame["p_bin"], mixture_frame["rh_bin"])
    ]
    mixture_frame["merged_domain"] = mixture_frame["initial_domain"]
    return mixture_frame, thresholds


def _domain_stats(assignment: pd.DataFrame) -> pd.DataFrame:
    return (
        assignment.groupby("merged_domain", as_index=False)
        .agg(
            sample_count=("sample_id", "size"),
            mixture_count=("mixture_id", "nunique"),
            t_center=("t_bin", "mean"),
            p_center=("p_bin", "mean"),
            rh_center=("rh_bin", "mean"),
        )
        .sort_values(["sample_count", "mixture_count", "merged_domain"], ascending=[True, True, True])
        .reset_index(drop=True)
    )


def _pick_merge_target(source_row: pd.Series, stats: pd.DataFrame) -> str:
    candidates = stats.loc[stats["merged_domain"] != source_row["merged_domain"]].copy()
    if candidates.empty:
        return str(source_row["merged_domain"])
    candidates["distance"] = (
        (candidates["t_center"] - float(source_row["t_center"])).abs()
        + (candidates["p_center"] - float(source_row["p_center"])).abs()
        + (candidates["rh_center"] - float(source_row["rh_center"])).abs()
    )
    best = candidates.sort_values(["distance", "sample_count", "mixture_count", "merged_domain"], ascending=[True, False, False, True]).iloc[0]
    return str(best["merged_domain"])


def _merge_sparse_domains(assignment: pd.DataFrame, min_domain_samples: int) -> pd.DataFrame:
    merged = assignment.copy()
    while True:
        stats = _domain_stats(merged)
        sparse = stats.loc[stats["sample_count"] < min_domain_samples]
        if sparse.empty or len(stats) <= 1:
            return merged
        source = sparse.iloc[0]
        target_domain = _pick_merge_target(source, stats)
        merged.loc[merged["merged_domain"] == str(source["merged_domain"]), "merged_domain"] = target_domain


def _finalize_domain_ids(assignment: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    stats = _domain_stats(assignment)
    ordered_domains = stats.sort_values(["t_center", "p_center", "rh_center", "merged_domain"]).reset_index(drop=True)
    mapping = {str(row["merged_domain"]): f"D{index + 1:02d}" for index, row in ordered_domains.iterrows()}
    finalized = assignment.copy()
    finalized["domain_id"] = finalized["merged_domain"].map(mapping)
    return finalized, mapping


def build_domain_artifacts(data_dir: str | Path, output_dir: str | Path, min_domain_samples: int = 500) -> tuple[pd.DataFrame, dict[str, object]]:
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    detection_rows = _load_detection_rows(data_path)
    mixture_domains, thresholds = _initial_mixture_domains(detection_rows)
    assignment = detection_rows.merge(
        mixture_domains[["mixture_id", "t_bin", "p_bin", "rh_bin", "initial_domain", "merged_domain"]],
        on="mixture_id",
        how="left",
        validate="many_to_one",
    )
    assignment = _merge_sparse_domains(assignment, min_domain_samples=min_domain_samples)
    assignment, merged_mapping = _finalize_domain_ids(assignment)
    assignment["T_bin_label"] = assignment["t_bin"].map(TERNARY_LABELS)
    assignment["P_bin_label"] = assignment["p_bin"].map(TERNARY_LABELS)
    assignment["RH_bin_label"] = assignment["rh_bin"].map(BINARY_LABELS)

    domain_rows = []
    for domain_id, domain_frame in assignment.groupby("domain_id", sort=True):
        domain_rows.append({
            "domain_id": domain_id,
            "sample_count": int(len(domain_frame)),
            "mixture_count": int(domain_frame["mixture_id"].nunique()),
            "t_bin_values": sorted(int(value) for value in domain_frame["t_bin"].unique()),
            "p_bin_values": sorted(int(value) for value in domain_frame["p_bin"].unique()),
            "rh_bin_values": sorted(int(value) for value in domain_frame["rh_bin"].unique()),
            "initial_domains": sorted(domain_frame["initial_domain"].dropna().astype(str).unique().tolist()),
            "merged_domain_key": str(domain_frame["merged_domain"].iloc[0]),
            "mean_T_C": float(domain_frame["T_C"].mean()),
            "mean_P_MPa": float(domain_frame["P_MPa"].mean()),
            "mean_H_RH": float(domain_frame["H_RH"].mean()),
        })

    definition = {
        "source_data_dir": str(data_path.resolve()),
        "source_condition_file": str((data_path / "condition_grid_v1.csv").resolve()),
        "selection": {"status_in": list(DETECTION_STATUSES), "stage_id_in": list(DETECTION_STAGES)},
        "binning": {
            "T_C": {"q1": thresholds["T_C_q1"], "q2": thresholds["T_C_q2"], "labels": ["low", "mid", "high"]},
            "P_MPa": {"q1": thresholds["P_MPa_q1"], "q2": thresholds["P_MPa_q2"], "labels": ["low", "mid", "high"]},
            "H_RH": {"mid": thresholds["H_RH_mid"], "labels": ["dry", "wet"]},
        },
        "min_domain_samples": int(min_domain_samples),
        "initial_domain_count": int(pd.Series(assignment["initial_domain"]).nunique()),
        "final_domain_count": int(pd.Series(assignment["domain_id"]).nunique()),
        "detection_sample_count": int(len(assignment)),
        "detection_mixture_count": int(assignment["mixture_id"].nunique()),
        "merged_domain_mapping": merged_mapping,
        "domains": domain_rows,
    }

    assignment_columns = ["sample_id", "mixture_id", "domain_id", "initial_domain", "merged_domain", "t_bin", "p_bin", "rh_bin", "T_bin_label", "P_bin_label", "RH_bin_label", "T_C", "P_MPa", "H_RH", "stage_id", "status"]
    assignment = assignment[assignment_columns].sort_values(["domain_id", "mixture_id", "sample_id"]).reset_index(drop=True)
    assignment.to_csv(output_path / "domain_assignments.csv", index=False)
    (output_path / "domain_definition.json").write_text(json.dumps(definition, indent=2, ensure_ascii=False), encoding="utf-8")
    return assignment, definition


def main() -> None:
    parser = argparse.ArgumentParser(description="Build T-P-RH holdout domain definitions for Track A.")
    parser.add_argument("--data-dir", default=str(ROOT / "outputs" / "exp01_traditional"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "exp04_domain"))
    parser.add_argument("--min-domain-samples", type=int, default=500)
    args = parser.parse_args()
    assignment, definition = build_domain_artifacts(args.data_dir, args.output_dir, min_domain_samples=args.min_domain_samples)
    print(json.dumps({"domain_count": definition["final_domain_count"], "sample_count": len(assignment), "output_dir": str(Path(args.output_dir).resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
