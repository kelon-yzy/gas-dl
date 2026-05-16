from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
ML_ROOT = SRC_ROOT / "ml"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

logger = logging.getLogger(__name__)

from patent_model.dataset import PatentDataset
from patent_model.fault_labels import build_observed_fault_labels
from patent_model.data_loader import load_patent_dataset
from scripts.environment_compensation_common import add_model_args, build_meta_key, resolve_feature_profile_name
from scripts.run_four_component_model_grid import _combo_types
from scripts.train_patent_model import PreparedTrainingData, build_parser as build_train_parser, run_training
from pipeline.domain_split import build_domain_artifacts


DEFAULT_REPRESENTATIVE_COMBOS = ("pls_ridge", "xgboost_xgboost")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _subset_by_domain(dataset: PatentDataset, domain_assignment: pd.DataFrame, domain_id: str) -> tuple[PatentDataset, PatentDataset]:
    assignment = domain_assignment[["sample_id", "domain_id"]].drop_duplicates(subset=["sample_id"])
    sample_to_domain = assignment.set_index("sample_id")["domain_id"]
    sample_ids = pd.Series(dataset.sample_ids.astype(str))
    domains = sample_ids.map(sample_to_domain)
    if domains.isna().any():
        raise ValueError("Some training samples are missing domain assignment.")
    holdout_mask = domains.eq(domain_id).to_numpy(dtype=bool)
    train_idx = (~holdout_mask).nonzero()[0]
    test_idx = holdout_mask.nonzero()[0]
    if train_idx.size == 0 or test_idx.size == 0:
        raise ValueError(f"Holdout split for domain {domain_id} produced an empty train or test set.")
    return dataset.subset(train_idx), dataset.subset(test_idx)


def _build_train_args(args: argparse.Namespace, output_dir: Path, combo: str) -> argparse.Namespace:
    branch_model_type, meta_model_type = _combo_types(combo)
    train_args = build_train_parser().parse_args([])
    train_args.data_dir = str(args.data_dir)
    train_args.output_dir = str(output_dir)
    train_args.feature_profile = resolve_feature_profile_name(args.profile, "four")
    train_args.component_mode = "four"
    # test_ratio 不设置：domain holdout 场景下 split 由域指派完成，不启用 prepare_training_data 的 random split
    train_args.seed = args.seed
    # train_limit / test_limit 不在 domain holdout 场景生效（split 由域指派完成，skip prepare_training_data）
    train_args.n_perturbations = args.n_perturbations
    train_args.stacking_folds = args.stacking_folds
    train_args.perturbation_scale = args.perturbation_scale
    train_args.branch_model_type = branch_model_type
    train_args.meta_model_type = meta_model_type
    train_args.pls_n_components = args.pls_n_components
    train_args.xgb_n_estimators = args.xgb_n_estimators
    train_args.xgb_max_depth = args.xgb_max_depth
    train_args.xgb_learning_rate = args.xgb_learning_rate
    train_args.xgb_device = args.xgb_device
    train_args.xgb_n_jobs = args.xgb_n_jobs
    train_args.n_jobs = args.n_jobs
    train_args.mc_env_samples = 0
    train_args.mc_env_sigma_t = 0.5
    train_args.mc_env_sigma_p = 0.005
    train_args.mc_env_sigma_h = 1.0
    train_args.metadata_filter = args.metadata_filter
    train_args.stage_filter = args.stage_filter
    train_args.physical_range_filter = args.physical_range_filter
    train_args.label_closure_filter = args.label_closure_filter
    train_args.duplicate_filter = args.duplicate_filter
    train_args.duplicate_per_mixture_limit = args.duplicate_per_mixture_limit
    train_args.duplicate_filter_seed = args.duplicate_filter_seed
    return train_args


def _summary_rows(domain_id: str, combo: str, summary: dict[str, object], train_set: PatentDataset, test_set: PatentDataset) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model_name in ("acoustic", "optical", "thermal", build_meta_key(str(summary["meta_model_type"]))):
        rows.append(
            {
                "domain_id": domain_id,
                "combo": combo,
                "branch": summary["branch_model_type"],
                "meta": summary["meta_model_type"],
                "model_name": model_name,
                "holdout_macro_RMSE_pp": float(summary[f"{model_name}_macro_RMSE_pp"]),
                "holdout_macro_R2": float(summary.get(f"{model_name}_macro_R2", float("nan"))),
                "train_samples": int(train_set.n_samples),
                "holdout_samples": int(test_set.n_samples),
                "train_mixtures": int(train_set.metadata["mixture_id"].nunique()),
                "holdout_mixtures": int(test_set.metadata["mixture_id"].nunique()),
                "fit_seconds": summary.get("fit_seconds"),
                "evaluate_seconds": summary.get("evaluate_seconds"),
                "total_seconds": summary.get("total_seconds"),
                "n_jobs": summary.get("n_jobs"),
                "xgb_n_jobs": summary.get("xgb_n_jobs"),
                "metadata_filter": summary.get("metadata_filter"),
                "stage_filter": summary.get("stage_filter"),
                "physical_range_filter": summary.get("physical_range_filter"),
                "label_closure_filter": summary.get("label_closure_filter"),
                "duplicate_filter": summary.get("duplicate_filter"),
                "best_model_by_macro_RMSE_pp": summary.get("best_model_by_macro_RMSE_pp"),
            }
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Track A domain holdout for representative traditional models.")
    parser.add_argument("--data-dir", default=str(ROOT / "outputs" / "exp01_traditional"))
    parser.add_argument("--output-root", default=str(ROOT / "outputs" / "exp04_domain"))
    parser.add_argument("--profile", default="v3_raw_no_env")
    parser.add_argument("--combo-list", nargs="*", default=list(DEFAULT_REPRESENTATIVE_COMBOS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--n-perturbations", type=positive_int, default=24)
    parser.add_argument("--stacking-folds", type=positive_int, default=5)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--metadata-filter", default="detection")
    parser.add_argument("--stage-filter", default="stable", choices=("none", "stable"))
    parser.add_argument("--physical-range-filter", default="none")
    parser.add_argument("--label-closure-filter", default="none")
    parser.add_argument("--duplicate-filter", default="per_mixture_limit")
    parser.add_argument("--duplicate-per-mixture-limit", type=positive_int, default=3)
    parser.add_argument("--duplicate-filter-seed", type=int, default=42)
    parser.add_argument("--min-domain-samples", type=positive_int, default=500)
    add_model_args(parser, positive_int)
    return parser


def main(argv: list[str] | None = None) -> dict[str, object]:
    args = build_parser().parse_args(argv)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    domain_assignment, domain_definition = build_domain_artifacts(args.data_dir, output_root, min_domain_samples=args.min_domain_samples)
    domain_expected = {d["domain_id"]: d["sample_count"] for d in domain_definition["domains"]}
    feature_profile_name = resolve_feature_profile_name(args.profile, "four")
    base_dataset = load_patent_dataset(
        args.data_dir,
        profile=feature_profile_name,
        metadata_filter=args.metadata_filter,
        stage_filter=args.stage_filter,
        physical_range_filter=args.physical_range_filter,
        label_closure_filter=args.label_closure_filter,
        duplicate_filter=args.duplicate_filter,
        duplicate_per_mixture_limit=args.duplicate_per_mixture_limit,
        duplicate_filter_seed=args.duplicate_filter_seed,
    )
    base_dataset = base_dataset.with_fault_labels(build_observed_fault_labels(base_dataset))

    rows: list[dict[str, object]] = []
    domain_actual_stats: list[dict[str, object]] = []
    for domain_id in sorted(domain_assignment["domain_id"].dropna().astype(str).unique().tolist()):
        train_set, test_set = _subset_by_domain(base_dataset, domain_assignment, domain_id)
        actual_total = train_set.n_samples + test_set.n_samples
        expected = domain_expected.get(domain_id, 0)
        logger.info("domain=%s expected_samples=%d actual_samples=%d train=%d test=%d",
                     domain_id, expected, actual_total, train_set.n_samples, test_set.n_samples)
        if actual_total < args.min_domain_samples:
            logger.warning("domain=%s has only %d actual samples (min_domain_samples=%d), results may be unreliable",
                           domain_id, actual_total, args.min_domain_samples)
        domain_actual_stats.append({
            "domain_id": domain_id,
            "expected_samples": expected,
            "actual_samples": actual_total,
            "train_samples": int(train_set.n_samples),
            "test_samples": int(test_set.n_samples),
        })
        domain_dir = output_root / domain_id
        domain_dir.mkdir(parents=True, exist_ok=True)
        for combo in args.combo_list:
            combo_dir = domain_dir / combo
            train_args = _build_train_args(args, combo_dir, combo)
            prepared = PreparedTrainingData(
                data_dir=Path(args.data_dir),
                feature_profile_name=feature_profile_name,
                train_original_samples=train_set.n_samples,
                train=train_set,
                test=test_set,
            )
            summary = run_training(train_args, prepared_data=prepared, branch_artifacts=None, prediction_cache_holder={})
            combo_rows = _summary_rows(domain_id, combo, summary, train_set, test_set)
            for row in combo_rows:
                row["run_dir"] = str(combo_dir)
            rows.extend(combo_rows)

    summary_frame = pd.DataFrame(rows).sort_values(["combo", "model_name", "domain_id"]).reset_index(drop=True)
    if not summary_frame.empty:
        combo_baseline = summary_frame.groupby(["combo", "model_name"], as_index=False)["holdout_macro_RMSE_pp"].mean().rename(columns={"holdout_macro_RMSE_pp": "mean_holdout_macro_RMSE_pp"})
        summary_frame = summary_frame.merge(combo_baseline, on=["combo", "model_name"], how="left")
        summary_frame["deviation_from_cross_domain_mean"] = summary_frame["holdout_macro_RMSE_pp"] - summary_frame["mean_holdout_macro_RMSE_pp"]

    summary_path = output_root / "domain_holdout_summary.csv"
    summary_frame.to_csv(summary_path, index=False)
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=str(ROOT)).strip()
    except Exception:
        git_hash = "unknown"
    analysis = {
        "exp_id": "exp04_domain",
        "profile": args.profile,
        "seed": args.seed,
        "combos": list(args.combo_list),
        "domain_count": int(domain_definition["final_domain_count"]),
        "min_domain_samples": args.min_domain_samples,
        "summary_csv": summary_path.name,
        "domain_definition_json": "domain_definition.json",
        "git_commit": git_hash,
        "data_filters": {
            "metadata_filter": args.metadata_filter,
            "physical_range_filter": args.physical_range_filter,
            "label_closure_filter": args.label_closure_filter,
            "duplicate_filter": args.duplicate_filter,
            "duplicate_per_mixture_limit": args.duplicate_per_mixture_limit,
            "duplicate_filter_seed": args.duplicate_filter_seed,
        },
        "model_hyperparams": {
            "pls_n_components": args.pls_n_components,
            "xgb_n_estimators": args.xgb_n_estimators,
            "xgb_max_depth": args.xgb_max_depth,
            "xgb_learning_rate": args.xgb_learning_rate,
            "xgb_n_jobs": args.xgb_n_jobs,
            "xgb_device": args.xgb_device,
            "n_jobs": args.n_jobs,
            "n_perturbations": args.n_perturbations,
            "stacking_folds": args.stacking_folds,
        },
        "domain_actual_samples": domain_actual_stats,
        "total_actual_samples": int(base_dataset.n_samples),
        "total_actual_mixtures": int(base_dataset.metadata["mixture_id"].nunique()),
    }
    (output_root / "analysis.json").write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
    return analysis


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False, indent=2))
