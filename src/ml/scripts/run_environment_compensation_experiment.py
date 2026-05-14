"""Run raw vs environment-compensated V1 traditional-model experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from patent_model.feature_profiles import FEATURE_PROFILES
from patent_model.logging_utils import get_logger
from scripts._cli_utils import positive_int
from scripts.environment_compensation_common import (
    PROFILES,
    add_model_args,
    build_meta_key,
    extend_model_cli_args,
    profile_data_dir,
    require_known_profile_mode,
    resolve_feature_profile_name,
)
from scripts.train_patent_model import main as train_main


logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run V1 environment compensation comparison.")
    parser.add_argument("--raw-data-dir", default="../output")
    parser.add_argument("--env-data-dir", default="../../simulation-data/output_environment")
    parser.add_argument("--output-dir", default="outputs/environment_compensation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-limit", type=positive_int)
    parser.add_argument("--test-limit", type=positive_int)
    parser.add_argument("--n-perturbations", type=positive_int, default=24)
    parser.add_argument("--stacking-folds", type=positive_int, default=5)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--mc-env-samples", type=int, default=4)
    parser.add_argument("--mc-env-sigma-t", type=float, default=0.5)
    parser.add_argument("--mc-env-sigma-p", type=float, default=0.005)
    parser.add_argument("--mc-env-sigma-h", type=float, default=1.0)
    add_model_args(parser, positive_int)
    return parser


def _train_profile(args: argparse.Namespace, profile: str, output_dir: Path) -> dict[str, object]:
    logger.info("compensation profile=%s", profile)
    data_dir = profile_data_dir(profile, Path(args.raw_data_dir), Path(args.env_data_dir))
    argv = [
        "--data-dir",
        str(data_dir),
        "--output-dir",
        str(output_dir / profile),
        "--feature-profile",
        resolve_feature_profile_name(profile, args.component_mode),
        "--test-ratio",
        str(args.test_ratio),
        "--seed",
        str(args.seed),
        "--n-perturbations",
        str(args.n_perturbations),
        "--stacking-folds",
        str(args.stacking_folds),
    ]
    extend_model_cli_args(args, argv)
    if args.train_limit is not None:
        argv.extend(["--train-limit", str(args.train_limit)])
    if args.test_limit is not None:
        argv.extend(["--test-limit", str(args.test_limit)])
    if profile == "derived_env_mc_aug":
        argv.extend(
            [
                "--mc-env-samples",
                str(args.mc_env_samples),
                "--mc-env-sigma-t",
                str(args.mc_env_sigma_t),
                "--mc-env-sigma-p",
                str(args.mc_env_sigma_p),
                "--mc-env-sigma-h",
                str(args.mc_env_sigma_h),
            ]
        )
    return train_main(argv)


def main(argv: list[str] | None = None) -> dict[str, object]:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_summaries = {}
    metric_frames = []
    for profile in PROFILES:
        require_known_profile_mode(profile, args.component_mode)
        run_summaries[profile] = _train_profile(args, profile, output_dir)
        metrics = pd.read_csv(output_dir / profile / "component_metrics.csv")
        metrics.insert(0, "profile", profile)
        metrics.insert(1, "branch_model_type", args.branch_model_type)
        metrics.insert(2, "meta_model_type", args.meta_model_type)
        metric_frames.append(metrics)

    comparison = pd.concat(metric_frames, ignore_index=True)
    comparison.to_csv(output_dir / "environment_compensation_summary.csv", index=False)

    summary = {
        "profiles": list(PROFILES),
        "split_policy": "grouped_by_mixture_id",
        "calibration_status": "pending",
        "branch_model_type": args.branch_model_type,
        "meta_model_type": args.meta_model_type,
        "component_mode": args.component_mode,
        "meta_key": build_meta_key(args.meta_model_type),
        "main_metric": f"dynamic_{args.meta_model_type} macro RMSE_pp",
        "runs": run_summaries,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


if __name__ == "__main__":
    main()
