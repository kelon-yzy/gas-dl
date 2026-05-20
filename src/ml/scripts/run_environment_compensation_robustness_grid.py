"""批量运行环境补偿鲁棒性测试，并汇总全部模型组合结果。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts._cli_utils import positive_int
from scripts.environment_compensation_common import add_model_args, extend_model_cli_args
from scripts.run_environment_compensation_experiment import main as compensation_main
from scripts.run_environment_compensation_robustness import main as robustness_main


COMBO_ORDER = (
  "svr_ridge",
  "svr_pls",
  "svr_xgboost",
  "pls_ridge",
  "pls_pls",
  "pls_xgboost",
  "xgboost_ridge",
  "xgboost_pls",
  "xgboost_xgboost",
)


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="Run full environment-compensation robustness grid.")
  parser.add_argument("--raw-data-dir", default="../output")
  parser.add_argument("--env-data-dir", default="../../simulation-data/output_environment")
  parser.add_argument("--output-dir", default="outputs/environment_compensation_robustness_grid")
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--train-limit", type=positive_int)
  parser.add_argument("--test-limit", type=positive_int)
  parser.add_argument("--n-perturbations", type=positive_int, default=10)
  parser.add_argument("--stacking-folds", type=positive_int, default=3)
  parser.add_argument("--test-ratio", type=float, default=0.2)
  parser.add_argument("--noise-level-count", type=positive_int, default=5)
  parser.add_argument("--noise-pressure-mpa", type=float, default=0.101325)
  parser.add_argument("--noise-pressure-samples", type=positive_int)
  parser.add_argument("--pressure-bins", type=positive_int, default=6)
  parser.add_argument("--mc-env-samples", type=int, default=4)
  parser.add_argument("--mc-env-sigma-t", type=float, default=0.5)
  parser.add_argument("--mc-env-sigma-p", type=float, default=0.005)
  parser.add_argument("--mc-env-sigma-h", type=float, default=1.0)
  parser.add_argument("--combo-list", nargs="*", default=list(COMBO_ORDER))
  add_model_args(parser, positive_int)
  return parser


def _combo_types(combo: str) -> tuple[str, str]:
  branch, meta = combo.split("_", 1)
  return branch, meta


def _build_common_argv(args: argparse.Namespace) -> list[str]:
  argv = [
    "--raw-data-dir",
    str(args.raw_data_dir),
    "--env-data-dir",
    str(args.env_data_dir),
    "--component-mode",
    args.component_mode,
    "--seed",
    str(args.seed),
    "--n-perturbations",
    str(args.n_perturbations),
    "--stacking-folds",
    str(args.stacking_folds),
    "--test-ratio",
    str(args.test_ratio),
    "--mc-env-samples",
    str(args.mc_env_samples),
    "--mc-env-sigma-t",
    str(args.mc_env_sigma_t),
    "--mc-env-sigma-p",
    str(args.mc_env_sigma_p),
    "--mc-env-sigma-h",
    str(args.mc_env_sigma_h),
  ]
  if args.train_limit is not None:
    argv.extend(["--train-limit", str(args.train_limit)])
  if args.test_limit is not None:
    argv.extend(["--test-limit", str(args.test_limit)])
  if args.noise_pressure_samples is not None:
    argv.extend(["--noise-pressure-samples", str(args.noise_pressure_samples)])
  return argv


def _run_combo(args: argparse.Namespace, combo: str, compensation_dir: Path, robustness_dir: Path) -> dict[str, object]:
  branch_model_type, meta_model_type = _combo_types(combo)
  args.branch_model_type = branch_model_type
  args.meta_model_type = meta_model_type

  compensation_argv = _build_common_argv(args)
  compensation_argv.extend(["--output-dir", str(compensation_dir)])
  extend_model_cli_args(args, compensation_argv)
  compensation_main(compensation_argv)

  robustness_argv = _build_common_argv(args)
  robustness_argv.extend(
    [
      "--output-dir",
      str(robustness_dir),
      "--compensation-results",
      str(compensation_dir / "environment_compensation_summary.csv"),
      "--noise-level-count",
      str(args.noise_level_count),
      "--noise-pressure-mpa",
      str(args.noise_pressure_mpa),
      "--pressure-bins",
      str(args.pressure_bins),
    ]
  )
  extend_model_cli_args(args, robustness_argv)
  robustness_main(robustness_argv)

  return {
    "combo": combo,
    "branch_model_type": branch_model_type,
    "meta_model_type": meta_model_type,
    "compensation_dir": str(compensation_dir),
    "robustness_dir": str(robustness_dir),
  }


def _aggregate_results(output_dir: Path, combo_results: list[dict[str, object]]) -> tuple[pd.DataFrame, pd.DataFrame]:
  rows: list[pd.DataFrame] = []
  for result in combo_results:
    summary_path = Path(result["robustness_dir"]) / "robustness_summary.csv"
    summary = pd.read_csv(summary_path)
    summary.insert(1, "combo", result["combo"])
    summary.insert(2, "branch_model_type", result["branch_model_type"])
    summary.insert(3, "meta_model_type", result["meta_model_type"])
    rows.append(summary)

  full_summary = pd.concat(rows, ignore_index=True)
  ranking = (
    full_summary.groupby(["combo", "branch_model_type", "meta_model_type", "model_name"], as_index=False)
    .agg(
      mean_detection_rmse=("detection_macro_RMSE_pp", "mean"),
      mean_noise_worst_rmse=("noise_macro_RMSE_pp_worst", "mean"),
      mean_noise_delta_rmse=("noise_macro_RMSE_pp_increase", "mean"),
      mean_pressure_worst_rmse=("pressure_macro_RMSE_pp_worst", "mean"),
    )
  )
  ranking["detection_rank"] = ranking["mean_detection_rmse"].rank(method="min")
  ranking["noise_rank"] = ranking["mean_noise_worst_rmse"].rank(method="min")
  ranking["noise_delta_rank"] = ranking["mean_noise_delta_rmse"].rank(method="min")
  ranking["pressure_rank"] = ranking["mean_pressure_worst_rmse"].rank(method="min")
  ranking["overall_rank_score"] = ranking[["detection_rank", "noise_rank", "noise_delta_rank", "pressure_rank"]].sum(axis=1)
  ranking = ranking.sort_values(["overall_rank_score", "mean_detection_rmse", "mean_pressure_worst_rmse"]).reset_index(drop=True)

  full_summary.to_csv(output_dir / "robustness_grid_summary.csv", index=False)
  ranking.to_csv(output_dir / "robustness_grid_ranking.csv", index=False)
  return full_summary, ranking


def main(argv: list[str] | None = None) -> dict[str, object]:
  args = build_parser().parse_args(argv)
  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  combo_results = []
  for combo in args.combo_list:
    compensation_dir = output_dir / f"compensation_{combo}"
    robustness_dir = output_dir / f"robustness_{combo}"
    combo_results.append(_run_combo(args, combo, compensation_dir, robustness_dir))

  full_summary, ranking = _aggregate_results(output_dir, combo_results)
  analysis = {
    "component_mode": args.component_mode,
    "combo_count": len(combo_results),
    "profiles": sorted(full_summary["profile"].unique().tolist()),
    "best_combo": None if ranking.empty else str((ranking[ranking["model_name"] == "fused"] if (ranking["model_name"] == "fused").any() else ranking).iloc[0]["combo"]),
    "outputs": [
      "robustness_grid_summary.csv",
      "robustness_grid_ranking.csv",
    ],
  }
  (output_dir / "robustness_grid_analysis.json").write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
  return analysis


if __name__ == "__main__":
  main()

