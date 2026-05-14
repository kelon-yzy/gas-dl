"""批量运行四组分传统模型主网格。"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import pandas as pd

from patent_model.logging_utils import get_logger
from patent_model.modeling import MultiComponentPatentModel
from scripts._cli_utils import positive_int
from scripts.environment_compensation_common import add_model_args, profile_data_dir, resolve_feature_profile_name
from scripts.train_patent_model import build_model_config, build_parser as build_train_parser
from scripts.train_patent_model import prepare_training_data, run_training

PIPELINE_PARENT = Path(__file__).resolve().parents[2]
if str(PIPELINE_PARENT) not in sys.path:
  sys.path.insert(0, str(PIPELINE_PARENT))

from pipeline.cli_progress import build_cli_progress

logger = get_logger(__name__)


DEFAULT_PROFILES = ("v3_raw_no_env", "v3_raw_tph")
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


@dataclass(frozen=True)
class ExecutionPlanItem:
  profile: str
  branch_model_type: str
  meta_model_types: tuple[str, ...]


@dataclass(frozen=True)
class ExecutionTaskResult:
  rows: list[dict[str, object]]
  run_count: int


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="Run four-component traditional model grid.")
  parser.add_argument("--raw-data-dir", default="../output")
  parser.add_argument("--env-data-dir", default="../output")
  parser.add_argument("--output-root", default="outputs")
  parser.add_argument("--tag", default="v3sync")
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--train-limit", type=positive_int)
  parser.add_argument("--test-limit", type=positive_int)
  parser.add_argument("--n-perturbations", type=positive_int, default=24)
  parser.add_argument("--stacking-folds", type=positive_int, default=5)
  parser.add_argument("--test-ratio", type=float, default=0.2)
  parser.add_argument("--mc-env-samples", type=int, default=0)
  parser.add_argument("--mc-env-sigma-t", type=float, default=0.5)
  parser.add_argument("--mc-env-sigma-p", type=float, default=0.005)
  parser.add_argument("--mc-env-sigma-h", type=float, default=1.0)
  parser.add_argument("--metadata-filter", default="none")
  parser.add_argument("--physical-range-filter", default="none")
  parser.add_argument("--label-closure-filter", default="none")
  parser.add_argument("--duplicate-filter", default="per_mixture_limit")
  parser.add_argument("--duplicate-per-mixture-limit", type=positive_int, default=3)
  parser.add_argument("--duplicate-filter-seed", type=int, default=42)
  parser.add_argument("--profiles", nargs="*", default=list(DEFAULT_PROFILES))
  parser.add_argument("--combo-list", nargs="*", default=list(COMBO_ORDER))
  parser.add_argument("--max-workers", type=positive_int, default=1)
  parser.add_argument("--ui", action="store_true")
  parser.add_argument("--no-ui", action="store_true")
  add_model_args(parser, positive_int)
  return parser


def _combo_types(combo: str) -> tuple[str, str]:
  branch, meta = combo.split("_", 1)
  return branch, meta


def build_execution_plan(args: argparse.Namespace) -> list[ExecutionPlanItem]:
  plan: list[ExecutionPlanItem] = []
  for profile in args.profiles:
    grouped_meta: dict[str, list[str]] = {}
    for combo in args.combo_list:
      branch_model_type, meta_model_type = _combo_types(combo)
      grouped_meta.setdefault(branch_model_type, []).append(meta_model_type)
    for branch_model_type, meta_model_types in grouped_meta.items():
      plan.append(
        ExecutionPlanItem(
          profile=profile,
          branch_model_type=branch_model_type,
          meta_model_types=tuple(meta_model_types),
        )
      )
  return plan


def _build_train_args(args: argparse.Namespace, profile: str, branch_model_type: str, meta_model_type: str, output_dir: Path) -> argparse.Namespace:
  train_args = build_train_parser().parse_args([])
  train_args.data_dir = str(profile_data_dir(profile, Path(args.raw_data_dir), Path(args.env_data_dir)))
  train_args.output_dir = str(output_dir)
  train_args.feature_profile = resolve_feature_profile_name(profile, "four")
  train_args.component_mode = "four"
  train_args.test_ratio = args.test_ratio
  train_args.seed = args.seed
  train_args.train_limit = args.train_limit
  train_args.test_limit = args.test_limit
  train_args.n_perturbations = args.n_perturbations
  train_args.stacking_folds = args.stacking_folds
  train_args.perturbation_scale = getattr(args, "perturbation_scale", train_args.perturbation_scale)
  train_args.branch_model_type = branch_model_type
  train_args.meta_model_type = meta_model_type
  train_args.pls_n_components = args.pls_n_components
  train_args.xgb_n_estimators = args.xgb_n_estimators
  train_args.xgb_max_depth = args.xgb_max_depth
  train_args.xgb_learning_rate = args.xgb_learning_rate
  train_args.xgb_device = args.xgb_device
  train_args.xgb_n_jobs = args.xgb_n_jobs
  train_args.n_jobs = args.n_jobs
  train_args.mc_env_samples = args.mc_env_samples if profile == "v3_env" else 0
  train_args.mc_env_sigma_t = args.mc_env_sigma_t
  train_args.mc_env_sigma_p = args.mc_env_sigma_p
  train_args.mc_env_sigma_h = args.mc_env_sigma_h
  train_args.metadata_filter = args.metadata_filter
  train_args.physical_range_filter = args.physical_range_filter
  train_args.label_closure_filter = args.label_closure_filter
  train_args.duplicate_filter = args.duplicate_filter
  train_args.duplicate_per_mixture_limit = args.duplicate_per_mixture_limit
  train_args.duplicate_filter_seed = args.duplicate_filter_seed
  return train_args


def _row_from_summary(profile: str, combo: str, summary: dict[str, object], combo_dir: Path) -> dict[str, object]:
  row = {
    "profile": profile,
    "combo": combo,
    "branch": summary["branch_model_type"],
    "meta": summary["meta_model_type"],
    "data_dir": summary["data_dir"],
    "resolved_feature_profile": summary["resolved_feature_profile"],
    "macro_RMSE_pp": summary[f"dynamic_{summary['meta_model_type']}_macro_RMSE_pp"],
    "macro_MRE_pct": summary[f"dynamic_{summary['meta_model_type']}_macro_MRE_pct"],
    "macro_R2": summary[f"dynamic_{summary['meta_model_type']}_macro_R2"],
    "macro_MaxRE_pct": summary[f"dynamic_{summary['meta_model_type']}_macro_MaxRE_pct"],
    "train_samples": summary["train_samples"],
    "test_samples": summary["test_samples"],
    "fit_seconds": summary.get("fit_seconds"),
    "evaluate_seconds": summary.get("evaluate_seconds"),
    "write_seconds": summary.get("write_seconds"),
    "total_seconds": summary.get("total_seconds"),
    "n_jobs": summary.get("n_jobs"),
    "xgb_n_jobs": summary.get("xgb_n_jobs"),
    "prediction_cache_reused": summary.get("prediction_cache_reused"),
    "metadata_filter": summary.get("metadata_filter"),
    "physical_range_filter": summary.get("physical_range_filter"),
    "label_closure_filter": summary.get("label_closure_filter"),
    "duplicate_filter": summary.get("duplicate_filter"),
    "run_dir": str(combo_dir),
  }
  filter_report = summary.get("filter_report", {})
  if isinstance(filter_report, dict):
    for report_name in ("metadata_filter", "physical_range_filter", "label_closure_filter", "duplicate_filter"):
      report = filter_report.get(report_name)
      if not isinstance(report, dict):
        continue
      for field_name in (
        "before_samples",
        "after_samples",
        "removed_samples",
        "before_unique_mixtures",
        "after_unique_mixtures",
      ):
        if field_name in report:
          row[f"{report_name}_{field_name}"] = report[field_name]
  return row


def _run_plan_item(args: argparse.Namespace, output_root: Path, plan_item: ExecutionPlanItem, progress=None, completed_before: int = 0, total_runs: int = 0) -> ExecutionTaskResult:
  profile = plan_item.profile
  logger.info(
    "grid item profile=%s branch=%s meta_count=%d",
    profile,
    plan_item.branch_model_type,
    len(plan_item.meta_model_types),
  )
  profile_root = output_root / f"four_component_{args.tag}_grid_{profile}"
  profile_root.mkdir(parents=True, exist_ok=True)
  base_branch_model_args = _build_train_args(
    args,
    profile,
    plan_item.branch_model_type,
    plan_item.meta_model_types[0],
    profile_root / f"{plan_item.branch_model_type}_{plan_item.meta_model_types[0]}",
  )
  if progress is not None:
    progress.update_stage(
      stage="prepare_training_data",
      current_task=f"profile={profile} branch={plan_item.branch_model_type}",
      completed=completed_before,
      total=total_runs,
    )
  prepared_data = prepare_training_data(base_branch_model_args)
  branch_config_model = MultiComponentPatentModel(
    config=build_model_config(base_branch_model_args, prepared_data.feature_profile_name),
    component_names=prepared_data.train.component_names,
  )
  if progress is not None:
    progress.update_metric(train=prepared_data.train.n_samples, test=prepared_data.test.n_samples)
    progress.update_stage(
      stage="fit_branch_stage",
      current_task=f"profile={profile} branch={plan_item.branch_model_type}",
      completed=completed_before,
      total=total_runs,
    )
  branch_artifacts = branch_config_model.fit_branch_stage(prepared_data.train)
  branch_prediction_cache_holder = {}

  rows = []
  run_count = 0
  for meta_model_type in plan_item.meta_model_types:
    combo = f"{plan_item.branch_model_type}_{meta_model_type}"
    combo_dir = profile_root / combo
    train_args = _build_train_args(args, profile, plan_item.branch_model_type, meta_model_type, combo_dir)
    summary = run_training(
      train_args,
      prepared_data=prepared_data,
      branch_artifacts=branch_artifacts,
      prediction_cache_holder=branch_prediction_cache_holder,
      progress=progress,
      progress_context={
        "profile": profile,
        "combo": combo,
        "branch": plan_item.branch_model_type,
        "meta": meta_model_type,
        "completed": completed_before + run_count,
        "total": total_runs,
      },
    )
    rows.append(_row_from_summary(profile, combo, summary, combo_dir))
    run_count += 1
    if progress is not None:
      progress.log_message(
        f"completed {combo} macro_RMSE={summary[f'dynamic_{summary['meta_model_type']}_macro_RMSE_pp']:.4f} total={summary.get('total_seconds', 0.0):.2f}s cache={summary.get('prediction_cache_reused')}"
      )
      progress.update_stage(
        stage="write_summary",
        current_task=f"profile={profile} combo={combo}",
        completed=completed_before + run_count,
        total=total_runs,
      )
  return ExecutionTaskResult(rows=rows, run_count=run_count)


def main(argv: list[str] | None = None) -> dict[str, object]:
  args = build_parser().parse_args(argv)
  output_root = Path(args.output_root)
  output_root.mkdir(parents=True, exist_ok=True)

  rows = []
  run_count = 0
  plan = build_execution_plan(args)
  total_runs = sum(len(item.meta_model_types) for item in plan)
  progress = build_cli_progress(force=args.ui, disable=args.no_ui)
  progress.start_run(mode="traditional", title=args.tag, seed=args.seed, stage="plan")
  progress.update_metric(profiles=len(args.profiles), combos=total_runs, max_workers=args.max_workers)
  if args.max_workers == 1:
    for plan_item in plan:
      result = _run_plan_item(args, output_root, plan_item, progress=progress, completed_before=run_count, total_runs=total_runs)
      rows.extend(result.rows)
      run_count += result.run_count
  else:
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
      futures = [executor.submit(_run_plan_item, args, output_root, plan_item) for plan_item in plan]
      for future in futures:
        result = future.result()
        rows.extend(result.rows)
        run_count += result.run_count

  frame = pd.DataFrame(rows).sort_values(["profile", "combo"]).reset_index(drop=True)
  summary_path = output_root / f"four_component_{args.tag}_grid_summary.csv"
  frame.to_csv(summary_path, index=False)
  analysis = {
    "tag": args.tag,
    "run_count": run_count,
    "profiles": list(args.profiles),
    "combos": list(args.combo_list),
    "summary_csv": summary_path.name,
  }
  progress.finish_run(status="done", completed=run_count, total=total_runs, summary_csv=summary_path.name)
  (output_root / f"four_component_{args.tag}_grid_analysis.json").write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
  return analysis


if __name__ == "__main__":
  main()

