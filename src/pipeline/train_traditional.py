from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
ML_ROOT = SRC_ROOT / "ml"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from pipeline.cli_progress import build_cli_progress
from scripts.run_four_component_model_grid import main as run_four_component_model_grid


def main() -> None:
    parser = argparse.ArgumentParser(description="Run G1 traditional-model grid for V3.1 dual-channel data.")
    parser.add_argument("--data-dir", default=str(ROOT / "outputs" / "exp01_traditional"))
    parser.add_argument("--output-root", default=str(ROOT / "outputs" / "exp01_traditional"))
    parser.add_argument("--tag", default="formal")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--profiles", nargs="*", default=None)
    parser.add_argument("--combo-list", nargs="*", default=None)
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--xgb-n-jobs", type=int, default=None)
    parser.add_argument("--n-perturbations", type=int, default=None)
    parser.add_argument("--stacking-folds", type=int, default=None)
    parser.add_argument("--stage-filter", default="stable", choices=("none", "stable"))
    parser.add_argument("--ui", action="store_true")
    parser.add_argument("--no-ui", action="store_true")
    args = parser.parse_args()

    progress = build_cli_progress(force=args.ui, disable=args.no_ui)
    progress.start_run(mode="traditional", title=args.tag, seed=args.seed, stage="dispatch")
    progress.update_metric(
        profiles=len(args.profiles or []),
        combos=len(args.combo_list or []),
        max_workers=args.max_workers,
    )

    argv = [
        "--raw-data-dir", args.data_dir,
        "--env-data-dir", args.data_dir,
        "--output-root", args.output_root,
        "--tag", args.tag,
        "--seed", str(args.seed),
        "--component-mode", "four",
        "--max-workers", str(args.max_workers),
        "--stage-filter", args.stage_filter,
    ]
    if args.train_limit is not None:
        argv.extend(["--train-limit", str(args.train_limit)])
    if args.test_limit is not None:
        argv.extend(["--test-limit", str(args.test_limit)])
    if args.profiles:
        argv.append("--profiles")
        argv.extend(args.profiles)
    if args.combo_list:
        argv.append("--combo-list")
        argv.extend(args.combo_list)
    if args.n_jobs is not None:
        argv.extend(["--n-jobs", str(args.n_jobs)])
    if args.xgb_n_jobs is not None:
        argv.extend(["--xgb-n-jobs", str(args.xgb_n_jobs)])
    if args.n_perturbations is not None:
        argv.extend(["--n-perturbations", str(args.n_perturbations)])
    if args.stacking_folds is not None:
        argv.extend(["--stacking-folds", str(args.stacking_folds)])
    if args.ui:
        argv.append("--ui")
    if args.no_ui:
        argv.append("--no-ui")

    result = run_four_component_model_grid(argv)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    progress.finish_run(
        status="done",
        run_count=result.get("run_count"),
        profiles=len(result.get("profiles", [])),
        combos=len(result.get("combos", [])),
    )
    print(json.dumps({"finished_at": stamp, "result": result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

