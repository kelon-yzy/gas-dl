from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
ML_ROOT = SRC_ROOT / "ml"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

REQUIRED_LOG_COLUMNS = ("epoch", "train_loss", "val_loss", "val_macro_RMSE", "val_macro_MAE")
OPTIONAL_LOG_COLUMNS = ("lr",)
SUPPORTED_FORMATS = ("png", "svg", "pdf")


@dataclass(frozen=True)
class TrainingRunLog:
    run_dir: Path
    relative_name: str
    display_name: str
    frame: pd.DataFrame
    summary: dict[str, Any] | None
    config: dict[str, Any] | None
    best_epoch: int
    best_val_macro_rmse: float


def _resolve_cli_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def _safe_slug(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", value.strip())
    return cleaned.strip("_") or "run"


def _run_artifact_name(root: Path, run_dir: Path) -> str:
    relative = run_dir.relative_to(root)
    parts = [_safe_slug(part) for part in relative.parts if part not in ("", ".")]
    if not parts:
        return _safe_slug(run_dir.name)
    return "__".join(parts)


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _build_warning(run_dir: Path, reason: str) -> str:
    return f"跳过 {run_dir}: {reason}"


def _load_training_run(root: Path, run_dir: Path) -> tuple[TrainingRunLog | None, str | None]:
    log_path = run_dir / "train_log.csv"
    try:
        frame = pd.read_csv(log_path)
    except pd.errors.EmptyDataError:
        return None, _build_warning(run_dir, "train_log.csv 为空")
    if frame.empty:
        return None, _build_warning(run_dir, "train_log.csv 没有有效行")

    missing = [column for column in REQUIRED_LOG_COLUMNS if column not in frame.columns]
    if missing:
        return None, _build_warning(run_dir, f"缺少必要列: {', '.join(missing)}")

    ordered = frame.sort_values("epoch").reset_index(drop=True)
    best_idx = ordered["val_macro_RMSE"].idxmin()
    best_epoch = int(ordered.loc[best_idx, "epoch"])
    best_val_macro_rmse = float(ordered.loc[best_idx, "val_macro_RMSE"])
    return TrainingRunLog(
        run_dir=run_dir,
        relative_name=_run_artifact_name(root, run_dir),
        display_name=run_dir.name,
        frame=ordered,
        summary=_load_json_if_exists(run_dir / "summary.json"),
        config=_load_json_if_exists(run_dir / "config.json"),
        best_epoch=best_epoch,
        best_val_macro_rmse=best_val_macro_rmse,
    ), None


def find_training_run_dirs(root: Path) -> list[Path]:
    return sorted(path.parent for path in root.rglob("train_log.csv"))


def _import_plotting_modules():
    import matplotlib

    if "matplotlib.pyplot" not in sys.modules:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from patent_model.plotting_style import setup_chinese_fonts

    setup_chinese_fonts()
    return plt


def _save_figure(fig, base_path: Path, formats: tuple[str, ...], dpi: int) -> list[Path]:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    for fmt in formats:
        target = base_path.with_suffix(f".{fmt}")
        save_kwargs: dict[str, Any] = {}
        if fmt == "png":
            save_kwargs["dpi"] = dpi
        fig.savefig(target, bbox_inches="tight", **save_kwargs)
        saved_paths.append(target)
    return saved_paths


def _build_run_title(run: TrainingRunLog) -> str:
    run_name = run.display_name
    if run.summary and run.summary.get("run_name"):
        run_name = str(run.summary["run_name"])
    model_name = None
    if run.summary and run.summary.get("model"):
        model_name = str(run.summary["model"])
    elif run.config and run.config.get("model", {}).get("name"):
        model_name = str(run.config["model"]["name"])
    if model_name:
        return f"{run_name} | {model_name}"
    return run_name


def plot_training_run(run: TrainingRunLog, output_dir: Path, formats: tuple[str, ...], dpi: int) -> list[Path]:
    plt = _import_plotting_modules()
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    epochs = run.frame["epoch"]

    axes[0, 0].plot(epochs, run.frame["train_loss"], label="train_loss", color="#1f77b4", linewidth=2.0)
    axes[0, 0].plot(epochs, run.frame["val_loss"], label="val_loss", color="#ff7f0e", linewidth=2.0)
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].legend()

    axes[0, 1].plot(epochs, run.frame["val_macro_RMSE"], color="#d62728", linewidth=2.0)
    axes[0, 1].scatter([run.best_epoch], [run.best_val_macro_rmse], color="#111111", zorder=3)
    axes[0, 1].axvline(run.best_epoch, color="#666666", linestyle="--", linewidth=1.0)
    axes[0, 1].annotate(
        f"best={run.best_epoch}\nRMSE={run.best_val_macro_rmse:.4f}",
        xy=(run.best_epoch, run.best_val_macro_rmse),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#cccccc", "alpha": 0.9},
    )
    axes[0, 1].set_title("Validation macro_RMSE")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("macro_RMSE")
    axes[0, 1].grid(alpha=0.25)

    axes[1, 0].plot(epochs, run.frame["val_macro_MAE"], color="#2ca02c", linewidth=2.0)
    axes[1, 0].set_title("Validation macro_MAE")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("macro_MAE")
    axes[1, 0].grid(alpha=0.25)

    if "lr" in run.frame.columns:
        axes[1, 1].plot(epochs, run.frame["lr"], color="#9467bd", linewidth=2.0)
        axes[1, 1].set_title("Learning Rate")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("lr")
        axes[1, 1].grid(alpha=0.25)
    else:
        axes[1, 1].set_visible(False)

    fig.suptitle(_build_run_title(run), fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    saved_paths = _save_figure(fig, output_dir / f"{run.relative_name}_training_curves", formats, dpi)
    plt.close(fig)
    return saved_paths


def plot_aggregate_val_rmse(runs: list[TrainingRunLog], output_dir: Path, formats: tuple[str, ...], dpi: int) -> list[Path]:
    plt = _import_plotting_modules()
    fig, ax = plt.subplots(figsize=(12, 7))
    for run in runs:
        ax.plot(run.frame["epoch"], run.frame["val_macro_RMSE"], linewidth=1.8, label=run.display_name)
    ax.set_title("All Runs Validation macro_RMSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("macro_RMSE")
    ax.grid(alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    saved_paths = _save_figure(fig, output_dir / "all_runs_val_macro_RMSE", formats, dpi)
    plt.close(fig)
    return saved_paths


def generate_training_curve_artifacts(
    root: Path,
    output_dir: Path,
    formats: tuple[str, ...] = ("png", "svg"),
    dpi: int = 300,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    runs: list[TrainingRunLog] = []
    generated_files: list[str] = []

    for run_dir in find_training_run_dirs(root):
        run, warning = _load_training_run(root, run_dir)
        if warning is not None:
            print(warning)
            warnings.append(warning)
            continue
        runs.append(run)

    for run in runs:
        generated_files.extend(str(path) for path in plot_training_run(run, output_dir, formats, dpi))

    if runs:
        generated_files.extend(str(path) for path in plot_aggregate_val_rmse(runs, output_dir, formats, dpi))
    else:
        warning = f"未找到可绘图的训练日志: {root}"
        print(warning)
        warnings.append(warning)

    return {
        "root": str(root),
        "output_dir": str(output_dir),
        "processed_runs": len(runs),
        "skipped_runs": len(warnings) if runs or warnings else 0,
        "warnings": warnings,
        "generated_files": generated_files,
        "formats": list(formats),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="批量输出深度学习训练曲线图。")
    parser.add_argument("--root", default="outputs", help="递归扫描包含 train_log.csv 的输出根目录")
    parser.add_argument("--output-dir", default="outputs/deep_training_curves", help="图表输出目录")
    parser.add_argument("--formats", nargs="+", default=["png", "svg"], choices=SUPPORTED_FORMATS, help="输出格式列表")
    parser.add_argument("--dpi", type=int, default=300, help="PNG 输出分辨率")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    result = generate_training_curve_artifacts(
        root=_resolve_cli_path(args.root),
        output_dir=_resolve_cli_path(args.output_dir),
        formats=tuple(args.formats),
        dpi=int(args.dpi),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()
