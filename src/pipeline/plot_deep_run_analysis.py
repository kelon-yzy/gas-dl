from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
ML_ROOT = SRC_ROOT / "ml"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from pipeline.plot_deep_training_curves import (
    SUPPORTED_FORMATS,
    _import_plotting_modules,
    _load_training_run,
    _resolve_cli_path,
    _save_figure,
    plot_training_run,
)

COMPONENTS = ("H2", "CH4", "CO2", "N2")


def _load_predictions(run_dir: Path) -> pd.DataFrame:
    return pd.read_csv(run_dir / "predictions.csv")


def _component_arrays(frame: pd.DataFrame, component: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    true_values = frame[f"y_true_{component}"].to_numpy(dtype=float)
    pred_values = frame[f"y_pred_{component}"].to_numpy(dtype=float)
    errors = pred_values - true_values
    return true_values, pred_values, errors


def _component_stats(frame: pd.DataFrame, component: str) -> dict[str, float]:
    true_values, pred_values, errors = _component_arrays(frame, component)
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    mae = float(np.mean(np.abs(errors)))
    bias = float(np.mean(errors))
    ss_tot = float(np.sum(np.square(true_values - np.mean(true_values))))
    if ss_tot == 0.0:
        r2 = math.nan
    else:
        r2 = float(1.0 - np.sum(np.square(errors)) / ss_tot)
    return {
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "r2": r2,
        "max_abs_error": float(np.max(np.abs(errors))),
    }


def _stats_text(stats: dict[str, float]) -> str:
    r2_text = "N/A" if math.isnan(stats["r2"]) else f"{stats['r2']:.4f}"
    return "\n".join(
        (
            f"RMSE={stats['rmse']:.3f}",
            f"MAE={stats['mae']:.3f}",
            f"Bias={stats['bias']:.3f}",
            f"R²={r2_text}",
        )
    )


def plot_prediction_scatter(run_dir: Path, output_dir: Path, formats: tuple[str, ...], dpi: int) -> list[Path]:
    plt = _import_plotting_modules()
    frame = _load_predictions(run_dir)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes_flat = axes.flatten()

    for axis, component in zip(axes_flat, COMPONENTS):
        true_values, pred_values, errors = _component_arrays(frame, component)
        stats = _component_stats(frame, component)
        color_values = np.abs(errors)
        scatter = axis.scatter(
            true_values,
            pred_values,
            c=color_values,
            cmap="viridis",
            s=18,
            alpha=0.78,
            edgecolors="none",
        )
        lower = min(float(np.min(true_values)), float(np.min(pred_values)))
        upper = max(float(np.max(true_values)), float(np.max(pred_values)))
        span = upper - lower
        margin = 0.03 * span if span > 0 else 1.0
        axis.plot([lower - margin, upper + margin], [lower - margin, upper + margin], linestyle="--", color="#444444", linewidth=1.1)
        axis.set_xlim(lower - margin, upper + margin)
        axis.set_ylim(lower - margin, upper + margin)
        axis.set_title(component)
        axis.set_xlabel("True (%)")
        axis.set_ylabel("Pred (%)")
        axis.grid(alpha=0.25)
        axis.text(
            0.04,
            0.96,
            _stats_text(stats),
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#d0d0d0", "alpha": 0.92},
        )
        fig.colorbar(scatter, ax=axis, fraction=0.046, pad=0.04, label="|Pred-True|")

    fig.suptitle(f"{run_dir.name} | 真值-预测散点", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    saved_paths = _save_figure(fig, output_dir / f"{run_dir.name}_prediction_scatter", formats, dpi)
    plt.close(fig)
    return saved_paths


def plot_error_distributions(run_dir: Path, output_dir: Path, formats: tuple[str, ...], dpi: int) -> list[Path]:
    plt = _import_plotting_modules()
    frame = _load_predictions(run_dir)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes_flat = axes.flatten()

    for axis, component in zip(axes_flat, COMPONENTS):
        _, _, errors = _component_arrays(frame, component)
        stats = _component_stats(frame, component)
        bins = min(40, max(10, len(errors) // 30))
        axis.hist(errors, bins=bins, color="#4c78a8", alpha=0.82, edgecolor="white")
        axis.axvline(0.0, color="#222222", linestyle="--", linewidth=1.0)
        axis.axvline(stats["bias"], color="#d62728", linestyle="-", linewidth=1.2)
        axis.set_title(component)
        axis.set_xlabel("Pred - True (%)")
        axis.set_ylabel("Count")
        axis.grid(alpha=0.22)
        axis.text(
            0.04,
            0.96,
            "\n".join(
                (
                    f"Bias={stats['bias']:.3f}",
                    f"MAE={stats['mae']:.3f}",
                    f"Max|e|={stats['max_abs_error']:.3f}",
                )
            ),
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#d0d0d0", "alpha": 0.92},
        )

    fig.suptitle(f"{run_dir.name} | 误差分布", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    saved_paths = _save_figure(fig, output_dir / f"{run_dir.name}_error_distributions", formats, dpi)
    plt.close(fig)
    return saved_paths


def generate_run_analysis_artifacts(
    run_dir: Path,
    output_dir: Path,
    formats: tuple[str, ...] = ("png", "svg"),
    dpi: int = 300,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    run, warning = _load_training_run(run_dir.parent, run_dir)
    if warning is not None or run is None:
        raise ValueError(warning)

    generated_files: list[str] = []
    generated_files.extend(str(path) for path in plot_training_run(run, output_dir, formats, dpi))
    generated_files.extend(str(path) for path in plot_prediction_scatter(run_dir, output_dir, formats, dpi))
    generated_files.extend(str(path) for path in plot_error_distributions(run_dir, output_dir, formats, dpi))

    return {
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "generated_files": generated_files,
        "formats": list(formats),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="输出单个深度学习 run 的图文化分析结果。")
    parser.add_argument("--run-dir", required=True, help="包含 train_log.csv 和 predictions.csv 的 run 目录")
    parser.add_argument("--output-dir", help="图表输出目录；默认落到 run_dir/analysis_plots")
    parser.add_argument("--formats", nargs="+", default=["png", "svg"], choices=SUPPORTED_FORMATS, help="输出格式列表")
    parser.add_argument("--dpi", type=int, default=300, help="PNG 输出分辨率")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    run_dir = _resolve_cli_path(args.run_dir)
    output_dir = _resolve_cli_path(args.output_dir) if args.output_dir else run_dir / "analysis_plots"
    result = generate_run_analysis_artifacts(
        run_dir=run_dir,
        output_dir=output_dir,
        formats=tuple(args.formats),
        dpi=int(args.dpi),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()
