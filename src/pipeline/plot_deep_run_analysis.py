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
    _resolve_cli_path,
    _save_figure,
    plot_training_run,
)
from pipeline.run_plot_data import RunAnalysisBundle, load_run_analysis_bundle


def _component_axes(plt, count: int, *, figsize: tuple[float, float]) -> tuple[Any, list[Any]]:
    cols = 2 if count > 1 else 1
    rows = math.ceil(count / cols)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if hasattr(axes, "flatten"):
        flat_axes = list(axes.flatten())
    else:
        flat_axes = [axes]
    return fig, flat_axes


def _component_arrays(bundle: RunAnalysisBundle, component: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    true_values = bundle.predictions[f"y_true_{component}"].to_numpy(dtype=float)
    pred_values = bundle.predictions[f"y_pred_{component}"].to_numpy(dtype=float)
    errors = pred_values - true_values
    return true_values, pred_values, errors


def _component_stats(bundle: RunAnalysisBundle, component: str) -> dict[str, float]:
    true_values, pred_values, errors = _component_arrays(bundle, component)
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


def plot_prediction_scatter(bundle: RunAnalysisBundle, output_dir: Path, formats: tuple[str, ...], dpi: int) -> list[Path]:
    plt = _import_plotting_modules()
    fig, axes_flat = _component_axes(plt, len(bundle.components), figsize=(12, 10))

    for axis, component in zip(axes_flat, bundle.components, strict=False):
        true_values, pred_values, errors = _component_arrays(bundle, component)
        stats = _component_stats(bundle, component)
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
        axis.set_title(bundle.component_display_names.get(component, component))
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

    for axis in axes_flat[len(bundle.components) :]:
        axis.set_visible(False)

    fig.suptitle(f"{bundle.run_name} | 真值-预测散点", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    saved_paths = _save_figure(fig, output_dir / f"{bundle.run_dir.name}_prediction_scatter", formats, dpi)
    plt.close(fig)
    return saved_paths


def plot_error_distributions(bundle: RunAnalysisBundle, output_dir: Path, formats: tuple[str, ...], dpi: int) -> list[Path]:
    plt = _import_plotting_modules()
    fig, axes_flat = _component_axes(plt, len(bundle.components), figsize=(12, 9))

    for axis, component in zip(axes_flat, bundle.components, strict=False):
        _, _, errors = _component_arrays(bundle, component)
        stats = _component_stats(bundle, component)
        bins = min(40, max(10, len(errors) // 30))
        axis.hist(errors, bins=bins, color="#4c78a8", alpha=0.82, edgecolor="white")
        axis.axvline(0.0, color="#222222", linestyle="--", linewidth=1.0)
        axis.axvline(stats["bias"], color="#d62728", linestyle="-", linewidth=1.2)
        axis.set_title(bundle.component_display_names.get(component, component))
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

    for axis in axes_flat[len(bundle.components) :]:
        axis.set_visible(False)

    fig.suptitle(f"{bundle.run_name} | 误差分布", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    saved_paths = _save_figure(fig, output_dir / f"{bundle.run_dir.name}_error_distributions", formats, dpi)
    plt.close(fig)
    return saved_paths


def plot_component_metrics_summary(bundle: RunAnalysisBundle, output_dir: Path, formats: tuple[str, ...], dpi: int) -> list[Path]:
    plt = _import_plotting_modules()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    metrics = (
        ("RMSE", "#1f77b4"),
        ("MAE", "#2ca02c"),
        ("R2", "#d62728"),
    )
    labels = [bundle.component_display_names.get(component, component) for component in bundle.components]

    for axis, (metric_name, color) in zip(axes, metrics, strict=False):
        values = bundle.component_metrics[metric_name].to_numpy(dtype=float)
        bars = axis.bar(labels, values, color=color, alpha=0.88)
        axis.set_title(metric_name)
        axis.grid(axis="y", alpha=0.22)
        axis.tick_params(axis="x", rotation=0)
        for bar, value in zip(bars, values, strict=False):
            axis.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.suptitle(
        (
            f"{bundle.run_name} | 组件指标概览\n"
            f"macro_RMSE={float(bundle.summary['macro_RMSE']):.4f} | "
            f"macro_MAE={float(bundle.summary['macro_MAE']):.4f} | "
            f"mean_abs_sum_error={float(bundle.summary['mean_abs_sum_error']):.4f}"
        ),
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    saved_paths = _save_figure(fig, output_dir / f"{bundle.run_dir.name}_component_metrics_summary", formats, dpi)
    plt.close(fig)
    return saved_paths


def generate_run_analysis_artifacts(
    run_dir: Path,
    output_dir: Path,
    formats: tuple[str, ...] = ("png", "svg"),
    dpi: int = 300,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_run_analysis_bundle(run_dir, require_report_files=True)

    generated_files: list[str] = []
    generated_files.extend(str(path) for path in plot_training_run(bundle, output_dir, formats, dpi))
    generated_files.extend(str(path) for path in plot_prediction_scatter(bundle, output_dir, formats, dpi))
    generated_files.extend(str(path) for path in plot_error_distributions(bundle, output_dir, formats, dpi))
    generated_files.extend(str(path) for path in plot_component_metrics_summary(bundle, output_dir, formats, dpi))

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
