"""鲁棒性分析脚本：噪声实验和压力实验。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from patent_model.data_loader import grouped_train_test_split, load_patent_dataset
from patent_model.fault_labels import build_observed_fault_labels
from patent_model.modeling import ModelConfig, MultiComponentPatentModel
from patent_model.robustness import (
    DEFAULT_ENVIRONMENT_NOISE_LEVELS,
    METRIC_PLOT_SPECS,
    evaluate_environment_noise,
    evaluate_pressure_bins,
    plot_noise_component_metric,
    plot_noise_component_rmse,
    plot_noise_macro_metric,
    plot_noise_macro_rmse,
    plot_pressure_component_metric,
    plot_pressure_component_rmse,
    plot_pressure_macro_metric,
    plot_pressure_macro_rmse,
    select_pressure_slice,
)
from scripts._cli_utils import limit_dataset, positive_int


def build_parser() -> argparse.ArgumentParser:
    """定义鲁棒性分析脚本的命令行参数。"""

    parser = argparse.ArgumentParser(description="Run environment-noise and pressure robustness analysis.")
    parser.add_argument("--data-dir", default="../output", help="Path to the simulation data export directory.")
    parser.add_argument("--output-dir", default="outputs/robustness", help="Directory for metrics and plots.")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Group split test ratio by mixture_id.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--train-limit", type=positive_int, help="Optional cap for training samples.")
    parser.add_argument("--test-limit", type=positive_int, help="Optional cap for test samples.")
    parser.add_argument("--n-perturbations", type=positive_int, default=24, help="MC perturbation count per modality.")
    parser.add_argument("--stacking-folds", type=positive_int, default=5, help="OOF folds for Ridge meta learner.")
    parser.add_argument("--noise-level-count", type=positive_int, default=len(DEFAULT_ENVIRONMENT_NOISE_LEVELS), help="Use the first N default noise levels.")
    parser.add_argument("--noise-pressure-mpa", type=float, default=0.101325, help="Pressure used for environment-noise curves, in MPa. Default is 1 atm.")
    parser.add_argument("--noise-pressure-samples", type=positive_int, help="Number of test samples closest to --noise-pressure-mpa for noise curves.")
    parser.add_argument("--pressure-bins", type=positive_int, default=6, help="Number of pressure bins for line plots.")
    return parser


def main(argv: list[str] | None = None) -> dict[str, object]:
    """训练模型后执行噪声和压力两类鲁棒性分析。"""

    args = build_parser().parse_args(argv)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # 第 1 段：读取数据并补上分析标签。
    # 与训练脚本保持一致，先读数据并补充故障标签。
    dataset = load_patent_dataset(args.data_dir)
    observed_labels = build_observed_fault_labels(dataset)
    dataset = dataset.with_fault_labels(observed_labels)

    # 第 2 段：切出 train/test，并训练一套主模型作为后续分析基础。
    train, test = grouped_train_test_split(dataset, test_ratio=args.test_ratio, seed=args.seed)
    train = limit_dataset(train, args.train_limit)
    test = limit_dataset(test, args.test_limit)

    config = ModelConfig(
        stacking_folds=args.stacking_folds,
        n_perturbations=args.n_perturbations,
        random_state=args.seed,
    )
    model = MultiComponentPatentModel(config=config).fit(train)

    # 第 3 段：分别组织噪声实验数据和压力实验数据。
    # 一条分支评估环境噪声，另一条分支评估压力区间。
    levels = DEFAULT_ENVIRONMENT_NOISE_LEVELS[: args.noise_level_count]
    noise_test = select_pressure_slice(test, target_pressure_mpa=args.noise_pressure_mpa, max_samples=args.noise_pressure_samples)
    noise_metrics = evaluate_environment_noise(model, noise_test, levels=levels, seed=args.seed)
    pressure_stage_metrics, pressure_bin_metrics = evaluate_pressure_bins(model, test, n_bins=args.pressure_bins)

    # 第 4 段：先落指标表，再按指标种类输出对应图表。
    noise_metrics.to_csv(output / "environment_noise_metrics.csv", index=False)
    pressure_stage_metrics.to_csv(output / "pressure_stage_metrics.csv", index=False)
    pressure_bin_metrics.to_csv(output / "pressure_bin_metrics.csv", index=False)

    plot_noise_macro_rmse(noise_metrics, output / "noise_macro_rmse.png")
    plot_noise_component_rmse(noise_metrics, output / "noise_component_rmse_dynamic_ridge.png")
    plot_pressure_macro_rmse(pressure_bin_metrics, output / "pressure_macro_rmse.png")
    plot_pressure_component_rmse(pressure_bin_metrics, output / "pressure_component_rmse_dynamic_ridge.png")
    for metric_column, spec in METRIC_PLOT_SPECS.items():
        if metric_column == "RMSE_pp":
            continue
        metric_slug = spec["slug"]
        plot_noise_macro_metric(noise_metrics, output / f"noise_macro_{metric_slug}.png", metric_column)
        plot_noise_component_metric(noise_metrics, output / f"noise_component_{metric_slug}_dynamic_ridge.png", metric_column)
        plot_pressure_macro_metric(pressure_bin_metrics, output / f"pressure_macro_{metric_slug}.png", metric_column)
        plot_pressure_component_metric(pressure_bin_metrics, output / f"pressure_component_{metric_slug}_dynamic_ridge.png", metric_column)

    # 第 5 段：把这次实验的设置和输出清单写进 summary.json，方便回看。
    summary = {
        "data_dir": str(Path(args.data_dir).resolve()),
        "train_samples": train.n_samples,
        "test_samples": test.n_samples,
        "noise_test_samples": noise_test.n_samples,
        "noise_pressure_mpa": args.noise_pressure_mpa,
        "noise_pressure_mean_mpa": float(noise_test.environment[:, 1].mean()),
        "seed": args.seed,
        "n_perturbations": args.n_perturbations,
        "noise_levels": [level.__dict__ for level in levels],
        "pressure_bins": args.pressure_bins,
        "outputs": [
            "environment_noise_metrics.csv",
            "pressure_stage_metrics.csv",
            "pressure_bin_metrics.csv",
            "noise_macro_rmse.png",
            "noise_component_rmse_dynamic_ridge.png",
            "noise_macro_mre.png",
            "noise_component_mre_dynamic_ridge.png",
            "noise_macro_r2.png",
            "noise_component_r2_dynamic_ridge.png",
            "noise_macro_max_re.png",
            "noise_component_max_re_dynamic_ridge.png",
            "pressure_macro_rmse.png",
            "pressure_component_rmse_dynamic_ridge.png",
            "pressure_macro_mre.png",
            "pressure_component_mre_dynamic_ridge.png",
            "pressure_macro_r2.png",
            "pressure_component_r2_dynamic_ridge.png",
            "pressure_macro_max_re.png",
            "pressure_component_max_re_dynamic_ridge.png",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


if __name__ == "__main__":
    main()
