"""Run profile-aware robustness tests for V1 environment compensation results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from patent_model.data_loader import grouped_train_test_split, load_patent_dataset
from patent_model.dataset import PatentDataset
from patent_model.environment_augmentation import augment_derived_env_training_data
from patent_model.feature_profiles import FEATURE_PROFILES
from patent_model.fault_labels import build_observed_fault_labels
from patent_model.logging_utils import get_logger
from patent_model.modeling import ModelConfig, MultiComponentPatentModel
from patent_model.plotting_style import (
    PROFILE_COLORS,
    PROFILE_LABELS,
    setup_chinese_fonts,
)
from patent_model.robustness import (
    DEFAULT_ENVIRONMENT_NOISE_LEVELS,
    evaluate_pressure_bins,
    evaluate_profile_environment_noise,
    select_pressure_slice,
)
from scripts._cli_utils import limit_dataset, positive_int
from scripts.environment_compensation_common import (
    PROFILES,
    add_model_args,
    build_meta_key,
    build_model_config,
    profile_data_dir,
    require_known_profile_mode,
    resolve_feature_profile_name,
)


logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run profile-aware robustness tests for environment compensation.")
    parser.add_argument("--raw-data-dir", default="../output")
    parser.add_argument("--env-data-dir", default="../../simulation-data/output_environment")
    parser.add_argument("--output-dir", default="outputs/environment_compensation_robustness")
    parser.add_argument("--compensation-results", default="outputs/environment_compensation_quick/environment_compensation_summary.csv")
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-limit", type=positive_int)
    parser.add_argument("--test-limit", type=positive_int)
    parser.add_argument("--n-perturbations", type=positive_int, default=24)
    parser.add_argument("--stacking-folds", type=positive_int, default=5)
    parser.add_argument("--noise-level-count", type=positive_int, default=len(DEFAULT_ENVIRONMENT_NOISE_LEVELS))
    parser.add_argument("--noise-pressure-mpa", type=float, default=0.101325)
    parser.add_argument("--noise-pressure-samples", type=positive_int)
    parser.add_argument("--pressure-bins", type=positive_int, default=6)
    parser.add_argument("--mc-env-samples", type=int, default=4)
    parser.add_argument("--mc-env-sigma-t", type=float, default=0.5)
    parser.add_argument("--mc-env-sigma-p", type=float, default=0.005)
    parser.add_argument("--mc-env-sigma-h", type=float, default=1.0)
    add_model_args(parser, positive_int)
    return parser


def _load_split_model(args: argparse.Namespace, profile: str) -> tuple[MultiComponentPatentModel, PatentDataset, PatentDataset, int]:
    feature_profile_name = resolve_feature_profile_name(profile, args.component_mode)
    dataset = load_patent_dataset(profile_data_dir(profile, Path(args.raw_data_dir), Path(args.env_data_dir)), profile=feature_profile_name)
    observed_labels = build_observed_fault_labels(dataset)
    dataset = dataset.with_fault_labels(observed_labels)

    train, test = grouped_train_test_split(dataset, test_ratio=args.test_ratio, seed=args.seed)
    train = limit_dataset(train, args.train_limit)
    test = limit_dataset(test, args.test_limit)
    train_original_samples = train.n_samples
    if profile == "derived_env_mc_aug":
        train = augment_derived_env_training_data(
            train,
            mc_samples=args.mc_env_samples,
            sigma_t=args.mc_env_sigma_t,
            sigma_p=args.mc_env_sigma_p,
            sigma_h=args.mc_env_sigma_h,
            seed=args.seed,
            profile=feature_profile_name,
        )
    config = build_model_config(args, feature_profile_name)
    model = MultiComponentPatentModel(config=config, component_names=train.component_names).fit(train)
    return model, train, test, train_original_samples


def _detection_macro_by_profile(path: Path, meta_key: str) -> dict[str, float]:
    if not path.exists():
        return {}
    metrics = pd.read_csv(path)
    metrics = metrics[metrics["model"] == meta_key]
    return metrics.groupby("profile")["RMSE_pp"].mean().to_dict()


def _build_summary(
    noise_metrics: pd.DataFrame,
    pressure_metrics: pd.DataFrame,
    detection_macro: dict[str, float],
    meta_key: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for profile in PROFILES:
        profile_noise = noise_metrics[(noise_metrics["profile"] == profile) & (noise_metrics["model"] == meta_key)]
        noise_macro = profile_noise.groupby("noise_level")["RMSE_pp"].mean()
        profile_pressure = pressure_metrics[(pressure_metrics["profile"] == profile) & (pressure_metrics["model"] == meta_key)]
        pressure_macro = profile_pressure.groupby("pressure_bin")["RMSE_pp"].mean()
        level_0 = float(noise_macro.get("level_0", np.nan))
        worst_noise = float(noise_macro.max()) if not noise_macro.empty else np.nan
        worst_pressure = float(pressure_macro.max()) if not pressure_macro.empty else np.nan
        rows.append(
            {
                "profile": profile,
                "detection_macro_RMSE_pp": detection_macro.get(profile, np.nan),
                "noise_macro_RMSE_pp_level_0": level_0,
                "noise_macro_RMSE_pp_worst": worst_noise,
                "noise_macro_RMSE_pp_increase": worst_noise - level_0,
                "pressure_macro_RMSE_pp_worst": worst_pressure,
            }
        )
    return pd.DataFrame(rows)


def _analysis(summary: pd.DataFrame, meta_key: str) -> dict[str, object]:
    valid_detection = summary.dropna(subset=["detection_macro_RMSE_pp"])
    valid_noise = summary.dropna(subset=["noise_macro_RMSE_pp_worst"])
    valid_pressure = summary.dropna(subset=["pressure_macro_RMSE_pp_worst"])
    return {
        "main_metric": f"{meta_key} macro RMSE_pp",
        "best_detection_profile": None if valid_detection.empty else str(valid_detection.sort_values("detection_macro_RMSE_pp").iloc[0]["profile"]),
        "best_noise_robust_profile": None if valid_noise.empty else str(valid_noise.sort_values("noise_macro_RMSE_pp_worst").iloc[0]["profile"]),
        "best_pressure_robust_profile": None if valid_pressure.empty else str(valid_pressure.sort_values("pressure_macro_RMSE_pp_worst").iloc[0]["profile"]),
        "risk_notes": [
            "MRE_pct and MaxRE_pct are retained but not used as main metrics because labels near zero can inflate relative error.",
            "Current data are structure-level simulation outputs; robustness numbers are engineering comparison results, not real sensor performance claims.",
        ],
    }


def _save_figure(fig: plt.Figure, output_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _plot_summary_bar(summary: pd.DataFrame, output_path: Path) -> None:
    metrics = [
        "detection_macro_RMSE_pp",
        "noise_macro_RMSE_pp_worst",
        "pressure_macro_RMSE_pp_worst",
    ]
    labels = ["检测", "噪声最差", "压力最差"]
    x = np.arange(len(summary))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9, 5))
    for offset, metric, label in zip((-width, 0.0, width), metrics, labels):
        ax.bar(x + offset, summary[metric], width=width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([PROFILE_LABELS.get(profile, profile) for profile in summary["profile"]])
    ax.set_ylabel("macro RMSE_pp")
    ax.set_title("环境补偿 profile 鲁棒性总览")
    ax.grid(axis="y", alpha=0.22)
    ax.legend()
    _save_figure(fig, output_path)


def _plot_noise_macro(noise_metrics: pd.DataFrame, output_path: Path, meta_key: str) -> None:
    subset = noise_metrics[noise_metrics["model"] == meta_key]
    macro = subset.groupby(["profile", "noise_level"], as_index=False)["RMSE_pp"].mean()
    level_order = sorted(macro["noise_level"].unique(), key=lambda item: int(str(item).split("_")[-1]))
    fig, ax = plt.subplots(figsize=(9, 5))
    for profile in PROFILES:
        values = macro[macro["profile"] == profile].set_index("noise_level").reindex(level_order)
        if values.empty:
            continue
        ax.plot(level_order, values["RMSE_pp"], marker="o", linewidth=2.0, color=PROFILE_COLORS[profile], label=PROFILE_LABELS[profile])
    ax.set_xlabel("环境噪声等级")
    ax.set_ylabel(f"{meta_key} macro RMSE_pp")
    ax.set_title("不同环境噪声等级下的鲁棒性")
    ax.grid(alpha=0.22)
    ax.legend()
    _save_figure(fig, output_path)


def _plot_pressure_macro(pressure_metrics: pd.DataFrame, output_path: Path, meta_key: str) -> None:
    subset = pressure_metrics[pressure_metrics["model"] == meta_key].copy()
    macro = subset.groupby(["profile", "pressure_bin", "pressure_mid"], as_index=False, observed=True)["RMSE_pp"].mean()
    fig, ax = plt.subplots(figsize=(9, 5))
    for profile in PROFILES:
        values = macro[macro["profile"] == profile].sort_values("pressure_mid")
        if values.empty:
            continue
        ax.plot(values["pressure_mid"], values["RMSE_pp"], marker="o", linewidth=2.0, color=PROFILE_COLORS[profile], label=PROFILE_LABELS[profile])
    ax.set_xlabel("压力分箱中点 MPa")
    ax.set_ylabel(f"{meta_key} macro RMSE_pp")
    ax.set_title("不同压力区间下的鲁棒性")
    ax.grid(alpha=0.22)
    ax.legend()
    _save_figure(fig, output_path)


def _plot_component_noise(noise_metrics: pd.DataFrame, output_path: Path, meta_key: str) -> None:
    subset = noise_metrics[noise_metrics["model"] == meta_key]
    components = sorted(subset["component"].unique())
    level_order = sorted(subset["noise_level"].unique(), key=lambda item: int(str(item).split("_")[-1]))
    fig, axes = plt.subplots(1, len(components), figsize=(4.3 * len(components), 4), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, component in zip(axes, components):
        comp = subset[subset["component"] == component]
        for profile in PROFILES:
            values = comp[comp["profile"] == profile].set_index("noise_level").reindex(level_order)
            if values.empty:
                continue
            ax.plot(level_order, values["RMSE_pp"], marker="o", linewidth=1.8, color=PROFILE_COLORS[profile], label=PROFILE_LABELS[profile])
        ax.set_title(component)
        ax.set_xlabel("噪声等级")
        ax.set_ylabel("RMSE_pp")
        ax.grid(alpha=0.22)
    axes[0].legend(fontsize=8)
    fig.suptitle("不同组分在环境噪声下的 RMSE")
    _save_figure(fig, output_path)


def _write_plots(
    output: Path,
    summary: pd.DataFrame,
    noise_metrics: pd.DataFrame,
    pressure_metrics: pd.DataFrame,
    meta_key: str,
) -> list[str]:
    plot_paths = [
        output / "robustness_summary_bar.png",
        output / "noise_macro_rmse_by_level.png",
        output / "pressure_macro_rmse_by_bin.png",
        output / f"noise_component_rmse_{meta_key}.png",
    ]
    _plot_summary_bar(summary, plot_paths[0])
    _plot_noise_macro(noise_metrics, plot_paths[1], meta_key)
    _plot_pressure_macro(pressure_metrics, plot_paths[2], meta_key)
    _plot_component_noise(noise_metrics, plot_paths[3], meta_key)
    return [path.name for path in plot_paths]


def main(argv: list[str] | None = None) -> dict[str, object]:
    setup_chinese_fonts()
    args = build_parser().parse_args(argv)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    meta_key = build_meta_key(args.meta_model_type)

    levels = DEFAULT_ENVIRONMENT_NOISE_LEVELS[: args.noise_level_count]
    noise_frames = []
    pressure_frames = []
    run_rows = []
    for profile in PROFILES:
        require_known_profile_mode(profile, args.component_mode)
        logger.info("robustness profile=%s", profile)
        model, train, test, train_original_samples = _load_split_model(args, profile)
        noise_test = select_pressure_slice(test, target_pressure_mpa=args.noise_pressure_mpa, max_samples=args.noise_pressure_samples)
        noise_metrics = evaluate_profile_environment_noise(model, noise_test, profile=resolve_feature_profile_name(profile, args.component_mode), levels=levels, seed=args.seed)
        noise_metrics["profile"] = profile
        _, pressure_bin_metrics = evaluate_pressure_bins(model, test, n_bins=args.pressure_bins)
        pressure_bin_metrics = pressure_bin_metrics.copy()
        pressure_bin_metrics.insert(0, "profile", profile)
        noise_frames.append(noise_metrics)
        pressure_frames.append(pressure_bin_metrics)
        run_rows.append(
            {
                "profile": profile,
                "train_original_samples": train_original_samples,
                "train_samples": train.n_samples,
                "test_samples": test.n_samples,
                "noise_test_samples": noise_test.n_samples,
            }
        )

    all_noise = pd.concat(noise_frames, ignore_index=True)
    all_pressure = pd.concat(pressure_frames, ignore_index=True)
    detection_macro = _detection_macro_by_profile(Path(args.compensation_results), meta_key)
    summary = _build_summary(all_noise, all_pressure, detection_macro, meta_key)
    analysis = _analysis(summary, meta_key)
    analysis["component_mode"] = args.component_mode
    analysis["branch_model_type"] = args.branch_model_type
    analysis["meta_model_type"] = args.meta_model_type
    analysis["meta_key"] = meta_key
    plot_outputs = _write_plots(output, summary, all_noise, all_pressure, meta_key)
    analysis["plot_outputs"] = plot_outputs

    all_noise.to_csv(output / "profile_environment_noise_metrics.csv", index=False)
    all_pressure.to_csv(output / "profile_pressure_bin_metrics.csv", index=False)
    pd.DataFrame(run_rows).to_csv(output / "run_samples.csv", index=False)
    summary.to_csv(output / "robustness_summary.csv", index=False)
    (output / "analysis.json").write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
    return analysis


if __name__ == "__main__":
    main()
