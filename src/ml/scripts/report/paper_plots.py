"""Paper-oriented report plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from patent_model.plotting_style import PROFILE_COLORS, PROFILE_LABELS, PROFILE_ORDER
from scripts.report.constants import COMBO_LABELS, COMBO_ORDER
from scripts.report.decision import _candidate_label, _highlight_mask


def _save_figure(fig: plt.Figure, output_path: Path, dpi: int, close: bool = True) -> None:
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    if close:
        plt.close(fig)


def _save_dual_format(fig: plt.Figure, base_path: Path, dpi: int) -> list[str]:
    _save_figure(fig, base_path.with_suffix(".png"), dpi, close=False)
    _save_figure(fig, base_path.with_suffix(".svg"), dpi, close=True)
    return [base_path.with_suffix(".png").name, base_path.with_suffix(".svg").name]


def _paper_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def _plot_best_profile_bars(best_runs: pd.DataFrame) -> plt.Figure:
    subset = best_runs.sort_values("profile", key=lambda s: s.map({name: idx for idx, name in enumerate(PROFILE_ORDER)}))
    x = np.arange(len(subset))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))

    axes[0].bar(x, subset["macro_RMSE_pp"], color=[PROFILE_COLORS[p] for p in subset["profile"]], width=0.6)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([PROFILE_LABELS[p] for p in subset["profile"]], rotation=0)
    axes[0].set_ylabel("macro RMSE_pp")
    axes[0].set_title("主实验最佳 profile 对比")
    axes[0].grid(axis="y", alpha=0.22)
    for idx, value in enumerate(subset["macro_RMSE_pp"]):
        axes[0].text(idx, value + 0.04, f"{value:.3f}", ha="center", va="bottom", fontsize=9)

    axes[1].bar(x, subset["macro_R2"], color=[PROFILE_COLORS[p] for p in subset["profile"]], width=0.6)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([PROFILE_LABELS[p] for p in subset["profile"]], rotation=0)
    axes[1].set_ylabel("macro R2")
    axes[1].set_title("主实验最佳 profile 对比")
    axes[1].grid(axis="y", alpha=0.22)
    for idx, value in enumerate(subset["macro_R2"]):
        axes[1].text(idx, value + 0.01, f"{value:.3f}", ha="center", va="bottom", fontsize=9)

    return fig


def _plot_main_heatmap(main_runs: pd.DataFrame) -> plt.Figure:
    pivot = main_runs.pivot(index="profile", columns="combo", values="macro_RMSE_pp").reindex(index=PROFILE_ORDER, columns=COMBO_ORDER)
    fig, ax = plt.subplots(figsize=(12.4, 4.8))
    values = pivot.to_numpy()
    im = ax.imshow(values, aspect="auto", cmap="Reds")
    ax.set_xticks(np.arange(len(COMBO_ORDER)))
    ax.set_xticklabels([COMBO_LABELS[c] for c in COMBO_ORDER], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(PROFILE_ORDER)))
    ax.set_yticklabels([PROFILE_LABELS[p] for p in PROFILE_ORDER])
    ax.set_title("四组分主实验：全部模型组合宏 RMSE 热图")
    ax.set_xlabel("基学习器 / 融合器组合")
    ax.set_ylabel("数据 profile")
    threshold = float(np.nanmin(values) + (np.nanmax(values) - np.nanmin(values)) * 0.55)
    for i, profile in enumerate(PROFILE_ORDER):
        for j, combo in enumerate(COMBO_ORDER):
            value = pivot.loc[profile, combo]
            text_color = "white" if value >= threshold else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8, color=text_color)
    fig.colorbar(im, ax=ax, label="macro RMSE_pp")
    return fig


def _plot_component_breakdown(component_df: pd.DataFrame) -> plt.Figure:
    order = ["H2", "CH4", "CO2", "N2"]
    values = component_df.set_index("component").reindex(order)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    colors = ["#3b8f5a", "#2f6fbb", "#c65f2f", "#7a4fb3"]
    bars = ax.bar(order, values["RMSE_pp"], color=colors, width=0.6)
    ax.set_yscale("log")
    ax.set_ylabel("RMSE_pp (log scale)")
    ax.set_title("最佳模型的组分误差分解")
    ax.grid(axis="y", alpha=0.22, which="both")
    for bar, value in zip(bars, values["RMSE_pp"]):
        ax.text(bar.get_x() + bar.get_width() / 2.0, value * 1.12, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    return fig


def _plot_env_compensation(env_best: pd.DataFrame) -> plt.Figure:
    subset = env_best.sort_values("macro_RMSE_pp")
    fig, ax = plt.subplots(figsize=(10.2, 4.9))
    y = np.arange(len(subset))
    bars = ax.barh(y, subset["macro_RMSE_pp"], color=[PROFILE_COLORS.get(p, "#666666") for p in subset["profile"]], height=0.65)
    ax.set_yticks(y)
    labels = [f"{run}  |  {profile}" for run, profile in zip(subset["run"], subset["profile"])]
    ax.set_yticklabels(labels)
    ax.set_xlabel("macro RMSE_pp")
    ax.set_title("环境补偿网格：各模型家族的最优 profile")
    ax.grid(axis="x", alpha=0.22)
    for bar, value in zip(bars, subset["macro_RMSE_pp"]):
        ax.text(value + 0.02, bar.get_y() + bar.get_height() / 2.0, f"{value:.3f}", va="center", fontsize=9)
    return fig


def _plot_robustness_summary(summary: pd.DataFrame) -> plt.Figure:
    subset = summary.sort_values("profile", key=lambda s: s.map({name: idx for idx, name in enumerate(PROFILE_ORDER)}))
    metrics = [
        ("detection_macro_RMSE_pp", "检测"),
        ("noise_macro_RMSE_pp_worst", "噪声最差"),
        ("pressure_macro_RMSE_pp_worst", "压力最差"),
    ]
    x = np.arange(len(subset))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10.6, 4.8))
    offsets = (-width, 0.0, width)
    for offset, (metric, label) in zip(offsets, metrics):
        ax.bar(x + offset, subset[metric], width=width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([PROFILE_LABELS[p] for p in subset["profile"]])
    ax.set_ylabel("macro RMSE_pp")
    ax.set_title("鲁棒性总览：检测 / 噪声 / 压力")
    ax.grid(axis="y", alpha=0.22)
    ax.legend()
    return fig


def _plot_robustness_curves(noise: pd.DataFrame, pressure: pd.DataFrame, meta_key: str) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))

    noise_subset = noise[noise["model"] == meta_key]
    level_order = sorted(noise_subset["noise_level"].unique(), key=lambda item: int(str(item).split("_")[-1]))
    for profile in PROFILE_ORDER:
        values = noise_subset[noise_subset["profile"] == profile].groupby("noise_level")["RMSE_pp"].mean().reindex(level_order)
        axes[0].plot(level_order, values, marker="o", linewidth=2.0, color=PROFILE_COLORS[profile], label=PROFILE_LABELS[profile])
    axes[0].set_xlabel("环境噪声等级")
    axes[0].set_ylabel(f"{meta_key} macro RMSE_pp")
    axes[0].set_title("环境噪声曲线")
    axes[0].grid(alpha=0.22)

    pressure_subset = pressure[pressure["model"] == meta_key]
    for profile in PROFILE_ORDER:
        values = (
            pressure_subset[pressure_subset["profile"] == profile]
            .groupby("pressure_mid", as_index=False)["RMSE_pp"]
            .mean()
            .sort_values("pressure_mid")
        )
        axes[1].plot(values["pressure_mid"], values["RMSE_pp"], marker="o", linewidth=2.0, color=PROFILE_COLORS[profile], label=PROFILE_LABELS[profile])
    axes[1].set_xlabel("压力分箱中点 MPa")
    axes[1].set_ylabel(f"{meta_key} macro RMSE_pp")
    axes[1].set_title("压力分箱曲线")
    axes[1].grid(alpha=0.22)
    axes[1].legend(loc="best")

    return fig


def _plot_decision_dashboard(top_candidates: pd.DataFrame) -> plt.Figure:
    subset = top_candidates.sort_values("macro_RMSE_pp").head(10)
    labels = [_candidate_label(row.profile, row.combo, row.model_name) for row in subset.itertuples()]
    colors = ["#d84b4b" if row.recommendation == "recommended" else "#9aa5b1" if row.recommendation == "baseline" else "#4c78a8" for row in subset.itertuples()]
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0), gridspec_kw={"width_ratios": [1.45, 1.0]})

    y = np.arange(len(subset))
    axes[0].barh(y, subset["macro_RMSE_pp"], color=colors, height=0.68)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("macro RMSE_pp")
    axes[0].set_title("候选组合主指标排名")
    axes[0].grid(axis="x", alpha=0.22)
    for idx, row in enumerate(subset.itertuples()):
        axes[0].text(row.macro_RMSE_pp + 0.03, idx, f"R2={row.macro_R2:.3f}", va="center", fontsize=9)

    scatter = axes[1].scatter(
        subset["macro_R2"],
        subset["macro_MRE_pct"],
        s=np.clip(8000.0 / np.maximum(subset["macro_RMSE_pp"], 0.1), 70, 250),
        c=colors,
        alpha=0.85,
        edgecolors="black",
        linewidths=0.4,
    )
    del scatter
    axes[1].set_xlabel("macro R2")
    axes[1].set_ylabel("macro MRE_pct")
    axes[1].set_title("精度与相对误差平衡")
    axes[1].grid(alpha=0.22)
    for row in subset.itertuples():
        if row.recommendation == "recommended":
            axes[1].annotate(f"{COMBO_LABELS.get(row.combo, row.combo)} / {row.model_name}", (row.macro_R2, row.macro_MRE_pct), xytext=(8, -10), textcoords="offset points", fontsize=9)
    fig.suptitle("最终推荐组合决策总览", y=0.98)
    return fig


def _plot_metric_small_multiples(top_candidates: pd.DataFrame) -> plt.Figure:
    subset = top_candidates.sort_values("macro_RMSE_pp").head(6).copy()
    subset["label"] = [_candidate_label(row.profile, row.combo, row.model_name) for row in subset.itertuples()]
    metrics = [
        ("macro_RMSE_pp", "macro RMSE_pp", True),
        ("macro_R2", "macro R2", False),
        ("macro_MRE_pct", "macro MRE_pct", True),
        ("macro_MaxRE_pct", "macro MaxRE_pct", True),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.6))
    for ax, (column, title, ascending) in zip(axes.flat, metrics):
        ranked = subset.sort_values(column, ascending=ascending)
        colors = ["#d84b4b" if row.recommendation == "recommended" else "#4c78a8" for row in ranked.itertuples()]
        ax.barh(ranked["label"], ranked[column], color=colors, height=0.66)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.22)
    fig.suptitle("核心指标小多图对比", y=0.985)
    return fig


def _plot_pareto_frontier(decision_summary: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    for profile in PROFILE_ORDER:
        subset = decision_summary[decision_summary["profile"] == profile]
        if subset.empty:
            continue
        ax.scatter(subset["macro_RMSE_pp"], subset["macro_R2"], label=PROFILE_LABELS[profile], color=PROFILE_COLORS[profile], alpha=0.78, s=65)
    preferred = decision_summary[_highlight_mask(decision_summary)]
    if not preferred.empty:
        row = preferred.iloc[0]
        ax.scatter([row["macro_RMSE_pp"]], [row["macro_R2"]], color="#d84b4b", s=180, marker="*", edgecolors="black", linewidths=0.6, zorder=5)
        ax.annotate("推荐组合", (row["macro_RMSE_pp"], row["macro_R2"]), xytext=(8, 8), textcoords="offset points", fontsize=9)
    ax.set_xlabel("macro RMSE_pp")
    ax.set_ylabel("macro R2")
    ax.set_title("全部组合 Pareto 分布")
    ax.grid(alpha=0.22)
    ax.legend(loc="best")
    return fig


def _plot_candidate_component_compare(component_candidates: pd.DataFrame) -> plt.Figure:
    component_order = ["H2", "CH4", "CO2", "N2"]
    candidates = list(dict.fromkeys(component_candidates["candidate_label"].tolist()))
    width = 0.8 / max(len(candidates), 1)
    x = np.arange(len(component_order))
    fig, ax = plt.subplots(figsize=(10.8, 5.0))
    palette = ["#d84b4b", "#4c78a8", "#72b7b2", "#9d755d"]
    for idx, candidate in enumerate(candidates):
        values = component_candidates[component_candidates["candidate_label"] == candidate].set_index("component").reindex(component_order)["RMSE_pp"]
        offset = (idx - (len(candidates) - 1) / 2.0) * width
        ax.bar(x + offset, values, width=width, label=candidate, color=palette[idx % len(palette)])
    ax.set_xticks(x)
    ax.set_xticklabels(component_order)
    ax.set_ylabel("RMSE_pp")
    ax.set_title("候选组合组分误差对比")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(loc="best")
    return fig


def _plot_robustness_scoreboard(robustness_summary: pd.DataFrame) -> plt.Figure:
    subset = robustness_summary.copy()
    subset["label"] = [_candidate_label(row.profile, row.combo, row.model_name) for row in subset.itertuples()]
    subset = subset.sort_values(["detection_macro_RMSE_pp", "pressure_macro_RMSE_pp_worst"], ascending=[True, True]).head(8)
    y = np.arange(len(subset))
    fig, ax = plt.subplots(figsize=(12.2, 5.2))
    width = 0.24
    ax.barh(y - width, subset["detection_macro_RMSE_pp"], height=width, label="检测", color="#4c78a8")
    ax.barh(y, subset["noise_macro_RMSE_pp_worst"], height=width, label="噪声最差", color="#f58518")
    ax.barh(y + width, subset["pressure_macro_RMSE_pp_worst"], height=width, label="压力最差", color="#54a24b")
    ax.set_yticks(y)
    ax.set_yticklabels(subset["label"])
    ax.invert_yaxis()
    ax.set_xlabel("macro RMSE_pp")
    ax.set_title("鲁棒性记分板")
    ax.grid(axis="x", alpha=0.22)
    ax.legend(loc="best")
    return fig
