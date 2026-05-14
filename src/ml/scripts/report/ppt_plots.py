"""PPT-oriented report plots."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from patent_model.plotting_style import PROFILE_COLORS, PROFILE_LABELS, PROFILE_ORDER


def _ppt_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )


def _plot_ppt_overview(best_runs: pd.DataFrame, env_best: pd.DataFrame, component_df: pd.DataFrame, summary: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))

    best_subset = best_runs.sort_values("profile", key=lambda s: s.map({name: idx for idx, name in enumerate(PROFILE_ORDER)}))
    best_labels = [PROFILE_LABELS[p] for p in best_subset["profile"]]
    axes[0, 0].bar(best_labels, best_subset["macro_RMSE_pp"], color=[PROFILE_COLORS[p] for p in best_subset["profile"]], width=0.6)
    axes[0, 0].set_title("主实验最佳 profile")
    axes[0, 0].set_ylabel("RMSE_pp")
    axes[0, 0].tick_params(axis="x", rotation=20)
    axes[0, 0].grid(axis="y", alpha=0.22)

    env_subset = env_best.sort_values("macro_RMSE_pp").head(5)
    axes[0, 1].barh(np.arange(len(env_subset)), env_subset["macro_RMSE_pp"], color=[PROFILE_COLORS.get(p, "#666666") for p in env_subset["profile"]])
    axes[0, 1].set_yticks(np.arange(len(env_subset)))
    axes[0, 1].set_yticklabels([f"{row.run}" for row in env_subset.itertuples()])
    axes[0, 1].set_title("环境补偿最优组合")
    axes[0, 1].set_xlabel("RMSE_pp")
    axes[0, 1].grid(axis="x", alpha=0.22)

    comp_order = ["H2", "CH4", "CO2", "N2"]
    comp_values = component_df.set_index("component").reindex(comp_order)["RMSE_pp"]
    axes[1, 0].bar(comp_order, comp_values, color=["#3b8f5a", "#2f6fbb", "#c65f2f", "#7a4fb3"])
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("最佳模型组分误差")
    axes[1, 0].set_ylabel("RMSE_pp (log)")
    axes[1, 0].grid(axis="y", alpha=0.22, which="both")

    robustness_labels = [PROFILE_LABELS[p] for p in summary["profile"]]
    axes[1, 1].bar(robustness_labels, summary["detection_macro_RMSE_pp"], color=[PROFILE_COLORS[p] for p in summary["profile"]], width=0.6)
    axes[1, 1].set_title("鲁棒性检测总览")
    axes[1, 1].set_ylabel("RMSE_pp")
    axes[1, 1].tick_params(axis="x", rotation=20)
    axes[1, 1].grid(axis="y", alpha=0.22)

    fig.suptitle("四组分传统模型实验总览", fontsize=17, y=0.965)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig

