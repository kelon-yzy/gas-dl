"""G3a 环境单变量敏感度曲线分析。

用已训练模型的 predictions.csv，按 T_C / P_MPa / H_RH 分组计算 per-bin RMSE，
输出数据表和曲线图。无需重新训练。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
from pathlib import Path as _Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# 中文字体支持 — 复用项目已有工具
_ml_root = (_Path(__file__).resolve().parents[2] / "src" / "ml")
if str(_ml_root) not in sys.path:
    sys.path.insert(0, str(_ml_root))
try:
    from patent_model.plotting_style import setup_chinese_fonts
    setup_chinese_fonts()
except ImportError:
    pass

# ── 字号配置 ──
AXIS_LABEL_SIZE = 11
TICK_SIZE = 9
TITLE_SIZE = 13

COMPONENT_NAMES = ["H2", "CH4", "CO2", "N2"]

ENV_SPECS = {
    "temperature": {
        "column": "T_C",
        "label_zh": "温度 T (°C)",
        "n_bins": 10,
        "unit": "°C",
        "train_range_label": "训练集温度范围",
    },
    "pressure": {
        "column": "P_MPa",
        "label_zh": "气压 P (MPa)",
        "n_bins": 10,
        "unit": "MPa",
        "train_range_label": "训练集气压范围",
    },
    "humidity": {
        "column": "H_RH",
        "label_zh": "湿度 H (%RH)",
        "n_bins": 6,
        "unit": "%RH",
        "train_range_label": "训练集湿度范围",
    },
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="环境单变量敏感度曲线分析")
    p.add_argument("--predictions", required=True, help="predictions.csv 路径 (含 fused_pred_* 列)")
    p.add_argument("--condition-grid", required=True, help="condition_grid_v1.csv 路径")
    p.add_argument("--output-dir", required=True, help="输出目录")
    p.add_argument("--model-name", default="fused", help="模型名列前缀 (default: fused)")
    return p


def _load_and_merge(pred_path: Path, cond_path: Path, model_name: str) -> pd.DataFrame:
    pred = pd.read_csv(pred_path)
    cond = pd.read_csv(cond_path)
    # 只保留需要的列
    cond_cols = ["sample_id", "T_C", "P_MPa", "H_RH"]
    cond = cond[cond_cols].drop_duplicates(subset=["sample_id"])
    df = pred.merge(cond, on="sample_id", how="inner")
    if df.empty:
        raise ValueError("predictions 和 condition_grid 按 sample_id join 后为空")
    # 计算 per-component RMSE 所需列
    for comp in COMPONENT_NAMES:
        true_col = f"true_{comp}"
        pred_col = f"{model_name}_pred_{comp}"
        if true_col not in df.columns or pred_col not in df.columns:
            raise ValueError(f"缺少列: {true_col} 或 {pred_col}")
        df[f"abs_error_{comp}"] = (df[pred_col] - df[true_col]).abs()
        df[f"sq_error_{comp}"] = (df[pred_col] - df[true_col]) ** 2
    return df


def _bin_stats(df: pd.DataFrame, env_col: str, n_bins: int) -> pd.DataFrame:
    """按环境变量等频分箱，计算每箱的 per-component RMSE 和 macro_RMSE。"""
    df = df.copy()
    df["_bin"] = pd.qcut(df[env_col], q=n_bins, duplicates="drop")
    rows = []
    for bin_val, grp in df.groupby("_bin", observed=False):
        row = {
            "bin_left": float(bin_val.left),
            "bin_right": float(bin_val.right),
            "bin_center": float(bin_val.mid),
            "n_samples": len(grp),
            "mean_env": float(grp[env_col].mean()),
        }
        rmse_list = []
        for comp in COMPONENT_NAMES:
            mse = float(grp[f"sq_error_{comp}"].mean())
            rmse = float(np.sqrt(mse))
            row[f"{comp}_RMSE"] = rmse
            rmse_list.append(rmse)
        row["macro_RMSE"] = float(np.mean(rmse_list))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("bin_center").reset_index(drop=True)


def _plot_sensitivity(
    df_stats: pd.DataFrame,
    env_spec: dict,
    train_min: float,
    train_max: float,
    output_dir: Path,
    model_name: str,
) -> None:
    env_key = env_spec["column"]
    label_zh = env_spec["label_zh"]
    unit = env_spec["unit"]
    n_bins = env_spec["n_bins"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"环境变量敏感度 — {model_name} 模型", fontsize=TITLE_SIZE, fontweight="bold")

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
    for idx, (comp, color) in enumerate(zip(COMPONENT_NAMES, colors)):
        ax = axes[idx // 2][idx % 2]
        ax.plot(df_stats["bin_center"], df_stats[f"{comp}_RMSE"], "o-", color=color, linewidth=1.5, markersize=5)
        ax.set_title(f"{comp}", fontsize=AXIS_LABEL_SIZE)
        ax.set_xlabel(label_zh, fontsize=AXIS_LABEL_SIZE)
        ax.set_ylabel(f"{comp} RMSE ({unit} 相关)", fontsize=AXIS_LABEL_SIZE)
        # 标注训练集范围
        if not np.isnan(train_min) and not np.isnan(train_max):
            ax.axvspan(train_min, train_max, alpha=0.08, color="green", label="训练集覆盖区间")
        ax.legend(fontsize=TICK_SIZE, loc="upper left")
        ax.tick_params(labelsize=TICK_SIZE)
        ax.grid(True, alpha=0.3)
        _add_dense_y_ticks(ax)

    plt.tight_layout()
    fig.savefig(output_dir / f"sensitivity_per_component_{env_key}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 汇总图：macro_RMSE + 四个组分的轻量叠加
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(df_stats["bin_center"], df_stats["macro_RMSE"], "o-", color="#E91E63", linewidth=2, markersize=6, label="macro_RMSE")
    for comp, color in zip(COMPONENT_NAMES, colors):
        ax2.plot(df_stats["bin_center"], df_stats[f"{comp}_RMSE"], "--", color=color, linewidth=1, alpha=0.6, label=comp)
    ax2.set_title(f"环境敏感度总览 — {label_zh}", fontsize=TITLE_SIZE, fontweight="bold")
    ax2.set_xlabel(label_zh, fontsize=AXIS_LABEL_SIZE)
    ax2.set_ylabel("RMSE", fontsize=AXIS_LABEL_SIZE)
    if not np.isnan(train_min) and not np.isnan(train_max):
        ax2.axvspan(train_min, train_max, alpha=0.08, color="green", label="训练集覆盖区间")
    ax2.legend(fontsize=TICK_SIZE, ncol=2, loc="upper left")
    ax2.tick_params(labelsize=TICK_SIZE)
    ax2.grid(True, alpha=0.3)
    _add_dense_y_ticks(ax2)
    plt.tight_layout()
    fig2.savefig(output_dir / f"sensitivity_summary_{env_key}.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)


def _add_dense_y_ticks(ax: plt.Axes) -> None:
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=10))
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax.grid(axis="y", which="major", alpha=0.25)
    ax.grid(axis="y", which="minor", alpha=0.12, linestyle=":")
    ax.grid(axis="x", alpha=0.18)


def _compute_train_range(cond_path: Path) -> dict[str, tuple[float, float]]:
    """从 condition_grid 读取所有检测样本的训练集环境变量范围。"""
    cond = pd.read_csv(cond_path)
    # 训练集 = 所有合成检测样本 (status=synthetic_measurement, stage_id in distance/pressure)
    train_mask = cond["status"].eq("synthetic_measurement") & cond["stage_id"].isin(["distance_stage", "pressure_stage"])
    train_cond = cond[train_mask]
    return {
        "T_C": (float(train_cond["T_C"].min()), float(train_cond["T_C"].max())),
        "P_MPa": (float(train_cond["P_MPa"].min()), float(train_cond["P_MPa"].max())),
        "H_RH": (float(train_cond["H_RH"].min()), float(train_cond["H_RH"].max())),
    }


def main() -> None:
    args = build_parser().parse_args()
    pred_path = Path(args.predictions)
    cond_path = Path(args.condition_grid)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = args.model_name

    print(f"[G3a] 加载 predictions: {pred_path}")
    print(f"[G3a] 加载 condition_grid: {cond_path}")

    df = _load_and_merge(pred_path, cond_path, model_name)
    print(f"[G3a] 合并后样本数: {len(df)}")

    train_ranges = _compute_train_range(cond_path)

    summary_rows = []
    for env_key, spec in ENV_SPECS.items():
        col = spec["column"]
        label = spec["label_zh"]
        n_bins = spec["n_bins"]
        print(f"[G3a] 分析 {label} (bins={n_bins}) ...")
        stats = _bin_stats(df, col, n_bins)
        stats.to_csv(output_dir / f"sensitivity_{col}.csv", index=False)

        train_min, train_max = train_ranges[col]
        _plot_sensitivity(stats, spec, train_min, train_max, output_dir, model_name)

        # 汇总行
        for _, row in stats.iterrows():
            summary_rows.append({
                "env_variable": col,
                "env_label": label,
                "bin_center": row["bin_center"],
                "n_samples": row["n_samples"],
                "macro_RMSE": row["macro_RMSE"],
                "H2_RMSE": row["H2_RMSE"],
                "CH4_RMSE": row["CH4_RMSE"],
                "CO2_RMSE": row["CO2_RMSE"],
                "N2_RMSE": row["N2_RMSE"],
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "sensitivity_summary.csv", index=False)

    # 输出 summary.json
    result = {
        "model_name": model_name,
        "predictions_file": str(pred_path),
        "total_samples": int(len(df)),
        "train_ranges": {k: list(v) for k, v in train_ranges.items()},
        "env_variables_analyzed": list(ENV_SPECS.keys()),
        "output_files": [p.name for p in output_dir.iterdir() if p.is_file()],
    }
    (output_dir / "sensitivity_meta.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[G3a] 完成，输出: {output_dir}")


if __name__ == "__main__":
    main()
