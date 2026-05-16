"""环境噪声与压力鲁棒性分析工具。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from patent_model.config import MODEL_DISPLAY_NAMES_ZH, MODEL_NAMES
from patent_model.dataset import PatentDataset
from patent_model.feature_profiles import has_embedded_environment
from patent_model.modeling import MultiComponentPatentModel
from patent_model.plotting_style import setup_chinese_fonts

# Matplotlib 在无界面环境下保存图片，避免脚本在命令行里卡住。
setup_chinese_fonts()


COMPONENT_DISPLAY_NAMES_ZH: dict[str, str] = {
    "H2": "氢气 H2",
    "CH4": "甲烷 CH4",
    "CO2": "二氧化碳 CO2",
}

METRIC_PLOT_SPECS: dict[str, dict[str, str]] = {
    "RMSE_pp": {"name": "均方根误差（RMSE）", "ylabel": "RMSE（百分点）", "slug": "rmse"},
    "MRE_pct": {"name": "平均相对误差（MRE）", "ylabel": "MRE（%）", "slug": "mre"},
    "R2": {"name": "决定系数（R2）", "ylabel": "R2", "slug": "r2"},
    "MaxRE_pct": {"name": "最大相对误差（Max RE）", "ylabel": "Max RE（%）", "slug": "max_re"},
}


@dataclass(frozen=True)
class EnvironmentNoiseLevel:
    """一组环境噪声标准差配置。"""

    name: str
    sigma_t: float
    sigma_p: float
    sigma_h: float


DEFAULT_ENVIRONMENT_NOISE_LEVELS: tuple[EnvironmentNoiseLevel, ...] = (
    EnvironmentNoiseLevel("level_0", 0.0, 0.000, 0.0),
    EnvironmentNoiseLevel("level_1", 0.2, 0.002, 0.5),
    EnvironmentNoiseLevel("level_2", 0.5, 0.005, 1.0),
    EnvironmentNoiseLevel("level_3", 1.0, 0.010, 2.0),
    EnvironmentNoiseLevel("level_4", 2.0, 0.020, 5.0),
)


def noise_level_label_zh(level: EnvironmentNoiseLevel) -> str:
    """生成图表横轴用的中文噪声等级标签。"""

    suffix = level.name.removeprefix("level_")
    return f"等级{suffix}\nT={level.sigma_t:.1f}°C, P={level.sigma_p:.3f}MPa, H={level.sigma_h:.1f}%RH"


def noise_sigma_label_zh(level: EnvironmentNoiseLevel) -> str:
    """生成更紧凑的噪声标准差标签。"""

    return f"σT={level.sigma_t:.1f}\nσP={level.sigma_p:.3f}\nσH={level.sigma_h:.1f}"


def _add_dense_y_ticks(ax: plt.Axes) -> None:
    """统一图表坐标轴样式。"""

    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=12))
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax.grid(axis="y", which="major", alpha=0.25)
    ax.grid(axis="y", which="minor", alpha=0.12, linestyle=":")
    ax.grid(axis="x", alpha=0.18)


def add_environment_noise(
    dataset: PatentDataset,
    sigma_t: float,
    sigma_p: float,
    sigma_h: float,
    seed: int,
) -> PatentDataset:
    """只对环境变量加高斯噪声，保留三模态特征和标签不变。"""

    rng = np.random.default_rng(seed)
    # 按 T / P / H 三列分别采样噪声，再拼回 environment 矩阵。
    noise = np.column_stack(
        [
            rng.normal(0.0, sigma_t, size=dataset.n_samples),
            rng.normal(0.0, sigma_p, size=dataset.n_samples),
            rng.normal(0.0, sigma_h, size=dataset.n_samples),
        ]
    )
    metadata = dataset.metadata.copy()
    metadata["environment_noise_case"] = "gaussian"
    metadata["sigma_T"] = sigma_t
    metadata["sigma_P"] = sigma_p
    metadata["sigma_H"] = sigma_h
    return PatentDataset(
        sample_ids=dataset.sample_ids.copy(),
        acoustic=dataset.acoustic.copy(),
        optical=dataset.optical.copy(),
        thermal=dataset.thermal.copy(),
        environment=dataset.environment + noise,
        targets=dataset.targets.copy(),
        component_names=dataset.component_names,
        metadata=metadata,
        acoustic_columns=dataset.acoustic_columns,
        optical_columns=dataset.optical_columns,
        thermal_columns=dataset.thermal_columns,
        environment_columns=dataset.environment_columns,
        provenance=dict(dataset.provenance),
        filter_report=dict(dataset.filter_report),
    )


def _derived_environment_values(environment: np.ndarray) -> dict[str, np.ndarray]:
    """Recompute deterministic environment-derived columns from T/P/RH."""

    t_c = environment[:, 0].astype(float)
    p_mpa = environment[:, 1].astype(float)
    h_rh = np.clip(environment[:, 2].astype(float), 0.0, 100.0)
    t_k = t_c + 273.15
    p_kpa = p_mpa * 1000.0
    p_ws_kpa = 0.61094 * np.exp(17.625 * t_c / (t_c + 243.04))
    p_h2o_kpa = h_rh / 100.0 * p_ws_kpa
    return {
        "T_C": t_c,
        "P_MPa": p_mpa,
        "H_RH": h_rh,
        "T_K": t_k,
        "P_kPa": p_kpa,
        "p_H2O_kPa": p_h2o_kpa,
        "x_H2O": p_h2o_kpa / np.maximum(p_kpa, 1e-9),
        "AH_g_m3": 216.7 * (p_h2o_kpa * 10.0) / t_k,
        "P_dry_kPa": p_kpa - p_h2o_kpa,
    }


def _update_columns(values: np.ndarray, columns: tuple[str, ...], replacements: dict[str, np.ndarray]) -> np.ndarray:
    out = values.copy()
    for name, replacement in replacements.items():
        if name in columns:
            out[:, columns.index(name)] = replacement
    if "c_T_norm" in columns and "c_sound" in columns:
        c_idx = columns.index("c_sound")
        norm_idx = columns.index("c_T_norm")
        out[:, norm_idx] = out[:, c_idx] / np.sqrt(replacements["T_K"] / 293.15)
    return out


def _uses_embedded_derived_environment(profile: str) -> bool:
    if has_embedded_environment(profile):
        return True
    # V1 virtual profile not registered in FEATURE_PROFILES
    return profile in {"derived_env_mc_aug"}


def add_profile_environment_noise(
    dataset: PatentDataset,
    profile: str,
    sigma_t: float,
    sigma_p: float,
    sigma_h: float,
    seed: int,
) -> PatentDataset:
    """Apply environment noise with profile-aware updates for embedded derived columns."""

    noisy = add_environment_noise(dataset, sigma_t=sigma_t, sigma_p=sigma_p, sigma_h=sigma_h, seed=seed)
    if not _uses_embedded_derived_environment(profile):
        metadata = noisy.metadata.copy()
        metadata["feature_profile"] = profile
        return PatentDataset(
            sample_ids=noisy.sample_ids,
            acoustic=noisy.acoustic,
            optical=noisy.optical,
            thermal=noisy.thermal,
            environment=noisy.environment,
            targets=noisy.targets,
            component_names=noisy.component_names,
            metadata=metadata,
            acoustic_columns=noisy.acoustic_columns,
            optical_columns=noisy.optical_columns,
            thermal_columns=noisy.thermal_columns,
            environment_columns=noisy.environment_columns,
            provenance=dict(noisy.provenance),
            filter_report=dict(noisy.filter_report),
        )

    replacements = _derived_environment_values(noisy.environment)
    metadata = noisy.metadata.copy()
    metadata["feature_profile"] = profile
    return PatentDataset(
        sample_ids=noisy.sample_ids,
        acoustic=_update_columns(noisy.acoustic, noisy.acoustic_columns, replacements),
        optical=_update_columns(noisy.optical, noisy.optical_columns, replacements),
        thermal=_update_columns(noisy.thermal, noisy.thermal_columns, replacements),
        environment=noisy.environment,
        targets=noisy.targets,
        component_names=noisy.component_names,
        metadata=metadata,
        acoustic_columns=noisy.acoustic_columns,
        optical_columns=noisy.optical_columns,
        thermal_columns=noisy.thermal_columns,
        environment_columns=noisy.environment_columns,
        provenance=dict(noisy.provenance),
        filter_report=dict(noisy.filter_report),
    )


def select_pressure_slice(
    dataset: PatentDataset,
    target_pressure_mpa: float = 0.101325,
    max_samples: int | None = None,
) -> PatentDataset:
    """选出最接近目标压力的一批样本，用于画噪声曲线。"""

    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive when provided.")
    pressure = dataset.environment[:, 1].astype(float)
    order = np.argsort(np.abs(pressure - target_pressure_mpa), kind="mergesort")
    if max_samples is not None:
        # 指定样本数时，直接取最接近目标压力的前 N 条。
        order = order[: min(max_samples, dataset.n_samples)]
    else:
        # Fixed tolerance of ~5 kPa; raise if no sample falls within this range.
        tolerance = 0.005
        order = np.flatnonzero(np.abs(pressure - target_pressure_mpa) <= tolerance)
        if order.size == 0:
            nearest_pressure = pressure[np.argmin(np.abs(pressure - target_pressure_mpa))]
            raise ValueError(
                f"No samples within {tolerance} MPa of target {target_pressure_mpa:.6f} MPa. "
                f"Nearest pressure is {nearest_pressure:.6f} MPa. "
                f"Pass max_samples to select by rank instead."
            )
    selected = dataset.subset(np.array(order, dtype=int))
    metadata = selected.metadata.copy()
    metadata["noise_pressure_target_mpa"] = target_pressure_mpa
    metadata["noise_pressure_distance_mpa"] = np.abs(selected.environment[:, 1].astype(float) - target_pressure_mpa)
    return PatentDataset(
        sample_ids=selected.sample_ids,
        acoustic=selected.acoustic,
        optical=selected.optical,
        thermal=selected.thermal,
        environment=selected.environment,
        targets=selected.targets,
        component_names=selected.component_names,
        metadata=metadata,
        acoustic_columns=selected.acoustic_columns,
        optical_columns=selected.optical_columns,
        thermal_columns=selected.thermal_columns,
        environment_columns=selected.environment_columns,
        provenance=dict(selected.provenance),
        filter_report=dict(selected.filter_report),
    )


def evaluate_environment_noise(
    model: MultiComponentPatentModel,
    dataset: PatentDataset,
    levels: tuple[EnvironmentNoiseLevel, ...] | list[EnvironmentNoiseLevel] = DEFAULT_ENVIRONMENT_NOISE_LEVELS,
    seed: int = 42,
) -> pd.DataFrame:
    """在不同环境噪声等级下重复评估模型。"""

    frames: list[pd.DataFrame] = []
    # 每个噪声等级都从同一批测试样本出发，只改变环境噪声强度。
    for idx, level in enumerate(levels):
        noisy = add_environment_noise(dataset, level.sigma_t, level.sigma_p, level.sigma_h, seed=seed + idx)
        metrics, _ = model.evaluate(noisy)
        metrics = metrics.copy()
        metrics.insert(0, "noise_level", level.name)
        metrics.insert(1, "sigma_T", level.sigma_t)
        metrics.insert(2, "sigma_P", level.sigma_p)
        metrics.insert(3, "sigma_H", level.sigma_h)
        metrics.insert(4, "noise_pressure_mpa", float(dataset.metadata["noise_pressure_target_mpa"].iloc[0]) if "noise_pressure_target_mpa" in dataset.metadata else np.nan)
        metrics.insert(5, "noise_pressure_mean_mpa", float(dataset.environment[:, 1].mean()))
        frames.append(metrics)
    return pd.concat(frames, ignore_index=True)


def evaluate_profile_environment_noise(
    model: MultiComponentPatentModel,
    dataset: PatentDataset,
    profile: str,
    levels: tuple[EnvironmentNoiseLevel, ...] | list[EnvironmentNoiseLevel] = DEFAULT_ENVIRONMENT_NOISE_LEVELS,
    seed: int = 42,
) -> pd.DataFrame:
    """Evaluate noise robustness while respecting how a feature profile uses environment fields."""

    frames: list[pd.DataFrame] = []
    for idx, level in enumerate(levels):
        noisy = add_profile_environment_noise(dataset, profile, level.sigma_t, level.sigma_p, level.sigma_h, seed=seed + idx)
        metrics, _ = model.evaluate(noisy)
        metrics = metrics.copy()
        metrics.insert(0, "profile", profile)
        metrics.insert(1, "noise_level", level.name)
        metrics.insert(2, "sigma_T", level.sigma_t)
        metrics.insert(3, "sigma_P", level.sigma_p)
        metrics.insert(4, "sigma_H", level.sigma_h)
        metrics.insert(5, "noise_pressure_mpa", float(dataset.metadata["noise_pressure_target_mpa"].iloc[0]) if "noise_pressure_target_mpa" in dataset.metadata else np.nan)
        metrics.insert(6, "noise_pressure_mean_mpa", float(dataset.environment[:, 1].mean()))
        frames.append(metrics)
    return pd.concat(frames, ignore_index=True)


def evaluate_pressure_bins(
    model: MultiComponentPatentModel,
    dataset: PatentDataset,
    n_bins: int = 6,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """分别按压力阶段标签和压力数值分箱统计误差。"""

    metrics, prediction = model.evaluate(dataset)
    stage_frames: list[pd.DataFrame] = []
    # 先按已有的 pressure_stage 标签做阶段级对比。
    for stage, indices in dataset.metadata.groupby("pressure_stage", sort=True).groups.items():
        subset = dataset.subset(np.array(list(indices), dtype=int))
        stage_metrics, _ = model.evaluate(subset)
        stage_metrics = stage_metrics.copy()
        stage_metrics.insert(0, "pressure_stage", stage)
        stage_metrics.insert(1, "n_samples", subset.n_samples)
        stage_frames.append(stage_metrics)

    metadata = dataset.metadata.copy()
    pressure_values = metadata["P_MPa"] if "P_MPa" in metadata.columns else dataset.environment[:, 1]
    metadata["pressure_bin"] = pd.cut(pressure_values, bins=n_bins)
    bin_frames: list[pd.DataFrame] = []
    # 再按压力数值自动分箱，画出连续趋势图。
    for interval, indices in metadata.groupby("pressure_bin", observed=True, sort=True).groups.items():
        subset = dataset.subset(np.array(list(indices), dtype=int))
        bin_metrics, _ = model.evaluate(subset)
        bin_metrics = bin_metrics.copy()
        bin_metrics.insert(0, "pressure_bin", str(interval))
        bin_metrics.insert(1, "pressure_mid", float(interval.mid))
        bin_metrics.insert(2, "n_samples", subset.n_samples)
        bin_frames.append(bin_metrics)

    # Keep local variables above explicit; they make debugging easier when bins are empty.
    _ = metrics, prediction
    return pd.concat(stage_frames, ignore_index=True), pd.concat(bin_frames, ignore_index=True)


# 绘图辅助函数区：统一整理指标、标题和坐标轴格式。
def _metric_spec(metric_column: str) -> dict[str, str]:
    """返回指标对应的图表元信息。"""

    if metric_column not in METRIC_PLOT_SPECS:
        raise ValueError(f"Unsupported metric column: {metric_column}")
    return METRIC_PLOT_SPECS[metric_column]


def _macro_metric(metrics: pd.DataFrame, group_columns: list[str], metric_column: str) -> pd.DataFrame:
    """把逐组分结果汇总成整体指标。"""

    aggregation = "max" if metric_column == "MaxRE_pct" else "mean"
    return metrics.groupby([*group_columns, "model"], as_index=False)[metric_column].agg(aggregation)


def _noise_pressure_title(metrics: pd.DataFrame) -> str:
    """把目标压力信息拼进图表标题。"""

    if "noise_pressure_mpa" not in metrics.columns or metrics["noise_pressure_mpa"].isna().all():
        return ""
    target = float(metrics["noise_pressure_mpa"].dropna().iloc[0])
    mean = float(metrics["noise_pressure_mean_mpa"].dropna().iloc[0])
    return f"（目标压力 {target:.6f} MPa，约 1 atm；样本均值 {mean:.6f} MPa）"


def _noise_title(base: str, metrics: pd.DataFrame) -> str:
    """统一生成噪声实验图标题。"""

    pressure_title = _noise_pressure_title(metrics)
    return f"{base}\n{pressure_title}" if pressure_title else base


def plot_noise_macro_metric(metrics: pd.DataFrame, output_path: str | Path, metric_column: str) -> None:
    """绘制不同噪声等级下的整体指标曲线。"""

    spec = _metric_spec(metric_column)
    macro = _macro_metric(metrics, ["noise_level", "sigma_T", "sigma_P", "sigma_H"], metric_column)
    order = (
        metrics[["noise_level", "sigma_T", "sigma_P", "sigma_H"]]
        .drop_duplicates()
        .sort_values(["sigma_T", "sigma_P", "sigma_H"])["noise_level"]
        .tolist()
    )
    level_lookup = {
        row.noise_level: EnvironmentNoiseLevel(str(row.noise_level), float(row.sigma_T), float(row.sigma_P), float(row.sigma_H))
        for row in metrics[["noise_level", "sigma_T", "sigma_P", "sigma_H"]].drop_duplicates().itertuples(index=False)
    }
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(9, 5))
    # 一张图里对比 acoustic/optical/thermal/fused 四类结果，便于看动态融合是否优于单模态。
    for model_name in MODEL_NAMES:
        values = macro[macro["model"] == model_name].set_index("noise_level").reindex(order)
        if values.empty:
            continue
        ax.plot(x, values[metric_column], marker="o", label=MODEL_DISPLAY_NAMES_ZH.get(model_name, model_name))
    ax.set_xticks(x)
    ax.set_xticklabels([noise_sigma_label_zh(level_lookup[name]) for name in order], fontsize=8)
    ax.set_title(_noise_title(f"环境高斯噪声由低到高的整体{spec['name']}分析", metrics))
    ax.set_xlabel("环境噪声标准差参数")
    ax.set_ylabel(f"宏平均 {spec['ylabel']}" if metric_column != "MaxRE_pct" else spec["ylabel"])
    _add_dense_y_ticks(ax)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_noise_macro_rmse(metrics: pd.DataFrame, output_path: str | Path) -> None:
    """绘制整体 RMSE 噪声曲线。"""

    plot_noise_macro_metric(metrics, output_path, "RMSE_pp")


def plot_noise_component_metric(metrics: pd.DataFrame, output_path: str | Path, metric_column: str, model_name: str = "fused") -> None:
    """绘制指定融合模型下，各组分随噪声变化的曲线。"""

    spec = _metric_spec(metric_column)
    subset = metrics[metrics["model"] == model_name]
    order = subset[["noise_level", "sigma_T", "sigma_P", "sigma_H"]].drop_duplicates().sort_values(["sigma_T", "sigma_P", "sigma_H"])[
        "noise_level"
    ].tolist()
    level_lookup = {
        row.noise_level: EnvironmentNoiseLevel(str(row.noise_level), float(row.sigma_T), float(row.sigma_P), float(row.sigma_H))
        for row in subset[["noise_level", "sigma_T", "sigma_P", "sigma_H"]].drop_duplicates().itertuples(index=False)
    }
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(9, 5))
    # 这里固定某一种融合模型，再看不同组分对噪声的敏感度差异。
    for component in sorted(subset["component"].unique()):
        values = subset[subset["component"] == component].set_index("noise_level").reindex(order)
        ax.plot(x, values[metric_column], marker="o", label=COMPONENT_DISPLAY_NAMES_ZH.get(component, component))
    ax.set_xticks(x)
    ax.set_xticklabels([noise_sigma_label_zh(level_lookup[name]) for name in order], fontsize=8)
    ax.set_title(_noise_title(f"环境高斯噪声由低到高的各组分{spec['name']}分析（{MODEL_DISPLAY_NAMES_ZH.get(model_name, model_name)}）", metrics))
    ax.set_xlabel("环境噪声标准差参数")
    ax.set_ylabel(spec["ylabel"])
    _add_dense_y_ticks(ax)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_noise_component_rmse(metrics: pd.DataFrame, output_path: str | Path, model_name: str = "fused") -> None:
    """绘制各组分 RMSE 噪声曲线。"""

    plot_noise_component_metric(metrics, output_path, "RMSE_pp", model_name)


def plot_pressure_macro_metric(metrics: pd.DataFrame, output_path: str | Path, metric_column: str) -> None:
    """绘制不同压力区间下的整体指标曲线。"""

    spec = _metric_spec(metric_column)
    macro = _macro_metric(metrics, ["pressure_bin", "pressure_mid"], metric_column).sort_values("pressure_mid")
    fig, ax = plt.subplots(figsize=(9, 5))
    # 与噪声图类似，这里一张图对比所有模型家族的压力适应性。
    for model_name in MODEL_NAMES:
        values = macro[macro["model"] == model_name]
        if values.empty:
            continue
        ax.plot(values["pressure_mid"], values[metric_column], marker="o", label=MODEL_DISPLAY_NAMES_ZH.get(model_name, model_name))
    ax.set_title(f"不同压力区间下的整体{spec['name']}分析")
    ax.set_xlabel("压力分箱中点（MPa）")
    ax.set_ylabel(f"宏平均 {spec['ylabel']}" if metric_column != "MaxRE_pct" else spec["ylabel"])
    _add_dense_y_ticks(ax)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_pressure_macro_rmse(metrics: pd.DataFrame, output_path: str | Path) -> None:
    """绘制整体 RMSE 压力曲线。"""

    plot_pressure_macro_metric(metrics, output_path, "RMSE_pp")


def plot_pressure_component_metric(
    metrics: pd.DataFrame,
    output_path: str | Path,
    metric_column: str,
    model_name: str = "fused",
) -> None:
    """绘制指定融合模型下，各组分随压力变化的曲线。"""

    spec = _metric_spec(metric_column)
    subset = metrics[metrics["model"] == model_name].sort_values("pressure_mid")
    fig, ax = plt.subplots(figsize=(9, 5))
    # 固定融合方式后，看不同目标组分在压力变化下的误差走势。
    for component in sorted(subset["component"].unique()):
        values = subset[subset["component"] == component]
        ax.plot(values["pressure_mid"], values[metric_column], marker="o", label=COMPONENT_DISPLAY_NAMES_ZH.get(component, component))
    ax.set_title(f"不同压力区间下各组分{spec['name']}分析（{MODEL_DISPLAY_NAMES_ZH.get(model_name, model_name)}）")
    ax.set_xlabel("压力分箱中点（MPa）")
    ax.set_ylabel(spec["ylabel"])
    _add_dense_y_ticks(ax)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_pressure_component_rmse(
    metrics: pd.DataFrame,
    output_path: str | Path,
    model_name: str = "fused",
) -> None:
    """绘制各组分 RMSE 压力曲线。"""

    plot_pressure_component_metric(metrics, output_path, "RMSE_pp", model_name)
