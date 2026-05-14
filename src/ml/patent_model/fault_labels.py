"""故障标签推导与故障注入工具。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from patent_model.dataset import PatentDataset


# 人工注入故障时的强度档位。数值越大，扰动幅度越大。
_SEVERITY_SCALE = {
    "mild": 0.15,
    "medium": 0.35,
    "severe": 0.70,
}


def _required_acoustic_index(dataset: PatentDataset, column: str) -> int:
    """从 dataset.acoustic_columns 取列索引，缺失时抛 ValueError。"""

    if not dataset.acoustic_columns:
        raise ValueError("dataset.acoustic_columns is empty; fault label derivation cannot proceed.")
    try:
        return dataset.acoustic_columns.index(column)
    except ValueError as exc:
        raise ValueError(f"Required acoustic column '{column}' not in dataset.acoustic_columns") from exc


def _abs_quantile_threshold(values: pd.Series, quantile: float) -> float:
    """计算绝对值分位数阈值。"""

    return float(values.abs().quantile(quantile))


def _acoustic_delay_residuals(dataset: PatentDataset) -> np.ndarray:
    """按 mixture_id 拟合 TOF 与声程关系，提取声学残差。"""

    tof_idx = _required_acoustic_index(dataset, "TOF")
    length_idx = _required_acoustic_index(dataset, "L_m")
    metadata = dataset.metadata
    tof = dataset.acoustic[:, tof_idx]
    length = dataset.acoustic[:, length_idx]
    residuals = np.full(dataset.n_samples, np.nan, dtype=float)
    # 在每个 mixture 内部做局部拟合，避免不同工况直接混在一起比较。
    for _, group in metadata.groupby("mixture_id", sort=False):
        idx = group.index.to_numpy()
        if len(idx) < 3:
            continue
        x = length[idx]
        y = tof[idx]
        if np.unique(x).size < 2:
            continue
        slope, intercept = np.polyfit(x, y, deg=1)
        residuals[idx] = y - (slope * x + intercept)
    return residuals


def build_observed_fault_labels(dataset: PatentDataset, quantile: float = 0.95) -> pd.DataFrame:
    """根据现有统计特征推导自然故障标签。"""

    if not 0.5 < quantile < 1.0:
        raise ValueError("quantile must be between 0.5 and 1.0.")
    metadata = dataset.metadata

    # 第 1 段：分别估计三条模态自己的异常条件。
    # 光学异常看饱和标记和基线漂移。
    optical_drift = metadata[["optical_baseline_drift_ch4", "optical_baseline_drift_co2"]].abs().max(axis=1)
    optical_threshold = float(optical_drift.quantile(quantile))
    optical_saturation = (metadata["ndir_ch4_saturated"].fillna(0).astype(int) == 1) | (
        metadata["ndir_co2_saturated"].fillna(0).astype(int) == 1
    )
    optical_fault = optical_saturation | (optical_drift > optical_threshold)

    # 热导异常主要看基线漂移是否超过高分位阈值。
    thermal_threshold = _abs_quantile_threshold(metadata["thermal_baseline_drift"], quantile)
    thermal_fault = metadata["thermal_baseline_drift"].abs() > thermal_threshold

    # 声学异常同时参考 TOF 残差和衰减特征的异常区间。
    residuals = np.abs(_acoustic_delay_residuals(dataset))
    finite_residuals = residuals[np.isfinite(residuals)]
    residual_threshold = float(np.quantile(finite_residuals, quantile)) if finite_residuals.size else np.inf
    attenuation = metadata["attenuation_alpha"]
    low = float(attenuation.quantile(1.0 - quantile))
    high = float(attenuation.quantile(quantile))
    acoustic_fault = (residuals > residual_threshold) | (attenuation < low).to_numpy() | (attenuation > high).to_numpy()

    # 第 2 段：把三路异常合并成统一的 fault_case / fault_severity 语义标签。
    # 将三路异常汇总成 clean / single fault / mixed fault 三类标签。
    fault_count = optical_fault.astype(int) + thermal_fault.astype(int) + acoustic_fault.astype(int)
    mixed_fault = fault_count >= 2
    is_clean = fault_count == 0
    fault_case = np.select(
        [
            mixed_fault,
            optical_fault.to_numpy(),
            thermal_fault.to_numpy(),
            acoustic_fault,
        ],
        [
            "mixed_fault",
            "optical_fault",
            "thermal_fault",
            "acoustic_fault",
        ],
        default="clean",
    )
    fault_severity = np.select(
        [mixed_fault, fault_count == 1],
        ["observed_mixed", "observed_single"],
        default="observed_clean",
    )
    return pd.DataFrame(
        {
            "sample_id": dataset.sample_ids,
            "optical_fault": optical_fault.to_numpy(dtype=bool),
            "thermal_fault": thermal_fault.to_numpy(dtype=bool),
            "acoustic_fault": np.asarray(acoustic_fault, dtype=bool),
            "mixed_fault": np.asarray(mixed_fault, dtype=bool),
            "is_clean": np.asarray(is_clean, dtype=bool),
            "fault_case": fault_case,
            "fault_severity": fault_severity,
        }
    )


def _column_scale(values: np.ndarray) -> np.ndarray:
    """按列估计扰动尺度，避免零方差列导致噪声退化。"""

    scale = np.std(values, axis=0, ddof=0)
    return np.where(scale < 1e-9, 1.0, scale)


def inject_faults(dataset: PatentDataset, case: str, severity: str = "medium", seed: int = 42) -> PatentDataset:
    """在测试副本上注入人工故障，用于验证动态权重的容错表现。"""

    # 先校验用户给的故障类型和强度档位是否合法。
    if severity not in _SEVERITY_SCALE:
        raise ValueError(f"Unsupported severity '{severity}'. Choose from {sorted(_SEVERITY_SCALE)}.")
    if case not in {"clean", "optical_fail", "thermal_drift", "acoustic_bias", "mixed_fail"}:
        raise ValueError(f"Unsupported fault case '{case}'.")

    rng = np.random.default_rng(seed)
    scale = _SEVERITY_SCALE[severity]
    acoustic = dataset.acoustic.copy()
    optical = dataset.optical.copy()
    thermal = dataset.thermal.copy()

    # 第 1 段：按故障类型改动对应模态特征。
    # 光学故障主要改动 NDIR 相关特征。
    if case in {"optical_fail", "mixed_fail"}:
        optical_scale = _column_scale(optical)
        optical += rng.normal(0.0, optical_scale * scale, size=optical.shape)
        optical *= rng.normal(1.0 - scale * 0.25, scale * 0.08, size=optical.shape)

    # 热导故障表现为整体偏移和漂移增大。
    if case in {"thermal_drift", "mixed_fail"}:
        thermal_scale = _column_scale(thermal)
        drift = rng.normal(thermal_scale * scale, thermal_scale * scale * 0.15, size=thermal.shape)
        thermal += drift

    # 声学故障同时制造随机偏差和定向偏移。
    if case in {"acoustic_bias", "mixed_fail"}:
        acoustic_scale = _column_scale(acoustic)
        acoustic += rng.normal(0.0, acoustic_scale * scale, size=acoustic.shape)
        tof_idx = _required_acoustic_index(dataset, "TOF")
        amp_idx = _required_acoustic_index(dataset, "Amp")
        acoustic[:, tof_idx] += np.std(acoustic[:, tof_idx], ddof=0) * scale
        acoustic[:, amp_idx] *= max(0.05, 1.0 - scale * 0.45)

    # 第 2 段：标签保持不变，只在 metadata 中记录这次注入的故障语义。
    metadata = dataset.metadata.copy()
    metadata["fault_case"] = case
    metadata["fault_severity"] = severity
    return PatentDataset(
        sample_ids=dataset.sample_ids.copy(),
        acoustic=acoustic,
        optical=optical,
        thermal=thermal,
        environment=dataset.environment.copy(),
        targets=dataset.targets.copy(),
        component_names=dataset.component_names,
        metadata=metadata,
        acoustic_columns=dataset.acoustic_columns,
        optical_columns=dataset.optical_columns,
        thermal_columns=dataset.thermal_columns,
        environment_columns=dataset.environment_columns,
    )
