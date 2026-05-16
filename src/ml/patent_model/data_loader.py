"""训练数据读取与分组切分逻辑。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from patent_model.config import (
    ACOUSTIC_FEATURE_COLUMNS,
    COMPONENT_NAMES,
    ENVIRONMENT_FEATURE_COLUMNS,
    FOUR_COMPONENT_NAMES,
    FOUR_TARGET_COLUMNS,
    OPTICAL_FEATURE_COLUMNS,
    TARGET_COLUMNS,
    THERMAL_FEATURE_COLUMNS,
)
from patent_model.dataset import PatentDataset
from patent_model.feature_profiles import get_feature_profile


METADATA_FILTER_CHOICES = ("none", "detection")
STAGE_FILTER_CHOICES = ("none", "stable")
STABLE_STAGE_IDS = ("distance_stage", "pressure_stage")
QUALITY_FILTER_CHOICES = ("none", "strict")
DUPLICATE_FILTER_CHOICES = ("none", "per_mixture_limit")
DETECTION_STATUSES = ("synthetic_measurement",)
DETECTION_STAGES = ("distance_stage", "pressure_stage")
PHYSICAL_RANGE_LIMITS = {
    "sound_speed": (200.0, 800.0),
    "attenuation_alpha": (0.0, 10.0),
}
LABEL_CLOSURE_EXPECTED_SUM = 100.0
LABEL_CLOSURE_TOLERANCE = 1.0
FILTER_STAT_COLUMNS = (
    "sound_speed",
    "attenuation_alpha",
    "lambda_mix_calibrated",
    "x_H2",
    "x_CH4",
    "x_CO2",
    "x_N2",
    "target_sum",
)


# 基础 I/O 校验函数。主流程只负责拼装，不把这些检查内联到大函数里。
def _read_csv(path: Path) -> pd.DataFrame:
    """读取必需的 CSV，文件缺失时直接报错。"""

    if not path.exists():
        raise FileNotFoundError(f"Required data file not found: {path}")
    return pd.read_csv(path)


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...] | list[str], source: str) -> None:
    """校验输入表是否包含后续建模必需字段。"""

    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def _profile_columns(frame: pd.DataFrame, configured: tuple[str, ...] | list[str] | None) -> list[str]:
    if configured is None:
        return [column for column in frame.columns if column != "sample_id"]
    return list(configured)


def _target_spec(profile_spec: dict[str, object]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    component_mode = str(profile_spec.get("component_mode", "three"))
    if component_mode == "four":
        return FOUR_COMPONENT_NAMES, FOUR_TARGET_COLUMNS
    return COMPONENT_NAMES, TARGET_COLUMNS


def _normalize_metadata_filter(metadata_filter: str | None) -> str:
    """规范化元数据筛选模式，未知模式直接报错。"""

    mode = "none" if metadata_filter is None else str(metadata_filter).strip().lower()
    if mode not in METADATA_FILTER_CHOICES:
        raise ValueError(f"Unknown metadata_filter: {metadata_filter}. Expected one of {METADATA_FILTER_CHOICES}.")
    return mode


def _normalize_quality_filter(filter_name: str, filter_value: str | None) -> str:
    """规范化质量筛选模式，未知模式直接报错。"""

    mode = "none" if filter_value is None else str(filter_value).strip().lower()
    if mode not in QUALITY_FILTER_CHOICES:
        raise ValueError(f"Unknown {filter_name}: {filter_value}. Expected one of {QUALITY_FILTER_CHOICES}.")
    return mode


def _normalize_stage_filter(stage_filter: str | None) -> str:
    """规范化阶段筛选模式，未知模式直接报错。"""

    mode = "none" if stage_filter is None else str(stage_filter).strip().lower()
    if mode not in STAGE_FILTER_CHOICES:
        raise ValueError(f"Unknown stage_filter: {stage_filter}. Expected one of {STAGE_FILTER_CHOICES}.")
    return mode


def _normalize_duplicate_filter(duplicate_filter: str | None) -> str:
    """规范化重复度筛选模式，未知模式直接报错。"""

    mode = "none" if duplicate_filter is None else str(duplicate_filter).strip().lower()
    if mode not in DUPLICATE_FILTER_CHOICES:
        raise ValueError(f"Unknown duplicate_filter: {duplicate_filter}. Expected one of {DUPLICATE_FILTER_CHOICES}.")
    return mode


def _value_counts(series: pd.Series) -> dict[str, int]:
    """把 value_counts 转成稳定、可写入 JSON 的字典。"""

    counts = series.value_counts(dropna=False)
    return {str(key): int(value) for key, value in sorted(counts.items(), key=lambda item: str(item[0]))}


def _repeat_stats(metadata: pd.DataFrame) -> dict[str, float | int]:
    """统计每个 mixture 的重复采样规模，用于 Phase 1c 前后对比。"""

    if "mixture_id" not in metadata.columns or metadata.empty:
        return {
            "unique_mixtures": 0,
            "min_samples_per_mixture": 0,
            "max_samples_per_mixture": 0,
            "mean_samples_per_mixture": 0.0,
            "mixture_sample_ratio": 0.0,
        }
    counts = metadata["mixture_id"].value_counts(dropna=True)
    unique_mixtures = int(counts.size)
    sample_count = int(len(metadata))
    return {
        "unique_mixtures": unique_mixtures,
        "min_samples_per_mixture": int(counts.min()),
        "max_samples_per_mixture": int(counts.max()),
        "mean_samples_per_mixture": float(counts.mean()),
        "mixture_sample_ratio": float(unique_mixtures / sample_count) if sample_count > 0 else 0.0,
    }


def _metadata_stats_frame(dataset: PatentDataset) -> pd.DataFrame:
    """合并 metadata 与目标列，保证筛选报告覆盖四组分标签分布。"""

    frame = dataset.metadata.copy()
    for component_index, component_name in enumerate(dataset.component_names):
        target_column = f"x_{component_name}"
        if target_column not in frame.columns:
            frame[target_column] = dataset.targets[:, component_index]
    frame["target_sum"] = dataset.targets.sum(axis=1)
    return frame


def _stats_dict(frame: pd.DataFrame) -> dict[str, object]:
    """提取样本数、阶段分布和关键特征统计，供筛选报告复用。"""

    stats: dict[str, object] = {
        "samples": int(len(frame)),
        "unique_mixtures": int(frame["mixture_id"].nunique(dropna=True)) if "mixture_id" in frame.columns else 0,
    }
    if "status" in frame.columns:
        stats["status_counts"] = _value_counts(frame["status"])
    if "stage_id" in frame.columns:
        stats["stage_counts"] = _value_counts(frame["stage_id"])
    feature_stats: dict[str, dict[str, float | None]] = {}
    for column in FILTER_STAT_COLUMNS:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        finite = values[np.isfinite(values.to_numpy(dtype=float))]
        if finite.empty:
            feature_stats[column] = {"mean": None, "std": None, "min": None, "max": None}
            continue
        feature_stats[column] = {
            "mean": float(finite.mean()),
            "std": float(finite.std(ddof=0)),
            "min": float(finite.min()),
            "max": float(finite.max()),
        }
    stats["feature_stats"] = feature_stats
    return stats


def _build_detection_mask(metadata: pd.DataFrame) -> pd.Series:
    """Phase 1a 的检测任务样本定义：合成检测状态 + 距离/压力阶段。"""

    return metadata["status"].isin(DETECTION_STATUSES) & metadata["stage_id"].isin(DETECTION_STAGES)


def _removed_reason_counts(reasons: pd.Series) -> dict[str, int]:
    """统计剔除原因，空白原因不计入。"""

    non_empty = reasons[reasons.astype(str).str.len() > 0]
    if non_empty.empty:
        return {}
    return _value_counts(non_empty)


def _apply_metadata_filter(dataset: PatentDataset, metadata_filter: str) -> PatentDataset:
    """按元数据筛选样本，并把筛选前后统计写入 dataset.filter_report。"""

    if metadata_filter == "none":
        return dataset

    before_frame = _metadata_stats_frame(dataset)
    before_stats = _stats_dict(before_frame)
    if metadata_filter == "detection":
        keep_mask = _build_detection_mask(dataset.metadata)
        criteria = {
            "status_in": list(DETECTION_STATUSES),
            "stage_id_in": list(DETECTION_STAGES),
        }
    else:
        raise ValueError(f"Unknown metadata_filter: {metadata_filter}.")

    keep_indices = np.flatnonzero(keep_mask.to_numpy(dtype=bool))
    if keep_indices.size == 0:
        raise ValueError(f"metadata_filter={metadata_filter} removed all samples.")

    filtered = dataset.subset(keep_indices)
    after_frame = _metadata_stats_frame(filtered)
    after_stats = _stats_dict(after_frame)
    report = {
        "mode": metadata_filter,
        "criteria": criteria,
        "before_samples": int(dataset.n_samples),
        "after_samples": int(filtered.n_samples),
        "removed_samples": int(dataset.n_samples - filtered.n_samples),
        "before_unique_mixtures": int(before_stats["unique_mixtures"]),
        "after_unique_mixtures": int(after_stats["unique_mixtures"]),
        "before": before_stats,
        "after": after_stats,
    }
    filter_report = dict(dataset.filter_report)
    filter_report["metadata_filter"] = report
    return replace(filtered, filter_report=filter_report)


def _apply_stage_filter(dataset: PatentDataset, stage_filter: str) -> PatentDataset:
    """按采样阶段筛选稳定检测样本。"""

    if stage_filter == "none":
        return dataset
    if stage_filter != "stable":
        raise ValueError(f"Unknown stage_filter: {stage_filter}.")

    before_frame = _metadata_stats_frame(dataset)
    before_stats = _stats_dict(before_frame)
    keep_mask = dataset.metadata["stage_id"].isin(STABLE_STAGE_IDS)
    keep_indices = np.flatnonzero(keep_mask.to_numpy(dtype=bool))
    if keep_indices.size == 0:
        raise ValueError(f"stage_filter={stage_filter} removed all samples.")

    filtered = dataset.subset(keep_indices)
    after_frame = _metadata_stats_frame(filtered)
    after_stats = _stats_dict(after_frame)
    report = {
        "mode": stage_filter,
        "criteria": {"stage_id_in": list(STABLE_STAGE_IDS)},
        "before_samples": int(dataset.n_samples),
        "after_samples": int(filtered.n_samples),
        "removed_samples": int(dataset.n_samples - filtered.n_samples),
        "before_unique_mixtures": int(before_stats["unique_mixtures"]),
        "after_unique_mixtures": int(after_stats["unique_mixtures"]),
        "before": before_stats,
        "after": after_stats,
    }
    filter_report = dict(dataset.filter_report)
    filter_report["stage_filter"] = report
    return replace(filtered, filter_report=filter_report)


def _apply_physical_range_filter(dataset: PatentDataset, physical_range_filter: str) -> PatentDataset:
    """剔除物理量越界或关键物理特征非有限的样本。"""

    if physical_range_filter == "none":
        return dataset
    if physical_range_filter != "strict":
        raise ValueError(f"Unknown physical_range_filter: {physical_range_filter}.")

    stats_frame = _metadata_stats_frame(dataset)
    reasons = pd.Series("", index=stats_frame.index, dtype=object)
    keep_mask = pd.Series(True, index=stats_frame.index, dtype=bool)
    for column, (lower, upper) in PHYSICAL_RANGE_LIMITS.items():
        values = pd.to_numeric(stats_frame[column], errors="coerce")
        finite_mask = np.isfinite(values.to_numpy(dtype=float))
        non_finite_mask = pd.Series(~finite_mask, index=stats_frame.index)
        range_mask = pd.Series(values.between(lower, upper, inclusive="both"), index=stats_frame.index).fillna(False)
        out_of_range_mask = ~non_finite_mask & ~range_mask
        keep_mask &= ~non_finite_mask & ~out_of_range_mask
        reasons.loc[non_finite_mask & (reasons == "")] = f"non_finite_{column}"
        reasons.loc[out_of_range_mask & (reasons == "")] = f"{column}_out_of_range"

    keep_indices = np.flatnonzero(keep_mask.to_numpy(dtype=bool))
    if keep_indices.size == 0:
        raise ValueError("physical_range_filter=strict removed all samples.")

    filtered = dataset.subset(keep_indices)
    report = {
        "mode": physical_range_filter,
        "criteria": {
            column: {"min": lower, "max": upper}
            for column, (lower, upper) in PHYSICAL_RANGE_LIMITS.items()
        },
        "before_samples": int(dataset.n_samples),
        "after_samples": int(filtered.n_samples),
        "removed_samples": int(dataset.n_samples - filtered.n_samples),
        "before_unique_mixtures": int(stats_frame["mixture_id"].nunique(dropna=True)),
        "after_unique_mixtures": int(filtered.metadata["mixture_id"].nunique(dropna=True)),
        "removed_reason_counts": _removed_reason_counts(reasons[~keep_mask]),
        "before": _stats_dict(stats_frame),
        "after": _stats_dict(_metadata_stats_frame(filtered)),
    }
    filter_report = dict(dataset.filter_report)
    filter_report["physical_range_filter"] = report
    return replace(filtered, filter_report=filter_report)


def _apply_label_closure_filter(dataset: PatentDataset, label_closure_filter: str) -> PatentDataset:
    """剔除四组分标签和不闭合的样本。"""

    if label_closure_filter == "none":
        return dataset
    if label_closure_filter != "strict":
        raise ValueError(f"Unknown label_closure_filter: {label_closure_filter}.")
    if len(dataset.component_names) != 4:
        raise ValueError("label_closure_filter=strict requires four-component targets.")

    target_sums = dataset.targets.sum(axis=1)
    finite_mask = np.isfinite(dataset.targets).all(axis=1) & np.isfinite(target_sums)
    lower = LABEL_CLOSURE_EXPECTED_SUM - LABEL_CLOSURE_TOLERANCE
    upper = LABEL_CLOSURE_EXPECTED_SUM + LABEL_CLOSURE_TOLERANCE
    closure_mask = (target_sums >= lower) & (target_sums <= upper)
    keep_mask = finite_mask & closure_mask
    reasons = pd.Series("", index=np.arange(dataset.n_samples), dtype=object)
    reasons.loc[~finite_mask] = "non_finite_target"
    reasons.loc[finite_mask & ~closure_mask] = "target_sum_out_of_range"

    keep_indices = np.flatnonzero(keep_mask)
    if keep_indices.size == 0:
        raise ValueError("label_closure_filter=strict removed all samples.")

    filtered = dataset.subset(keep_indices)
    report = {
        "mode": label_closure_filter,
        "criteria": {
            "expected_sum": LABEL_CLOSURE_EXPECTED_SUM,
            "absolute_tolerance": LABEL_CLOSURE_TOLERANCE,
        },
        "before_samples": int(dataset.n_samples),
        "after_samples": int(filtered.n_samples),
        "removed_samples": int(dataset.n_samples - filtered.n_samples),
        "before_unique_mixtures": int(dataset.metadata["mixture_id"].nunique(dropna=True)),
        "after_unique_mixtures": int(filtered.metadata["mixture_id"].nunique(dropna=True)),
        "removed_reason_counts": _removed_reason_counts(reasons[~keep_mask]),
        "before": _stats_dict(_metadata_stats_frame(dataset)),
        "after": _stats_dict(_metadata_stats_frame(filtered)),
    }
    filter_report = dict(dataset.filter_report)
    filter_report["label_closure_filter"] = report
    return replace(filtered, filter_report=filter_report)


def _apply_duplicate_filter(
    dataset: PatentDataset,
    duplicate_filter: str,
    duplicate_per_mixture_limit: int | None,
    duplicate_filter_seed: int,
) -> PatentDataset:
    """按 mixture 重复度做可复现降采样，缓解高重复样本主导训练。"""

    if duplicate_filter == "none":
        return dataset
    if duplicate_filter != "per_mixture_limit":
        raise ValueError(f"Unknown duplicate_filter: {duplicate_filter}.")
    if duplicate_per_mixture_limit is None or int(duplicate_per_mixture_limit) < 1:
        raise ValueError("duplicate_per_mixture_limit must be >= 1 when duplicate_filter=per_mixture_limit.")

    limit = int(duplicate_per_mixture_limit)
    metadata = dataset.metadata.reset_index(drop=True)
    rng = np.random.default_rng(duplicate_filter_seed)
    keep_mask = np.zeros(dataset.n_samples, dtype=bool)
    reasons = pd.Series("", index=np.arange(dataset.n_samples), dtype=object)

    grouped = metadata.groupby("mixture_id", sort=True).indices
    for _, raw_indices in grouped.items():
        mixture_indices = np.asarray(raw_indices, dtype=int)
        if mixture_indices.size <= limit:
            keep_mask[mixture_indices] = True
            continue
        selected = np.sort(rng.choice(mixture_indices, size=limit, replace=False))
        keep_mask[selected] = True
        removed = np.setdiff1d(mixture_indices, selected, assume_unique=False)
        reasons.loc[removed] = "mixture_downsampled"

    keep_indices = np.flatnonzero(keep_mask)
    if keep_indices.size == 0:
        raise ValueError("duplicate_filter=per_mixture_limit removed all samples.")

    filtered = dataset.subset(keep_indices)
    report = {
        "mode": duplicate_filter,
        "criteria": {
            "per_mixture_limit": limit,
            "seed": int(duplicate_filter_seed),
        },
        "before_samples": int(dataset.n_samples),
        "after_samples": int(filtered.n_samples),
        "removed_samples": int(dataset.n_samples - filtered.n_samples),
        "before_unique_mixtures": int(dataset.metadata["mixture_id"].nunique(dropna=True)),
        "after_unique_mixtures": int(filtered.metadata["mixture_id"].nunique(dropna=True)),
        "before_repeat_stats": _repeat_stats(dataset.metadata),
        "after_repeat_stats": _repeat_stats(filtered.metadata),
        "removed_reason_counts": _removed_reason_counts(reasons[~keep_mask]),
    }
    filter_report = dict(dataset.filter_report)
    filter_report["duplicate_filter"] = report
    return replace(filtered, filter_report=filter_report)


def load_patent_dataset(
    data_dir: str | Path,
    profile: str = "raw_tph",
    metadata_filter: str | None = "none",
    stage_filter: str | None = "none",
    physical_range_filter: str | None = "none",
    label_closure_filter: str | None = "none",
    duplicate_filter: str | None = "none",
    duplicate_per_mixture_limit: int | None = None,
    duplicate_filter_seed: int = 42,
) -> PatentDataset:
    """从导出包读取三模态训练表，并组装成统一数据结构。"""

    # 第 1 步：读取训练、标签、工况和补充特征表。
    base = Path(data_dir)
    metadata_filter_mode = _normalize_metadata_filter(metadata_filter)
    stage_filter_mode = _normalize_stage_filter(stage_filter)
    physical_range_filter_mode = _normalize_quality_filter("physical_range_filter", physical_range_filter)
    label_closure_filter_mode = _normalize_quality_filter("label_closure_filter", label_closure_filter)
    duplicate_filter_mode = _normalize_duplicate_filter(duplicate_filter)
    profile_spec = get_feature_profile(profile)
    acoustic = _read_csv(base / str(profile_spec["acoustic_file"]))
    optical = _read_csv(base / str(profile_spec["optical_file"]))
    thermal = _read_csv(base / str(profile_spec["thermal_file"]))
    labels = _read_csv(base / "labels" / "labels.csv")
    condition = _read_csv(base / "condition_grid_v1.csv")
    feature_table = _read_csv(base / str(profile_spec["feature_table_file"]))
    component_names, target_columns = _target_spec(profile_spec)
    acoustic_columns = _profile_columns(acoustic, profile_spec["acoustic_columns"])
    optical_columns = _profile_columns(optical, profile_spec["optical_columns"])
    thermal_columns = _profile_columns(thermal, profile_spec["thermal_columns"])
    environment_columns = list(profile_spec["environment_columns"])

    # 第 2 步：校验每张表最少要有的列，缺列时尽早失败。
    _require_columns(acoustic, ["sample_id", *acoustic_columns], str(profile_spec["acoustic_file"]))
    _require_columns(optical, ["sample_id", *optical_columns], str(profile_spec["optical_file"]))
    _require_columns(thermal, ["sample_id", *thermal_columns], str(profile_spec["thermal_file"]))
    _require_columns(labels, ["sample_id", *TARGET_COLUMNS], "labels.csv")
    required_condition_columns = ["sample_id", "mixture_id", "stage_id", "repeat_id", "status"]
    if target_columns == FOUR_TARGET_COLUMNS:
        required_condition_columns.append("x_N2")
    _require_columns(condition, required_condition_columns, "condition_grid_v1.csv")

    # 三张训练表必须按同一批 sample_id 严格对齐，后续才可以直接拼接。
    sample_ids = acoustic["sample_id"].tolist()
    if sample_ids != optical["sample_id"].tolist() or sample_ids != thermal["sample_id"].tolist():
        raise ValueError("Training modality sample_id order is not aligned.")

    # 第 3 步：只用 sample_id 对齐标签；各模态矩阵直接来自各自训练表。
    frame = acoustic[["sample_id"]].merge(
        optical[["sample_id"]],
        on="sample_id",
        how="inner",
        validate="one_to_one",
    )
    frame = frame.merge(thermal[["sample_id"]], on="sample_id", how="inner", validate="one_to_one")
    frame = frame.merge(labels[["sample_id", *TARGET_COLUMNS]], on="sample_id", how="inner", validate="one_to_one")
    if len(frame) != len(acoustic):
        raise ValueError("Merged training frame lost rows; check sample_id coverage.")

    # 第 4 步：补上不直接进模型、但训练切分和分析会用到的元数据。
    # 再补充工况、阶段和质量分析字段，这部分主要给分组切分和结果分析使用。
    condition_columns = [
        "sample_id",
        "mixture_id",
        "stage_id",
        "repeat_id",
        "status",
        "pressure_stage",
        "distance_stage",
        "piston_position_m",
    ]
    if target_columns == FOUR_TARGET_COLUMNS or "x_N2" in condition.columns:
        condition_columns.append("x_N2")
    for optional_column in ("source_timestep", "source_phase_id"):
        if optional_column in condition.columns:
            condition_columns.append(optional_column)
    feature_columns = [
        "sample_id",
        "sound_speed",
        "attenuation_alpha",
        "ndir_ch4_saturated",
        "ndir_co2_saturated",
        "optical_baseline_drift_ch4",
        "optical_baseline_drift_co2",
        "thermal_baseline_drift",
        "lambda_mix_calibrated",
        "calibration_status",
    ]
    metadata = frame[["sample_id"]].merge(condition[condition_columns], on="sample_id", how="left", validate="one_to_one")
    metadata = metadata.merge(feature_table[feature_columns], on="sample_id", how="left", validate="one_to_one")
    metadata = metadata.reset_index(drop=True)
    if metadata["mixture_id"].isna().any():
        raise ValueError("Some training samples are missing mixture_id metadata.")

    targets_frame = frame[list(TARGET_COLUMNS)].copy()
    if target_columns == FOUR_TARGET_COLUMNS:
        if "x_N2" not in metadata.columns:
            raise ValueError("Four-component mode requires x_N2 in condition metadata.")
        if metadata["x_N2"].isna().any():
            raise ValueError("Four-component mode requires non-null x_N2 in condition metadata.")
        targets_frame["x_N2"] = metadata["x_N2"].to_numpy(dtype=float)

    environment_frame = acoustic[["sample_id"]].merge(
        condition[["sample_id", *environment_columns]],
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    if environment_frame[environment_columns].isna().any().any():
        raise ValueError("Some training samples are missing environment metadata.")

    # 第 5 步：按字段拆回 PatentDataset，后续各模块统一消费这个结构。
    dataset = PatentDataset(
        sample_ids=frame["sample_id"].to_numpy(dtype=object),
        acoustic=acoustic[acoustic_columns].to_numpy(dtype=float),
        optical=optical[optical_columns].to_numpy(dtype=float),
        thermal=thermal[thermal_columns].to_numpy(dtype=float),
        environment=environment_frame[environment_columns].to_numpy(dtype=float),
        targets=targets_frame[list(target_columns)].to_numpy(dtype=float),
        component_names=component_names,
        metadata=metadata,
        acoustic_columns=tuple(acoustic_columns),
        optical_columns=tuple(optical_columns),
        thermal_columns=tuple(thermal_columns),
        environment_columns=tuple(environment_columns),
        provenance={
            "data_dir": str(base.resolve()),
            "feature_profile": profile,
            "component_mode": str(profile_spec.get("component_mode", "three")),
        },
    )
    dataset = _apply_metadata_filter(dataset, metadata_filter_mode)
    dataset = _apply_stage_filter(dataset, stage_filter_mode)
    dataset = _apply_physical_range_filter(dataset, physical_range_filter_mode)
    dataset = _apply_label_closure_filter(dataset, label_closure_filter_mode)
    dataset = _apply_duplicate_filter(dataset, duplicate_filter_mode, duplicate_per_mixture_limit, duplicate_filter_seed)
    return dataset


def grouped_train_test_split(dataset: PatentDataset, test_ratio: float = 0.2, seed: int = 42) -> tuple[PatentDataset, PatentDataset]:
    """按 mixture_id 分组切分训练集和测试集，避免同组样本泄漏。"""

    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio must be between 0 and 1.")
    # 先打乱 group，再整组分配到 train/test，保证同组样本不会被拆散。
    groups = np.array(sorted(dataset.metadata["mixture_id"].unique()), dtype=object)
    if len(groups) < 2:
        raise ValueError("grouped_train_test_split requires at least 2 mixture groups.")
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    n_test_groups = min(len(groups) - 1, max(1, int(round(len(groups) * test_ratio))))
    test_groups = set(groups[:n_test_groups])
    test_mask = dataset.metadata["mixture_id"].isin(test_groups).to_numpy()
    train_idx = np.flatnonzero(~test_mask)
    test_idx = np.flatnonzero(test_mask)
    if train_idx.size == 0 or test_idx.size == 0:
        raise ValueError("grouped_train_test_split produced an empty train or test split.")
    return dataset.subset(train_idx), dataset.subset(test_idx)
