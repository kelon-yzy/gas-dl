# V2 数据集生成主入口。
#
# 两个 entry 共用 `_run_pipeline`：
#   - generate_sequence_dataset（三组分）
#   - generate_sequence_dataset_four_component（四组分，含 N2 标签）
#
# D4 应用：multi_path_baseline 旧 bool 参数已移除，调用方使用 multi_path_phase 字符串。

from pathlib import Path
import random

import numpy as np

from scripts.generate_v1_dataset import (
    DEFAULT_SAMPLE_COUNT,
    DEFAULT_SEED,
    _validate_sample_count,
)

from .acoustic_derived import (
    acoustic_derived_sequence_rows,
    acoustic_quality_summary,
)
from .conditions import (
    sequence_condition_rows,
    sequence_condition_rows_four_component,
)
from .constants import (
    ACOUSTIC_DERIVED_SEQUENCE_FIELDS,
    CONDITION_FIELDS,
    DEFAULT_DT_S,
    DEFAULT_TIMESTEPS,
    FOUR_COMPONENT_LABEL_FIELDS,
    FOUR_COMPONENT_N2_RANGE,
    LABEL_FIELDS,
    MODAL_SEQUENCE_FIELDS,
    SEQUENCE_CHANNELS,
    SEQUENCE_INDEX_FIELDS,
    SPLIT_FIELDS,
    normalize_multi_path_phase,
)
from .io import (
    build_output_paths,
    label_rows,
    sequence_index_rows,
    write_csv,
    write_json,
)
from .matrix_builder import build_sequence_matrix
from .quality import quality_summary
from .readme import write_readme
from .scalers import fit_sequence_scalers
from .splits import (
    split_distribution,
    split_mixture_ids,
    split_rows_for_conditions,
    split_warnings,
)


def generate_sequence_dataset(
    output_dir,
    sequence_count=DEFAULT_SAMPLE_COUNT,
    timesteps=DEFAULT_TIMESTEPS,
    seed=DEFAULT_SEED,
    dt_s=DEFAULT_DT_S,
    multi_path_phase=None,
):
    """生成完整的 V2 三组分时序仿真数据包。

    Args:
        output_dir: 输出根目录路径。
        sequence_count: 生成序列数，默认 2400。
        timesteps: 每条序列的时间步数，默认 120。
        seed: 随机种子，保证可复现。
        dt_s: 时间步长 (s)，默认 1.0。
        multi_path_phase: 'off' / 'baseline' / 'steady'，None 视为 'off'。

    Returns:
        dict: 输出文件路径映射（key=文件名标识, value=Path）。

    Raises:
        ValueError: sequence_count 超出限制或 timesteps < 4。
    """
    return _run_pipeline(
        output_dir=output_dir,
        sequence_count=sequence_count,
        timesteps=timesteps,
        seed=seed,
        dt_s=dt_s,
        multi_path_phase=multi_path_phase,
        four_component=False,
    )


def generate_sequence_dataset_four_component(
    output_dir,
    sequence_count=DEFAULT_SAMPLE_COUNT,
    timesteps=DEFAULT_TIMESTEPS,
    seed=DEFAULT_SEED,
    dt_s=DEFAULT_DT_S,
    n2_range=FOUR_COMPONENT_N2_RANGE,
    multi_path_phase=None,
):
    """生成四组分反演版本的 V2 时序仿真数据包。"""
    return _run_pipeline(
        output_dir=output_dir,
        sequence_count=sequence_count,
        timesteps=timesteps,
        seed=seed,
        dt_s=dt_s,
        multi_path_phase=multi_path_phase,
        four_component=True,
        n2_range=n2_range,
    )


def _run_pipeline(
    output_dir,
    sequence_count,
    timesteps,
    seed,
    dt_s,
    multi_path_phase,
    four_component,
    n2_range=FOUR_COMPONENT_N2_RANGE,
):
    """三/四组分共享的生成管线。

    流程：
      1. 用 V1 采样逻辑生成 sequence_count 个配气/工况条件
      2. 为每条序列构建 T×C 时序矩阵（一阶响应 + 噪声叠加）
      3. 按 mixture_id 分组划分 train/val/test
      4. 拟合 scaler（仅基于 train split）
      5. 写出所有产物文件（NPZ、CSV、JSON、README）

    主管线声学物理固定为 V1 链路；声学派生侧车 / 质量摘要按 acoustic_version="v1"
    字面量调用，保留 sidecar 工具继续支持 v2 校准链路的能力。
    """
    _validate_sample_count(sequence_count)
    if timesteps < 4:
        raise ValueError("timesteps must be >= 4")  # 至少 4 步以容纳 4 个阶段

    multi_path_phase = normalize_multi_path_phase(multi_path_phase)

    output_dir = Path(output_dir)
    paths = build_output_paths(output_dir)
    rng = random.Random(seed)

    # 步骤 1：生成配气/工况条件（复用 V1 采样逻辑）
    if four_component:
        conditions = sequence_condition_rows_four_component(sequence_count, rng, n2_range=n2_range)
        active_label_fields = FOUR_COMPONENT_LABEL_FIELDS
        dataset_version = "V2 sequence four_component_n2"
        dataset_dir_name = output_dir.name
    else:
        conditions = sequence_condition_rows(sequence_count, rng)
        active_label_fields = LABEL_FIELDS
        dataset_version = "V2 sequence"
        dataset_dir_name = "output_sequence"

    sequence_ids = [row["sequence_id"] for row in conditions]
    labels = np.array(
        [[float(row[name]) for name in active_label_fields] for row in conditions],
        dtype=np.float32,
    )

    # 步骤 2：构建时序张量 + 展开为长表行
    x_matrix, long_rows = build_sequence_matrix(
        conditions, timesteps, dt_s, rng,
        multi_path_phase=multi_path_phase,
    )
    derived_rows = acoustic_derived_sequence_rows(
        long_rows,
        acoustic_version="v1",
        multi_path_phase=multi_path_phase,
    )
    acoustic_summary = acoustic_quality_summary(
        conditions,
        long_rows,
        derived_rows,
        "v1",
        multi_path_phase,
    )

    # 步骤 3：划分 train/val/test（按 mixture_id 隔离）
    train_ids, val_ids, test_ids = split_mixture_ids([row["mixture_id"] for row in conditions], seed=seed)
    rows_by_split = split_rows_for_conditions(conditions, train_ids, val_ids, test_ids)

    # 步骤 4：写出所有 CSV 和 NPZ 产物
    write_csv(paths["sequence_index"], SEQUENCE_INDEX_FIELDS, sequence_index_rows(conditions, timesteps, dt_s))
    write_csv(paths["condition_grid_sequence"], CONDITION_FIELDS, conditions)
    write_csv(paths["modal_sequence_long"], MODAL_SEQUENCE_FIELDS, long_rows)
    write_csv(paths["acoustic_derived_sequence_long"], ACOUSTIC_DERIVED_SEQUENCE_FIELDS, derived_rows)
    write_csv(
        paths["sequence_labels"],
        ["sequence_id", *active_label_fields],
        label_rows(conditions, active_label_fields),
    )
    write_csv(paths["train_split"], SPLIT_FIELDS, rows_by_split["train"])
    write_csv(paths["val_split"], SPLIT_FIELDS, rows_by_split["val"])
    write_csv(paths["test_split"], SPLIT_FIELDS, rows_by_split["test"])

    # 压缩 NPZ：保存 X/y/sequence_ids/channel_names/label_names
    np.savez_compressed(
        paths["modal_sequence_npz"],
        X=x_matrix,
        y=labels,
        sequence_ids=np.array(sequence_ids),
        channel_names=np.array(SEQUENCE_CHANNELS),
        label_names=np.array(active_label_fields),
    )

    # 步骤 5：拟合 scaler（只在 train split 上计算 mean/std）
    train_sequence_ids = {row["sequence_id"] for row in rows_by_split["train"]}
    train_indexes = [index for index, sequence_id in enumerate(sequence_ids) if sequence_id in train_sequence_ids]
    sequence_scaler, modal_scaler = fit_sequence_scalers(x_matrix, train_indexes)
    write_json(paths["scaler_sequence"], sequence_scaler)
    write_json(paths["scaler_sequence_modal"], modal_scaler)

    # 步骤 6：质量报告和 README
    distribution = split_distribution(conditions, rows_by_split, active_label_fields)
    warnings = split_warnings(distribution, active_label_fields)
    write_json(
        paths["quality_summary"],
        quality_summary(
            sequence_ids=sequence_ids,
            timesteps=timesteps,
            split_distribution=distribution,
            split_warnings=warnings,
            label_fields=active_label_fields,
            dataset_version=dataset_version,
            acoustic_version="v1",
            multi_path_phase=multi_path_phase,
            acoustic_summary=acoustic_summary,
        ),
    )
    write_readme(
        paths["readme"],
        sequence_count=len(sequence_ids),
        timesteps=timesteps,
        split_distribution=distribution,
        label_fields=active_label_fields,
        dataset_dir_name=dataset_dir_name,
        dataset_version=dataset_version,
        acoustic_version="v1",
        multi_path_phase=multi_path_phase,
    )

    return paths
