# 通用 z-score scaler 拟合：仅基于 train split 索引。
#
# V2 主张量与 V3 slow 慢通道张量共享同一份实现。差异只在
# channel_names / modal_groups / transform_target 三个参数上。

import numpy as np


def fit_z_score_scalers(
    matrix,
    train_indexes,
    channel_names,
    modal_groups,
    transform_target="X",
    channel_axis=2,
):
    """基于 train split 拟合 z-score 标准化器。

    输出两套 scaler：
    1. sequence_scaler: 全通道的 mean/std（单层结构）
    2. modal_scaler: 按模态分组各自统计

    注意：std 的最小值用 1.0 替代（防止常数通道导致除零）。

    Args:
        matrix: (N, T, C) ndarray，从 train_indexes 子集计算 mean/std。
        train_indexes: train split 在 matrix 第 0 轴上的索引列表。
        channel_names: 通道名顺序，与 matrix 第 channel_axis 轴对齐。
        modal_groups: dict[str, list[str]]，模态名到通道名列表的映射。
        transform_target: scaler JSON 中 transform_target 字段值，
                          V2 为 "X"，V3 slow 包为 "slow"。
        channel_axis: 通道在 matrix 上的轴下标。
    """
    if not train_indexes:
        raise ValueError("Cannot fit scalers without train sequences.")

    train_x = matrix[train_indexes]
    mean = train_x.mean(axis=(0, 1))
    std = train_x.std(axis=(0, 1))
    std = np.where(std > 1e-15, std, 1.0)  # 常数通道用 1.0 代替

    sequence_scaler = {
        "method": "z_score",
        "fit_scope": "train_split_only",
        "transform_target": transform_target,
        "channel_axis": channel_axis,
        "channel_names": list(channel_names),
        "mean": [float(value) for value in mean],
        "std": [float(value) for value in std],
    }

    modal_scaler = {
        "method": "z_score",
        "fit_scope": "train_split_only",
        "transform_target": transform_target,
        "channel_axis": channel_axis,
        "modal_groups": dict(modal_groups),
        "modal_stats": {},
    }
    channel_index = {channel: index for index, channel in enumerate(channel_names)}
    for modal_name, channels in modal_groups.items():
        modal_scaler["modal_stats"][modal_name] = {
            "channel_names": list(channels),
            "mean": [float(mean[channel_index[channel]]) for channel in channels],
            "std": [float(std[channel_index[channel]]) for channel in channels],
        }

    return sequence_scaler, modal_scaler
