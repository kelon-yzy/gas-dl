# 通用质量摘要骨架。
#
# V2 / V3 共享的字段（dataset_version / shape / split_* / channel/label / scaler_policy）
# 在这里组装成基础 dict；调用方再 update 自己的扩展字段（V2 的 acoustic / multi_path，
# V3 的 waveform / spec / storage_format 等）。


def build_common_summary(
    sequence_ids,
    timesteps,
    split_distribution,
    split_warnings,
    *,
    label_fields,
    dataset_version,
    simulation_level,
    channel_names,
    shape,
):
    """生成 V2/V3 共享的基础质量摘要 dict。

    Args:
        sequence_ids: 序列 ID 列表，仅用于推断 sequences 数量。
        timesteps: 时间步数。
        split_distribution: split → 统计字典（来自 sim_common.splits.compute_split_distribution）。
        split_warnings: 字符串列表（来自 sim_common.splits.collect_split_warnings）。
        label_fields: 标签字段顺序。
        dataset_version: 字符串标签，例如 "V2 sequence" 或 "V3 waveform sequence"。
        simulation_level: "structure_level_dynamic_simulation" / "waveform_level_dynamic_simulation"。
        channel_names: 通道名顺序（V2 是 12 通道，V3 slow 是 8 通道）。
        shape: 数据形状字典（调用方决定包含哪些键，例如 V3 包含 waveform_samples）。

    Returns:
        基础 summary dict。调用方 .update({...}) 添加扩展字段。
    """
    return {
        "dataset_version": dataset_version,
        "calibration_status": "pending",
        "simulation_level": simulation_level,
        "shape": dict(shape),
        "split_policy": "random_seed_grouped_by_mixture_id",
        "split_distribution": split_distribution,
        "split_warnings": split_warnings,
        "scaler_policy": "fit_z_score_scalers_on_train_split_only",
        "channel_names": list(channel_names),
        "label_names": list(label_fields),
    }
