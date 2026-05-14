# 通用 mixture_id 分组划分 + split 分布与告警。
#
# 关键设计：按 mixture_id（不是 sequence_id）划分 train/val/test，
# 确保同一配气条件的多条重复序列不会跨 split。
# label_fields 全部参数化，三组分 / 四组分 / waveform slow 包都直接复用。

import numpy as np

from .v1_helpers import DEFAULT_SEED


def split_mixture_ids(mixture_ids, train_ratio=0.70, val_ratio=0.15, seed=DEFAULT_SEED):
    """按 mixture_id 去重后随机分组划分 train/val/test。

    Args:
        mixture_ids: 所有序列的 mixture_id 列表（可重复）。
        train_ratio: 训练集比例，默认 0.70。
        val_ratio: 验证集比例，默认 0.15。
        seed: 随机种子，保证可复现。

    Returns:
        tuple[set, set, set]: (train_ids, val_ids, test_ids)
    """
    rng = np.random.default_rng(seed)
    # 去重并排序以保证可复现
    ids = np.array(sorted(set(mixture_ids)))
    rng.shuffle(ids)

    n = len(ids)
    if n == 0:
        return set(), set(), set()
    if n == 1:
        return {str(ids[0])}, set(), set()
    if n == 2:
        return {str(ids[0])}, set(), {str(ids[1])}

    # 至少保证每组有一个 mixture_id
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1

    train_ids = {str(value) for value in ids[:n_train]}
    val_ids = {str(value) for value in ids[n_train : n_train + n_val]}
    test_ids = {str(value) for value in ids[n_train + n_val :]}
    return train_ids, val_ids, test_ids


def build_split_rows(conditions, train_ids, val_ids, test_ids):
    """根据 mixture_id 将序列条件分配到 train/val/test 三组。"""
    rows = {"train": [], "val": [], "test": []}
    for condition in conditions:
        split_name = _split_name_for_mixture(condition["mixture_id"], train_ids, val_ids, test_ids)
        rows[split_name].append(
            {"sequence_id": condition["sequence_id"], "mixture_id": condition["mixture_id"]}
        )
    return rows


def _split_name_for_mixture(mixture_id, train_ids, val_ids, test_ids):
    if mixture_id in train_ids:
        return "train"
    if mixture_id in val_ids:
        return "val"
    if mixture_id in test_ids:
        return "test"
    raise ValueError(f"mixture_id {mixture_id!r} was not assigned to a split")


def compute_split_distribution(conditions, split_rows, label_fields):
    """统计每个 split 的序列数、mixture 数和标签分布（min/max/mean）。"""
    by_sequence_id = {row["sequence_id"]: row for row in conditions}
    distribution = {}
    for split_name, rows in split_rows.items():
        split_conditions = [by_sequence_id[row["sequence_id"]] for row in rows]
        distribution[split_name] = {
            "sequence_count": len(split_conditions),
            "mixture_count": len({row["mixture_id"] for row in split_conditions}),
        }
        for label_name in label_fields:
            values = [float(row[label_name]) for row in split_conditions]
            distribution[split_name][label_name] = _value_stats(values)
    return distribution


def _value_stats(values):
    if not values:
        return {"min": None, "max": None, "mean": None}
    return {
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def collect_split_warnings(distribution, label_fields):
    """检查 split 分布是否存在问题（空 split、某 split 缺少标签变化等）。

    返回警告字符串列表，若列表为空则分布质量正常。
    """
    warnings = []
    for split_name, stats in distribution.items():
        if stats["sequence_count"] == 0:
            warnings.append(f"{split_name} split has no sequences.")
        for label_name in label_fields:
            label_stats = stats[label_name]
            if label_stats["min"] is None or label_stats["max"] is None:
                continue
            if label_stats["min"] == label_stats["max"]:
                warnings.append(f"{split_name} split has no {label_name} range.")
    return warnings
