# V2 数据集切分薄壳：实际逻辑在 sim_common.splits。
#
# 维持 V2 模块旧名（split_distribution / split_warnings / split_rows_for_conditions）
# 以兼容 sim_v2/orchestrator.py 与现有测试的 import 路径。

from sim_common.splits import (
    build_split_rows as split_rows_for_conditions,
    collect_split_warnings as split_warnings,
    compute_split_distribution as _compute_distribution_impl,
    split_mixture_ids,
)

from .constants import LABEL_FIELDS


def split_distribution(conditions, split_rows, label_fields=LABEL_FIELDS):
    """V2 默认 label_fields 为三组分。委托给 sim_common 实现。"""
    return _compute_distribution_impl(conditions, split_rows, label_fields)
