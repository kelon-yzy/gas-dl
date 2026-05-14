"""脚本共用的 CLI 工具与小型数据集裁剪函数。"""

from __future__ import annotations

import argparse

import numpy as np

from patent_model.dataset import PatentDataset


def positive_int(value: str) -> int:
    """argparse 类型校验：必须为正整数。"""

    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def limit_dataset(dataset: PatentDataset, limit: int | None) -> PatentDataset:
    """调试用：按前 N 条样本截断 PatentDataset。"""

    if limit is None or limit >= dataset.n_samples:
        return dataset
    return dataset.subset(np.arange(limit))
