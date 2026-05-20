# 仿真数据 IO 工具：CSV/JSON 写入、数值格式化、索引/标签行构造。
#
# `fmt` 复用 sim_common 内的 V1 兼容格式化逻辑，避免依赖顶层 scripts 包。

import csv
import json

from .v1_helpers import fmt


def write_csv(path, fieldnames, rows):
    """写入 CSV 文件，自动创建父目录。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, obj):
    """写入 JSON 文件，自动创建父目录。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=False)


def build_index_rows(conditions, timesteps, dt_s):
    """生成 sequence_index.csv 的行数据。

    与 V2 / V3 共享字段：sequence_id, mixture_id, stage_profile, status, n_timesteps, dt_s。
    """
    return [
        {
            "sequence_id": row["sequence_id"],
            "mixture_id": row["sequence_id"],
            "stage_profile": "standard_exposure",  # 当前仅支持标准曝光时序
            "status": "synthetic_measurement",
            "n_timesteps": str(timesteps),
            "dt_s": fmt(dt_s, 1),
        }
        for row in conditions
    ]


def build_label_rows(conditions, label_fields):
    """生成 sequence_labels.csv 的行数据。"""
    return [
        {"sequence_id": row["sequence_id"], **{name: row[name] for name in label_fields}}
        for row in conditions
    ]
