# 通用配气/工况条件采样：三组分 + 四组分。
#
# 复用 V1 _condition_rows 作为基础，过滤掉 calibration_control 样本。
# V2 与 V3 直接调用此处函数；V3 在拿到结果后可再做窗口过滤。

import math

from .v1_helpers import _condition_rows

from .constants import FOUR_COMPONENT_CH4_MIN, FOUR_COMPONENT_N2_RANGE
from .io import fmt


def build_synthetic_condition_rows(sequence_count, rng):
    """从 V1 条件行中筛选 synthetic_measurement 样本并转为序列条件格式（三组分）。

    生成策略：
    1. 先用 V1 的 _condition_rows 生成一定数量的候选条件行
    2. 过滤掉 calibration_control 样本（仅保留 synthetic_measurement）
    3. 如果候选数不够，逐步增大请求量（×1.25）重新采样
    4. 取前 sequence_count 个，赋予 Q 开头的 sequence_id

    V1 约 20% 的样本是 calibration_control（baseline/purge 阶段），
    所以需要 oversample 来保证拿到足够的 synthetic_measurement 样本。
    """
    requested = max(8, int(math.ceil(sequence_count / 0.70)) + 8)
    synthetic_rows = []

    while len(synthetic_rows) < sequence_count:
        rows = _condition_rows(requested, rng)
        synthetic_rows = [row for row in rows if row["status"] == "synthetic_measurement"]
        requested = int(requested * 1.25) + 1

    out_rows = []
    for index, row in enumerate(synthetic_rows[:sequence_count], start=1):
        out_rows.append(
            {
                "sequence_id": f"Q{index:06d}",
                "mixture_id": row["mixture_id"],
                "x_H2": row["x_H2"],
                "x_CH4": row["x_CH4"],
                "x_CO2": row["x_CO2"],
                "x_N2": "0.000000",          # 序列条件中 N2 统一为 0
                "T_C_base": row["T_C"],      # base 后缀表示这是稳态目标值
                "P_MPa_base": row["P_MPa"],
                "H_RH_base": row["H_RH"],
                "L_m_base": row["L_m"],
                "status": "synthetic_measurement",
            }
        )
    return out_rows


def build_synthetic_condition_rows_four_component(
    sequence_count, rng, n2_range=FOUR_COMPONENT_N2_RANGE,
):
    """生成四组分序列条件行。"""
    requested = max(8, int(math.ceil(sequence_count / 0.70)) + 8)
    synthetic_rows = []

    while len(synthetic_rows) < sequence_count:
        rows = _condition_rows(requested, rng)
        synthetic_rows = [row for row in rows if row["status"] == "synthetic_measurement"]
        requested = int(requested * 1.25) + 1

    out_rows = []
    for index, row in enumerate(synthetic_rows[:sequence_count], start=1):
        components = sample_four_component_measurement(
            rng,
            base_h2=float(row["x_H2"]),
            base_co2=float(row["x_CO2"]),
            n2_range=n2_range,
        )
        out_rows.append(
            {
                "sequence_id": f"Q{index:06d}",
                "mixture_id": row["mixture_id"],
                "x_H2": fmt(components["x_H2"], 6),
                "x_CH4": fmt(components["x_CH4"], 6),
                "x_CO2": fmt(components["x_CO2"], 6),
                "x_N2": fmt(components["x_N2"], 6),
                "T_C_base": row["T_C"],
                "P_MPa_base": row["P_MPa"],
                "H_RH_base": row["H_RH"],
                "L_m_base": row["L_m"],
                "status": "synthetic_measurement",
            }
        )
    return out_rows


def sample_four_component_measurement(rng, base_h2, base_co2, n2_range=FOUR_COMPONENT_N2_RANGE):
    """采样四组分测量条件，保持 CH4 为主体组分。"""
    n2_min, n2_max = n2_range

    for _ in range(64):
        x_h2 = min(base_h2, rng.uniform(0.0, 30.0))
        x_co2 = min(max(base_co2, 0.0), rng.uniform(0.0, 15.0))
        x_n2 = rng.uniform(n2_min, n2_max)
        x_ch4 = 100.0 - x_h2 - x_co2 - x_n2
        if x_ch4 >= FOUR_COMPONENT_CH4_MIN:
            x_h2, x_ch4, x_co2, x_n2 = normalize_measurement_components(x_h2, x_ch4, x_co2, x_n2)
            return {
                "x_H2": x_h2,
                "x_CH4": x_ch4,
                "x_CO2": x_co2,
                "x_N2": x_n2,
            }

    x_n2 = min(n2_max, max(n2_min, 100.0 - base_h2 - base_co2 - FOUR_COMPONENT_CH4_MIN))
    x_ch4 = 100.0 - base_h2 - base_co2 - x_n2
    x_h2, x_ch4, x_co2, x_n2 = normalize_measurement_components(base_h2, x_ch4, base_co2, x_n2)
    return {
        "x_H2": x_h2,
        "x_CH4": x_ch4,
        "x_CO2": x_co2,
        "x_N2": x_n2,
    }


def normalize_measurement_components(x_h2, x_ch4, x_co2, x_n2):
    total = x_h2 + x_ch4 + x_co2 + x_n2
    if total <= 0.0:
        raise ValueError("component total must be positive")
    scale = 100.0 / total
    return x_h2 * scale, x_ch4 * scale, x_co2 * scale, x_n2 * scale
