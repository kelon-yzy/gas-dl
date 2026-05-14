# 仿真数据共享层。
#
# 这里集中所有 V2 / V3（以及未来 V4）共用的纯工具函数与公共常量。
# 设计原则：sim_common 不依赖 sim_v2 / V3 任何模块，只可被它们依赖。
# 历史上散落在 V1 (scripts/generate_v1_dataset.py)、V2 (sim_v2/) 与 V3
# (scripts/generate_waveform_dataset.py) 的重复实现均统一收拢到此处。

from .constants import (
    BASELINE_PATH_LMS,
    FOUR_COMPONENT_CH4_MIN,
    FOUR_COMPONENT_LABEL_FIELDS,
    FOUR_COMPONENT_N2_RANGE,
    MULTI_PATH_PHASE_BASELINE,
    MULTI_PATH_PHASE_CHOICES,
    MULTI_PATH_PHASE_OFF,
    MULTI_PATH_PHASE_STEADY,
    normalize_multi_path_phase,
)
from .conditions import (
    build_synthetic_condition_rows,
    build_synthetic_condition_rows_four_component,
    normalize_measurement_components,
    sample_four_component_measurement,
)
from .io import (
    build_index_rows,
    build_label_rows,
    fmt,
    write_csv,
    write_json,
)
from .phases import phase_boundaries, phase_for_timestep
from .quality import build_common_summary
from .scalers import fit_z_score_scalers
from .splits import (
    build_split_rows,
    collect_split_warnings,
    compute_split_distribution,
    split_mixture_ids,
)

__all__ = [
    "BASELINE_PATH_LMS",
    "FOUR_COMPONENT_CH4_MIN",
    "FOUR_COMPONENT_LABEL_FIELDS",
    "FOUR_COMPONENT_N2_RANGE",
    "MULTI_PATH_PHASE_BASELINE",
    "MULTI_PATH_PHASE_CHOICES",
    "MULTI_PATH_PHASE_OFF",
    "MULTI_PATH_PHASE_STEADY",
    "build_common_summary",
    "build_index_rows",
    "build_label_rows",
    "build_split_rows",
    "build_synthetic_condition_rows",
    "build_synthetic_condition_rows_four_component",
    "collect_split_warnings",
    "compute_split_distribution",
    "fit_z_score_scalers",
    "fmt",
    "normalize_measurement_components",
    "normalize_multi_path_phase",
    "phase_boundaries",
    "phase_for_timestep",
    "sample_four_component_measurement",
    "split_mixture_ids",
    "write_csv",
    "write_json",
]
