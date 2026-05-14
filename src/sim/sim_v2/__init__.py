# V2 sequence dataset generator package.
#
# 模块职责：
#   - constants: 数据集常量、字段、TAU/NOISE 表、模态分组、多路径选项
#   - io:        路径布局、CSV/JSON 写入、index/label 行构造
#   - splits:    按 mixture_id 分组划分 + split 分布与告警
#   - conditions: 三组分/四组分条件采样
#   - dynamics:  通道目标值、一阶响应、动态参数、相位边界、值裁剪
#   - matrix_builder: (N, T, C) 矩阵装配 + 多路径子段调度
#   - acoustic_derived: 声学派生侧车 + 声学质量摘要
#   - scalers:   train split z-score 拟合
#   - quality:   质量摘要 JSON
#   - readme:    数据包 README
#   - orchestrator: generate_sequence_dataset 主入口与四组分入口
#
# 入口：generate_sequence_dataset / generate_sequence_dataset_four_component

from .constants import (
    ACOUSTIC_DERIVED_SEQUENCE_FIELDS,
    ACOUSTIC_PATH_RELATED_CHANNELS,
    ALPHA_FIT_R2_LOW_CONFIDENCE_THRESHOLD,
    BASELINE_PATH_LMS,
    CONDITION_FIELDS,
    DEFAULT_DT_S,
    DEFAULT_TIMESTEPS,
    FOUR_COMPONENT_CH4_MIN,
    FOUR_COMPONENT_DEFAULT_OUTPUT_DIR,
    FOUR_COMPONENT_LABEL_FIELDS,
    FOUR_COMPONENT_N2_RANGE,
    LABEL_FIELDS,
    MODAL_GROUPS,
    MODAL_SEQUENCE_FIELDS,
    MULTI_PATH_PHASE_BASELINE,
    MULTI_PATH_PHASE_CHOICES,
    MULTI_PATH_PHASE_OFF,
    MULTI_PATH_PHASE_STEADY,
    NOISE_FRACTION,
    SEQUENCE_CHANNELS,
    SEQUENCE_INDEX_FIELDS,
    SEQUENCE_LABEL_FIELDS,
    SPLIT_FIELDS,
    TAU_DECAY_SYSTEM_S,
    TAU_RISE_SYSTEM_S,
    normalize_multi_path_phase,
)
from .orchestrator import (
    generate_sequence_dataset,
    generate_sequence_dataset_four_component,
)

__all__ = [
    "generate_sequence_dataset",
    "generate_sequence_dataset_four_component",
    "ACOUSTIC_DERIVED_SEQUENCE_FIELDS",
    "ACOUSTIC_PATH_RELATED_CHANNELS",
    "ALPHA_FIT_R2_LOW_CONFIDENCE_THRESHOLD",
    "BASELINE_PATH_LMS",
    "CONDITION_FIELDS",
    "DEFAULT_DT_S",
    "DEFAULT_TIMESTEPS",
    "FOUR_COMPONENT_CH4_MIN",
    "FOUR_COMPONENT_DEFAULT_OUTPUT_DIR",
    "FOUR_COMPONENT_LABEL_FIELDS",
    "FOUR_COMPONENT_N2_RANGE",
    "LABEL_FIELDS",
    "MODAL_GROUPS",
    "MODAL_SEQUENCE_FIELDS",
    "MULTI_PATH_PHASE_BASELINE",
    "MULTI_PATH_PHASE_CHOICES",
    "MULTI_PATH_PHASE_OFF",
    "MULTI_PATH_PHASE_STEADY",
    "NOISE_FRACTION",
    "SEQUENCE_CHANNELS",
    "SEQUENCE_INDEX_FIELDS",
    "SEQUENCE_LABEL_FIELDS",
    "SPLIT_FIELDS",
    "TAU_DECAY_SYSTEM_S",
    "TAU_RISE_SYSTEM_S",
    "normalize_multi_path_phase",
]
