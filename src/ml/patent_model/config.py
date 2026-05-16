"""项目级常量定义。"""

from __future__ import annotations

# 默认仍使用三组分目标；四组分模式会显式切到 *_FOUR 常量。
COMPONENT_NAMES: tuple[str, ...] = ("H2", "CH4", "CO2")
TARGET_COLUMNS: tuple[str, ...] = ("x_H2", "x_CH4", "x_CO2")
FOUR_COMPONENT_NAMES: tuple[str, ...] = ("H2", "CH4", "CO2", "N2")
FOUR_TARGET_COLUMNS: tuple[str, ...] = ("x_H2", "x_CH4", "x_CO2", "x_N2")

# 三条模态分支名称，后续建模和输出都以这里为准。
BRANCH_NAMES: tuple[str, ...] = ("acoustic", "optical", "thermal")

# 三个分支分别使用的原始特征列。
ACOUSTIC_FEATURE_COLUMNS: tuple[str, ...] = ("TOF", "Amp", "f_peak", "A_fft_max", "L_m")
OPTICAL_FEATURE_COLUMNS: tuple[str, ...] = ("V_NDIR_CH4", "V_NDIR_CO2", "delta_I_CH4", "delta_I_CO2")
THERMAL_FEATURE_COLUMNS: tuple[str, ...] = ("V_TCS",)
ENVIRONMENT_FEATURE_COLUMNS: tuple[str, ...] = ("T_C", "P_MPa", "H_RH")

# 统一列出评估时会输出的模型家族名称。
MODEL_NAMES: tuple[str, ...] = (
    "acoustic",
    "optical",
    "thermal",
    "fused",
)

MODEL_DISPLAY_NAMES_ZH: dict[str, str] = {
    "acoustic": "声学单模态",
    "optical": "光学单模态",
    "thermal": "热导单模态",
    "fused": "动态融合输出",
}
