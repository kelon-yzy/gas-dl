# V2 / V3 共享常量与多路径相位归一化。
#
# 这些常量物理上是仿真链路无关的"配置项"：标签字段顺序、多路径扫描相位选项、
# 多 L_m 子段距离表。V2 / V3 共用以保证两边派生数据格式与字段顺序一致。

# 四组分标签字段顺序，V2/V3 都按这个顺序装配 NPZ y 张量与 condition CSV。
FOUR_COMPONENT_LABEL_FIELDS = ["x_H2", "x_CH4", "x_CO2", "x_N2"]

# 四组分采样配置：N2 占比范围（含两端）与 CH4 主体最低占比。
FOUR_COMPONENT_N2_RANGE = (0.0, 20.0)
FOUR_COMPONENT_CH4_MIN = 40.0

# 多 L_m baseline 扫描的子段距离表（米）。
# 与 V1 distance_stage 的 5 段一致，方便 V2/V3 对照拟合 ln(Amp) vs L_m。
BASELINE_PATH_LMS = (0.2, 0.6, 1.0, 1.4, 1.8)

# multi_path 扫描相位选项。V2 主管线、V3 派生（如未来需要）共享语义。
# - "off"      不启用多声程扫描
# - "baseline" baseline 段（纯 N2）切换 L_m，alpha_fit 反映 N2 链路衰减
# - "steady"   steady 段（混合气）切换 L_m，alpha_fit 反映目标混合气衰减（推荐）
MULTI_PATH_PHASE_OFF = "off"
MULTI_PATH_PHASE_BASELINE = "baseline"
MULTI_PATH_PHASE_STEADY = "steady"
MULTI_PATH_PHASE_CHOICES = (MULTI_PATH_PHASE_OFF, MULTI_PATH_PHASE_BASELINE, MULTI_PATH_PHASE_STEADY)


def normalize_multi_path_phase(value):
    """规范化 multi_path_phase 输入。

    None 视为 'off'。其他值必须是 MULTI_PATH_PHASE_CHOICES 之一。
    旧版 bool 兼容已移除（D4），调用方必须传入字符串或 None。
    """
    if value is None:
        return MULTI_PATH_PHASE_OFF
    if value in MULTI_PATH_PHASE_CHOICES:
        return value
    raise ValueError(
        f"multi_path_phase must be one of {MULTI_PATH_PHASE_CHOICES} or None, got {value!r}"
    )
