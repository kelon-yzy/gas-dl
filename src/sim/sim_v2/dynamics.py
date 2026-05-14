# V2 通道动态：通道目标值、一阶响应、动态参数、值裁剪与格式化。
#
# 这里集中所有"逐通道、逐时间步"的计算原语。matrix_builder 负责调度；
# acoustic_derived 在质量摘要中调用 phase 辅助函数。
# 相位边界 / phase_for_timestep 已迁移到 sim_common.phases，本模块对外保留 re-export。

import math

from scripts.generate_v1_dataset import (
    PROCESSING_PARAMS,
    _fmt,
    _generate_main_features,
)
from sim_common.phases import phase_boundaries, phase_for_timestep

from .constants import (
    NOISE_FRACTION,
    SEQUENCE_CHANNELS,
    TAU_DECAY_SYSTEM_S,
    TAU_RISE_SYSTEM_S,
)


def baseline_targets(condition, rng, l_m=None):
    """计算基线目标值：气室内为 100% N2 时各通道的传感器读数。

    调用 V1 的 _generate_main_features，输入条件为 x_H2=0, x_CH4=0, x_CO2=0, x_N2=100，
    模拟纯背景气体环境下的传感器输出。这是时序的"起点"值。

    Args:
        l_m: 可选 L_m 覆盖。None 时使用 condition['L_m_base']。
             multi-path baseline 模式下不同子段传入不同 L_m。
    """
    baseline_condition = v1_condition(condition, x_h2=0.0, x_ch4=0.0, x_co2=0.0, x_n2=100.0, l_m=l_m)
    main = _generate_main_features(baseline_condition, rng, PROCESSING_PARAMS)
    return channel_targets(condition, main, l_m_override=l_m)


def steady_targets(condition, rng, l_m=None):
    """计算稳态目标值：气室内为目标配气组分时各通道的传感器读数。

    调用 V1 的 _generate_main_features 计算目标配气条件下的传感器输出。
    这是时序在 exposure 阶段结束后趋近的"稳态"值。
    即 V1 特征级仿真中对应组分/工况下的单点输出。

    Args:
        l_m: 可选 L_m 覆盖。None 时使用 condition['L_m_base']。
             multi-path steady 模式下不同子段传入不同 L_m。
    """
    steady_condition = v1_condition(
        condition,
        x_h2=float(condition["x_H2"]),
        x_ch4=float(condition["x_CH4"]),
        x_co2=float(condition["x_CO2"]),
        x_n2=float(condition["x_N2"]),
        l_m=l_m,
    )
    main = _generate_main_features(steady_condition, rng, PROCESSING_PARAMS)
    return channel_targets(condition, main, l_m_override=l_m)


def v1_condition(condition, x_h2, x_ch4, x_co2, x_n2, l_m=None):
    """把 V2 序列条件转为 V1 兼容的条件 dict 格式。

    V1 的条件行键名为 x_H2/x_CH4/...（无 _base 后缀），
    V2 的条件行键名为 x_H2/x_CH4/.../T_C_base/...（有 _base 后缀），
    此函数做转换桥接。

    Args:
        l_m: 可选 L_m 覆盖。None 时使用 condition['L_m_base']。multi-path baseline
             模式下 baseline 段会传入不同子段的 L_m。
    """
    return {
        "x_H2": _fmt(x_h2, 6),
        "x_CH4": _fmt(x_ch4, 6),
        "x_CO2": _fmt(x_co2, 6),
        "x_N2": _fmt(x_n2, 6),
        "T_C": condition["T_C_base"],
        "P_MPa": condition["P_MPa_base"],
        "H_RH": condition["H_RH_base"],
        "L_m": condition["L_m_base"] if l_m is None else _fmt(l_m, 6),
    }


def channel_targets(condition, main, l_m_override=None):
    """从 V1 主特征提取结果 + 环境/机械条件组装为 12 通道目标值字典。

    12 个通道分三类：
    - 传感器信号：V_NDIR_CH4/CO2、V_TCS、TOF、Amp、f_peak、A_fft_max（来自 main）
    - 环境变量：T_C、P_MPa、H_RH（来自 condition）
    - 机械状态：L_m、piston_position_m（来自 condition，piston_position_m 与 L_m 等价）

    Args:
        l_m_override: 可选 L_m 覆盖。None 时使用 condition['L_m_base']。
                      multi-path baseline 模式下不同子段传入不同 L_m。
    """
    t_c = float(condition["T_C_base"])
    p_mpa = float(condition["P_MPa_base"])
    h_rh = float(condition["H_RH_base"])
    l_m = float(condition["L_m_base"]) if l_m_override is None else float(l_m_override)
    return {
        "V_NDIR_CH4": float(main["V_NDIR_CH4"]),
        "V_NDIR_CO2": float(main["V_NDIR_CO2"]),
        "V_TCS": float(main["V_TCS"]),
        "T_C": t_c,
        "P_MPa": p_mpa,
        "H_RH": h_rh,
        "L_m": l_m,
        "piston_position_m": l_m,  # 与 L_m 物理上等价，单独作为独立通道记录
        "TOF": float(main["TOF"]),
        "Amp": float(main["Amp"]),
        "f_peak": float(main["f_peak"]),
        "A_fft_max": float(main["A_fft_max"]),
    }


def channel_dynamic_params(rng):
    """为每个通道随机采样一套独立的动态响应和噪声参数。

    返回的参数字典包含每个通道的：
    - tau_rise_system_s: 一阶系统上升时间常数 (s)
    - tau_decay_system_s: 一阶系统衰减时间常数 (s)
    - noise_sigma: 高斯白噪声标准差
    - random_walk_sigma: random walk 每步的标准差（约为 noise_sigma 的 8%）
    - drift_slope: 线性漂移率（可正可负，约为 noise_sigma * 0.015/步）

    各参数的绝对值与通道信号量级 (channel_noise_scale) 成正比，
    确保噪声的相对强度在不同通道间可比。
    """
    params = {}
    for channel in SEQUENCE_CHANNELS:
        rise_min, rise_max = TAU_RISE_SYSTEM_S[channel]
        decay_min, decay_max = TAU_DECAY_SYSTEM_S[channel]
        base_scale = channel_noise_scale(channel)
        params[channel] = {
            "tau_rise_system_s": rng.uniform(rise_min, rise_max),
            "tau_decay_system_s": rng.uniform(decay_min, decay_max),
            "noise_sigma": base_scale * NOISE_FRACTION[channel],
            "random_walk_sigma": base_scale * NOISE_FRACTION[channel] * 0.08,
            "drift_slope": rng.uniform(-1.0, 1.0) * base_scale * NOISE_FRACTION[channel] * 0.015,
        }
    return params


def channel_noise_scale(channel):
    """返回通道的典型信号量级，用于将噪声相对比例转为绝对幅度。

    例如 V_NDIR_CH4 的典型值为 2.5V，噪声比例为 0.0025，
    所以高斯噪声 sigma = 2.5 * 0.0025 = 0.00625 V。
    """
    return {
        "V_NDIR_CH4": 2.5,       # 典型 NDIR 输出电压 (V)
        "V_NDIR_CO2": 2.5,
        "V_TCS": 1.5,            # 典型 TCS 输出电压 (V)
        "T_C": 30.0,             # 温度跨度 ~30°C
        "P_MPa": 0.7,            # 压力跨度 ~0.7 MPa
        "H_RH": 80.0,            # 湿度跨度 ~80%
        "L_m": 1.8,              # 距离跨度 ~1.8m
        "piston_position_m": 1.8,
        "TOF": 0.006,            # 典型 TOF 值 ~6ms (1.8m / 300m/s)
        "Amp": 1.0,              # 归一化幅度
        "f_peak": 41000.0,       # 中心频率 ~41kHz
        "A_fft_max": 900.0,      # FFT 幅值量级
    }[channel]


def channel_value(baseline, target, timestep, timesteps, tau_rise_system_s, tau_decay_system_s,
                  baseline_at_q1=None):
    """计算通道在指定时间步的一阶系统响应值（不含噪声）。

    四个阶段的数学模型：

    baseline 阶段 (t < q1):
        value = baseline
        （气室内仍为 N2，传感器输出维持基线值。multi-path 模式下 baseline 可能逐子段切换。）

    exposure + steady 阶段 (q1 ≤ t < q3):
        progress = 1 - exp(-(t - q1 + 1) / tau_rise)
        value = start + (target - start) * progress
        （一阶系统上升曲线，从 start 指数逼近 target；start = baseline_at_q1 或 baseline）

    recovery 阶段 (t ≥ q3):
        先计算 q3 时刻的状态值 recovery_start，
        再用一阶衰减曲线从 recovery_start 回退到 baseline：
        recovery_progress = exp(-(t - q3 + 1) / tau_decay)
        value = baseline + (recovery_start - baseline) * recovery_progress
        （recovery 段的目标恢复点仍是 default baseline，与 multi-path 起点无关）

    Args:
        baseline: 基线值（默认 = L_m_base 下的 N2 baseline）。recovery 段恢复目标。
        target: 稳态目标值（目标配气时的传感器输出）。
        timestep: 当前时间步。
        timesteps: 总时间步数。
        tau_rise_system_s: 上升时间常数 (s)。
        tau_decay_system_s: 衰减时间常数 (s)。
        baseline_at_q1: 可选 — exposure 段一阶系统起点。
                        multi-path baseline 模式下传入子段末段的 baseline，
                        让 exposure 一阶系统从 path 末端平滑过渡到 steady。
                        None 时与 baseline 相同（旧行为）。

    Returns:
        该时间步的传感器信号基础值（无噪声）。
    """
    q1, _, q3 = phase_boundaries(timesteps)

    # baseline 阶段：维持基线
    if timestep < q1:
        return baseline

    start = baseline if baseline_at_q1 is None else baseline_at_q1

    # exposure + steady 阶段：一阶上升
    if timestep < q3:
        progress = 1.0 - math.exp(-(timestep - q1 + 1) / tau_rise_system_s)
        return start + (target - start) * progress

    # recovery 阶段：一阶衰减回 default baseline
    start_progress = 1.0 - math.exp(-(q3 - q1) / tau_rise_system_s)
    recovery_start = start + (target - start) * start_progress
    recovery_progress = math.exp(-(timestep - q3 + 1) / tau_decay_system_s)
    return baseline + (recovery_start - baseline) * recovery_progress


def bounded_channel_value(channel, value):
    """对通道值做物理边界裁剪，确保输出在合理范围内。

    - 电压类（V_NDIR、V_TCS）、距离/位移类、声学类：≥ 1e-9（保证正值）
    - 压力：≥ 0.01 MPa
    - 湿度：0-100%
    - 温度：不裁剪（允许超出默认范围，反映传感器故障场景）
    """
    if channel in {"V_NDIR_CH4", "V_NDIR_CO2", "V_TCS", "TOF", "Amp", "A_fft_max", "L_m", "piston_position_m"}:
        return max(1e-9, value)
    if channel == "P_MPa":
        return max(0.01, value)
    if channel == "H_RH":
        return min(100.0, max(0.0, value))
    return value


def format_channel(channel, value):
    """根据通道类型选择合适的小数位数进行格式化输出。

    TOF 需要 8 位小数（纳秒级分辨），f_peak 仅需 3 位等。
    """
    digits = {
        "TOF": 8,
        "f_peak": 3,
        "A_fft_max": 4,
        "T_C": 4,
        "P_MPa": 5,
        "H_RH": 4,
        "L_m": 5,
        "piston_position_m": 5,
    }.get(channel, 6)  # 未指定通道默认 6 位
    return _fmt(value, digits)
