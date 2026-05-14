# V2 (N, T, C) 时序矩阵装配。
#
# 调度逻辑：
#   - 三相位 baseline / exposure+steady / recovery 由 dynamics.channel_value 算单通道
#   - multi-path baseline / steady 模式按子段切换 L_m，acoustic 通道走 path-aware 路径
#   - 噪声/漂移/random walk 在每步叠加，path 内常数段跳过（避免破坏拟合）

import numpy as np

from scripts.generate_v1_dataset import _fmt

from .constants import (
    ACOUSTIC_PATH_RELATED_CHANNELS,
    BASELINE_PATH_LMS,
    MULTI_PATH_PHASE_BASELINE,
    MULTI_PATH_PHASE_OFF,
    MULTI_PATH_PHASE_STEADY,
    SEQUENCE_CHANNELS,
    normalize_multi_path_phase,
)
from .dynamics import (
    baseline_targets,
    bounded_channel_value,
    channel_dynamic_params,
    channel_value,
    format_channel,
    phase_boundaries,
    phase_for_timestep,
    steady_targets,
)


def baseline_subsegment_index(timestep, baseline_end, num_paths):
    """baseline 段内 timestep 对应的子段索引。子段长度 = baseline_end / num_paths（向上取整）。

    Args:
        timestep: 当前 timestep (0-based)
        baseline_end: baseline 段结束 timestep（不含）
        num_paths: 子段数量

    Returns:
        int: [0, num_paths-1]
    """
    sub_size = max(1, baseline_end // num_paths)
    return min(num_paths - 1, timestep // sub_size)


def steady_subsegment_index(timestep, steady_start, steady_end, num_paths):
    """steady 段内 timestep 对应的子段索引。

    Args:
        timestep: 当前 timestep (0-based, 全局)
        steady_start: steady 段起始 timestep（含）
        steady_end:   steady 段结束 timestep（不含）
        num_paths:    子段数量

    Returns:
        int: [0, num_paths-1]
    """
    local = timestep - steady_start
    span = max(1, steady_end - steady_start)
    sub_size = max(1, span // num_paths)
    return min(num_paths - 1, local // sub_size)


def build_sequence_matrix(conditions, timesteps, dt_s, rng,
                          multi_path_phase=MULTI_PATH_PHASE_OFF):
    """为每条序列构建 (T, C) 二维时序矩阵并展开为长表行列表。

    对每条序列的处理流程：
    1. 计算 baseline 目标值（纯 N2 环境下的传感器输出）
    2. 计算 steady 目标值（目标配气下的传感器输出）
       - multi_path_phase='steady' 时预算 N 套（每套对应一个 L_m）
       - multi_path_phase='baseline' 时 baseline 端预算 N 套
    3. 为每个通道采样独立的动态参数（tau_rise, tau_decay, noise sigma 等）
    4. 逐时间步计算每个通道的一阶系统响应值 + 叠加噪声/漂移/random walk

    Args:
        conditions: 序列条件行列表。
        timesteps: 时间步数。
        dt_s: 时间步长。
        rng: 随机数生成器。
        multi_path_phase: 多声程扫描相位。
            - 'off':      不启用
            - 'baseline': baseline 段（纯 N2）切换 L_m，alpha_fit 拟合 N2 衰减
            - 'steady':   steady 段（混合气）切换 L_m，alpha_fit 拟合混合气衰减（推荐）

    Returns:
        tuple:
            - x_matrix: float32 ndarray, shape = (N, T, C)
            - long_rows: 长表 CSV 行列表 (N*T 个 dict)
    """
    multi_path_phase = normalize_multi_path_phase(multi_path_phase)
    x_matrix = np.zeros((len(conditions), timesteps, len(SEQUENCE_CHANNELS)), dtype=np.float32)
    long_rows = []
    q1, q2, q3 = phase_boundaries(timesteps)
    num_paths = len(BASELINE_PATH_LMS)
    is_baseline_scan = multi_path_phase == MULTI_PATH_PHASE_BASELINE
    is_steady_scan = multi_path_phase == MULTI_PATH_PHASE_STEADY

    for sequence_index, condition in enumerate(conditions):
        # 计算两个"端点"的传感器输出目标值（默认 L_m_base）
        baseline_targets_default = baseline_targets(condition, rng)
        steady_targets_default = steady_targets(condition, rng)

        # baseline 多声程：预算 N 套不同 L_m 下的 baseline targets
        baseline_targets_per_path = None
        if is_baseline_scan:
            baseline_targets_per_path = [
                baseline_targets(condition, rng, l_m=path_lm)
                for path_lm in BASELINE_PATH_LMS
            ]

        # steady 多声程：预算 N 套不同 L_m 下的 steady targets（混合气）
        steady_targets_per_path = None
        if is_steady_scan:
            steady_targets_per_path = [
                steady_targets(condition, rng, l_m=path_lm)
                for path_lm in BASELINE_PATH_LMS
            ]

        # 随机采样各通道的动态响应参数
        channel_params = channel_dynamic_params(rng)
        # random_walk 状态变量（每个通道独立累加，模拟低频漂移）
        random_walk = {channel: 0.0 for channel in SEQUENCE_CHANNELS}

        for timestep in range(timesteps):
            phase_id = phase_for_timestep(timestep, timesteps)
            row = {
                "sequence_id": condition["sequence_id"],
                "timestep": str(timestep),
                "timestamp_s": _fmt(timestep * dt_s, 4),
                "phase_id": phase_id,
            }

            # 在 baseline 段或 steady 段确定当前子段索引
            baseline_sub_index = None
            steady_sub_index = None
            if is_baseline_scan and timestep < q1:
                baseline_sub_index = baseline_subsegment_index(timestep, q1, num_paths)
            if is_steady_scan and q2 <= timestep < q3:
                steady_sub_index = steady_subsegment_index(timestep, q2, q3, num_paths)

            for channel_index, channel in enumerate(SEQUENCE_CHANNELS):
                is_acoustic_path_channel = channel in ACOUSTIC_PATH_RELATED_CHANNELS

                # 决定该 channel 在该 timestep 的 baseline 起点
                if (baseline_sub_index is not None
                        and is_acoustic_path_channel):
                    chan_baseline = baseline_targets_per_path[baseline_sub_index][channel]
                else:
                    chan_baseline = baseline_targets_default[channel]

                # exposure 段一阶系统起点：baseline-multi-path 时取 path 末段，让响应连续
                baseline_at_q1 = None
                if (is_baseline_scan and is_acoustic_path_channel):
                    baseline_at_q1 = baseline_targets_per_path[-1][channel]

                # 一阶系统响应值（exposure/recovery 等其他相位的过渡）
                value = channel_value(
                    baseline=chan_baseline,
                    target=steady_targets_default[channel],
                    timestep=timestep,
                    timesteps=timesteps,
                    tau_rise_system_s=channel_params[channel]["tau_rise_system_s"],
                    tau_decay_system_s=channel_params[channel]["tau_decay_system_s"],
                    baseline_at_q1=baseline_at_q1,
                )

                # steady 多声程：在 steady 段，acoustic 通道值 snap 到该 path 的 steady target
                # （声波/几何变化无惯性，活塞瞬时切换 L_m 即得新 Amp/TOF）
                if steady_sub_index is not None and is_acoustic_path_channel:
                    value = steady_targets_per_path[steady_sub_index][channel]

                # 噪声/漂移/random walk:
                # - baseline-scan 模式：baseline 段声学通道跳过（保持 path 内常数）
                # - steady-scan 模式：steady 段声学通道仍叠加（保留观测噪声）
                skip_noise = (is_baseline_scan
                              and timestep < q1
                              and is_acoustic_path_channel)
                if not skip_noise:
                    random_walk[channel] += rng.gauss(0.0, channel_params[channel]["random_walk_sigma"])
                    value += channel_params[channel]["drift_slope"] * timestep   # 线性漂移
                    value += random_walk[channel]                               # random walk
                    value += rng.gauss(0.0, channel_params[channel]["noise_sigma"])  # 高斯白噪声
                # 物理边界裁剪（如电压/光强不能为负）
                value = bounded_channel_value(channel, value)

                x_matrix[sequence_index, timestep, channel_index] = value
                row[channel] = format_channel(channel, value)

            long_rows.append(row)

    return x_matrix, long_rows
