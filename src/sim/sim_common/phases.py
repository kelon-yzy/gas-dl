# 时序四阶段相位边界与归属。
#
# V2 与 V3 共享相同的 baseline / exposure / steady / recovery 四阶段切分。
# 边界比例固定 1/6 / 7/12 / 5/6（对应 N=120 → 20 / 70 / 100），与
# sequence-models/data/channel_groups.py 中 TIME_WINDOWS 严格对齐。
# 需要变更时同步动 V2/V3 测试和 channel_groups。


def phase_boundaries(timesteps):
    """计算四个时序阶段的边界时间步索引。

    - q1: baseline -> exposure 切换点（1/6 处，N=120 → 20）
    - q2: exposure -> steady 切换点（7/12 处，N=120 → 70）
    - q3: steady -> recovery 切换点（5/6 处，N=120 → 100）

    使用整数除法保证 N=120 时严格命中 (20, 70, 100)，与
    sequence-models/data/channel_groups.py:TIME_WINDOWS 一致。
    每个边界至少保留 1 个时间步以包含对应阶段。
    """
    q1 = max(1, timesteps // 6)
    q2 = max(q1 + 1, 7 * timesteps // 12)
    q3 = max(q2 + 1, 5 * timesteps // 6)
    return q1, q2, q3


def phase_for_timestep(timestep, timesteps):
    """判断给定时间步属于哪个时序阶段。

    四阶段划分（N=120 时严格对齐 channel_groups.TIME_WINDOWS）：
    - baseline:  [0, 1/6)     N=120 → [0, 20)，气室初始纯 N2 状态
    - exposure:  [1/6, 7/12)  N=120 → [20, 70)，通入目标气体，信号快速上升
    - steady:    [7/12, 5/6)  N=120 → [70, 100)，接近稳态，动态权重训练关键段
    - recovery:  [5/6, 1]     N=120 → [100, 120)，切回 N2 吹扫，信号衰减
    """
    q1, q2, q3 = phase_boundaries(timesteps)
    if timestep < q1:
        return "baseline"
    if timestep < q2:
        return "exposure"
    if timestep < q3:
        return "steady"
    return "recovery"
