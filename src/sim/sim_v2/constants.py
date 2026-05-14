# V2 时序仿真数据集常量。
#
# V2 12 通道专属字段 / 物理表保留在此处；与 V3 共享的多路径相位、
# 四组分标签、距离表统一从 sim_common 取用。

from sim_common.constants import (
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

# ── 数据集默认参数 ────────────────────────────────────────────────────
DEFAULT_TIMESTEPS = 120            # 每条序列的时间步数（120 步 × 1s 步长 = 2 分钟）
DEFAULT_DT_S = 1.0                 # 时间步长 (s)

# ── 时序通道列表（12 个通道）───────────────────────────────────────────
# 通道排列顺序即 NPZ 中 channel 轴的索引顺序，后续所有模块须与此对齐
SEQUENCE_CHANNELS = [
    "V_NDIR_CH4",       # 0: NDIR 甲烷通道电压 (V)
    "V_NDIR_CO2",       # 1: NDIR 二氧化碳通道电压 (V)
    "V_TCS",            # 2: 热导传感器电压 (V)
    "T_C",              # 3: 环境温度 (°C)
    "P_MPa",            # 4: 环境绝对压力 (MPa)
    "H_RH",             # 5: 环境相对湿度 (%)
    "L_m",              # 6: 声程/光程长度 (m)
    "piston_position_m",# 7: 活塞/位移传感器位置 (m)，与 L_m 物理上等价
    "TOF",              # 8: 超声飞行时间 (s)
    "Amp",              # 9: 光纤麦克风信号幅度 (V)
    "f_peak",           #10: 声学共振峰频率 (Hz)
    "A_fft_max",        #11: FFT 最大幅值 (a.u.)
]

# 三组分 V2 标签（V2 主管线限定）
LABEL_FIELDS = ["x_H2", "x_CH4", "x_CO2"]

# ── 文件输出字段定义 ──────────────────────────────────────────────────

# 时序条件表：每条序列一行，记录配气组分和基准工况（base 值 = 稳态目标值）
CONDITION_FIELDS = [
    "sequence_id",
    "mixture_id",
    "x_H2",
    "x_CH4",
    "x_CO2",
    "x_N2",
    "T_C_base",
    "P_MPa_base",
    "H_RH_base",
    "L_m_base",
    "status",
]

# 序列索引表：用于快速检索序列的基本属性
SEQUENCE_INDEX_FIELDS = [
    "sequence_id",
    "mixture_id",
    "stage_profile",
    "status",
    "n_timesteps",
    "dt_s",
]

# 序列长表：将每条序列展开为 T 行（每行一个时间步），包含完整的 12 通道值 + 相位标识
MODAL_SEQUENCE_FIELDS = ["sequence_id", "timestep", "timestamp_s", "phase_id"] + SEQUENCE_CHANNELS
ACOUSTIC_DERIVED_SEQUENCE_FIELDS = [
    "sequence_id",
    "timestep",
    "timestamp_s",
    "phase_id",
    "Amp_ref_baseline",
    "Amp_ref_calibrated",
    "Amp_ref_fit",
    "attenuation_alpha_n2_baseline",
    "attenuation_alpha_rel",
    "attenuation_alpha_rel_clip",
    "attenuation_alpha_calibrated",
    "attenuation_alpha_calibrated_clip",
    "attenuation_alpha_fit",
    "fit_r2",
    "fit_rmse_log_amp",
    "fit_num_paths",
    "fit_L_m_range",
    "fit_status",
]

SEQUENCE_LABEL_FIELDS = ["sequence_id"] + LABEL_FIELDS
SPLIT_FIELDS = ["sequence_id", "mixture_id"]

FOUR_COMPONENT_DEFAULT_OUTPUT_DIR = "output_sequence_n2"

# ── 一阶系统等效响应时间常数（秒）───────────────────────────────────────
#
# 每个通道的上升/衰减时间常数以 (min, max) 元组表示，
# 每个序列生成时在 [min, max] 内随机采样一个值，模拟不同气路/传感器的个性差异。
#
# τ 的物理意义：一阶系统从初始值到稳态值，经过一个 τ 后完成约 63.2% 的过渡。
# 经过 3τ 后完成约 95%，经过 5τ 后完成约 99.3%。

# 上升时间常数（baseline -> exposure -> steady 阶段的信号增长）
TAU_RISE_SYSTEM_S = {
    # NDIR 光学通道：光电探测器响应快，但气室混合需要时间
    "V_NDIR_CH4": (8.0, 20.0),
    "V_NDIR_CO2": (6.0, 18.0),
    # TCS 热导通道：热传导有时间滞后，响应最慢
    "V_TCS": (10.0, 35.0),
    # 环境通道：温度/压力/湿度传感器通常有热惯性或机械惯性
    "T_C": (20.0, 80.0),
    "P_MPa": (20.0, 80.0),
    "H_RH": (20.0, 80.0),
    "L_m": (20.0, 80.0),
    "piston_position_m": (20.0, 80.0),
    # 声学通道：超声波传播无惯性，响应最快
    "TOF": (2.0, 8.0),
    "Amp": (3.0, 10.0),
    "f_peak": (2.0, 8.0),
    "A_fft_max": (3.0, 10.0),
}

# 衰减时间常数（steady -> recovery 阶段的信号衰减）
# 通常衰减比上升慢（传感器在"回零"过程中存在残余效应）
TAU_DECAY_SYSTEM_S = {
    "V_NDIR_CH4": (12.0, 30.0),
    "V_NDIR_CO2": (10.0, 28.0),
    "V_TCS": (20.0, 60.0),          # TCS 降温慢
    "T_C": (20.0, 80.0),
    "P_MPa": (20.0, 80.0),
    "H_RH": (20.0, 80.0),
    "L_m": (20.0, 80.0),
    "piston_position_m": (20.0, 80.0),
    "TOF": (5.0, 15.0),
    "Amp": (8.0, 20.0),
    "f_peak": (5.0, 15.0),
    "A_fft_max": (8.0, 20.0),
}

# ── 通道级噪声参数 ────────────────────────────────────────────────────
#
# NOISE_FRACTION: 高斯噪声标准差相对通道信号量级的比例
# 值越大，通道的信噪比越低

NOISE_FRACTION = {
    "V_NDIR_CH4": 0.0025,   # NDIR 噪声较小（光电检测信噪比高）
    "V_NDIR_CO2": 0.0025,
    "V_TCS": 0.003,         # 热导噪声略大
    "T_C": 0.001,           # 温度测量精度高（Pt100/热电偶级）
    "P_MPa": 0.001,
    "H_RH": 0.0015,
    "L_m": 0.0008,
    "piston_position_m": 0.0008,
    "TOF": 0.0015,          # 高精度计时，噪声极低
    "Amp": 0.003,
    "f_peak": 0.001,
    "A_fft_max": 0.003,
}

# ── 模态分组（按物理测量原理分类）───────────────────────────────────────
# 用于按模态独立标准化和后续模型的特征分组输入
MODAL_GROUPS = {
    "optical": ["V_NDIR_CH4", "V_NDIR_CO2"],            # NDIR 双通道光学
    "thermal": ["V_TCS"],                                 # TCS 单通道热导
    "environment": ["T_C", "P_MPa", "H_RH", "L_m", "piston_position_m"],  # 环境/机械状态
    "acoustic": ["TOF", "Amp", "f_peak", "A_fft_max"],  # 四维声学特征
}

# 仅这些通道会随子段切换 L_m 重新计算 baseline / steady target。
# 其他通道（NDIR/TCS/T_C/P_MPa/H_RH）与 L_m 无关，保持不变。
ACOUSTIC_PATH_RELATED_CHANNELS = frozenset({
    "TOF", "Amp", "f_peak", "A_fft_max", "L_m", "piston_position_m",
})

# fit_r2 低于该阈值的拟合结果在 fit_status 中标记为 low_confidence，下游模型应过滤后再使用。
ALPHA_FIT_R2_LOW_CONFIDENCE_THRESHOLD = 0.7
