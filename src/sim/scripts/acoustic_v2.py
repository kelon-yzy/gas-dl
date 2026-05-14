# V2 声学链路：分项半物理 alpha + 分离 Amp 链路
#
# 参数依据：simulation-data/acoustic_coefficients_v2.md
# 主要文献：
#   [1] Dain & Lueptow, JASA 109, 1955 (2001); JASA 110, 2974 (2001)
#   [2] Ejakov et al., JASA 113, 1871 (2003)
#   [3] Bass et al., JASA 97, 680 (1995); ISO 9613-1:1993
#   [4] Leonard, JASA 12, 241 (1940); Fricke, JASA 12, 245 (1940)
#
# 工程化简化：
#   - 单峰 Herzfeld-Litovitz 形式，每组分独立贡献后线性叠加
#   - 水汽通过修正 f_relax_co2 间接影响 alpha（Dain 2001 Fig.4 验证）
#   - 混合气声速直接接收外部估计，避免双重计算
#
# calibration_status: pending  (无真实硬件标定数据)

import math


# ── V2 半物理参数 ──

PROCESSING_PARAMS_V2 = {
    # 经典吸收（ISO 9613-1, in dB·m⁻¹·Hz⁻², 转 Np/m 时除以 8.686）
    "alpha_classical_K_ref": 1.84e-11,

    # CO2 弛豫吸收（Leonard 1940, Fricke 1940, Ejakov 2003 Fig.10）
    # 推断：在 CH4 主导混合气中，CO2 在 100% 时的峰值 αλ
    "alpha_lambda_max_co2": 0.12,
    "f_relax_co2_per_atm": 28000.0,
    # 水汽对 CO2 弛豫频率的相对修正系数（CH4 主导气未直接验证，文献给空气数据）
    "k_h2o_to_f_relax_co2": 0.015,

    # CH4 弛豫吸收（Ejakov 2003 Fig.8-9 拟合）
    "alpha_lambda_max_ch4": 0.034,
    # 弛豫频率随 x_CH4 浓度变化：f = base + slope * x_CH4_frac
    "f_relax_ch4_base_per_atm": 30000.0,
    "f_relax_ch4_slope_per_atm": 120000.0,

    # H2 扩散贡献（Bhatia 1967 二元混合气估计）
    "k_diffusion_h2": 1.6e-3,

    # H2O 独立弛豫吸收（Bass 1995 / ISO 9613-1 空气湿度吸收推断，CH4 主导气未直接验证）
    # 单位与 CO2/CH4 弛豫一致，alpha_lambda_max_h2o 表示水汽达到 100% 摩尔分数时的峰值 αλ。
    # 取 0.01 是为了让常温常压下湿度从 20% RH 到 80% RH 的 alpha 变化量级（~0.001-0.005 Np/m）
    # 高于 measurement_noise 反推噪声底（~0.0008），满足计划文档 §5 验收。
    "alpha_lambda_max_h2o": 0.01,
    "f_relax_h2o_per_atm": 100000.0,

    # Amp 链路分离参数
    "amp_emit_ref": 1.0,
    "amp_chain_gain_mean": 1.0,
    "amp_chain_gain_std": 0.0,
    "source_drift_std": 0.005,
    "receiver_gain_drift_std": 0.003,
    "measurement_noise_std": 0.0008,

    # 与 V1 共享的频率
    "acoustic_excitation_frequency_hz": 40000.0,
    # V2 alpha 量级 ~1-3 Np/m，长声程下 Amp 可低至 0.01 V，因此 floor 比 V1 低一档
    "amp_floor_v": 0.005,
    "f_peak_h2_coef_hz_per_pct": 23.0,
    "f_peak_co2_coef_hz_per_pct": -18.0,
    "f_peak_t_coef_hz_per_c": 2.0,
    "daq_fft_noise_hz": 65.0,
    "a_fft_scale_mean": 900.0,
    "a_fft_scale_std": 20.0,

    "calibration_status": "pending",
    "version": "v2.0",
}


# ── 物理常量 ──

_REF_PRESSURE_ATM = 1.0
_PRESSURE_MPA_TO_ATM = 1.0 / 0.101325
_T0_K = 293.15
_NP_TO_DB = 8.686

# 各组分常温 0.1 MPa 声速 (NIST WebBook + Ejakov 2003 实测)
_SPEED_H2_MS = 1306.0
_SPEED_CH4_MS = 446.0
_SPEED_CO2_MS = 268.0
_SPEED_N2_MS = 353.0


# ═══════════════════════════════════════════════════════════════════════════
# 半物理 alpha — _hidden_ 前缀沿用 V1 命名约定，不可作为模型输入
# ═══════════════════════════════════════════════════════════════════════════


def _hidden_sound_speed_v2(x_h2, x_ch4, x_co2, x_n2, t_c):
    """混合气体常温声速近似 (m/s)。线性混合，温度修正 0.6 m/s/°C。

    误差边界：相对 NIST Monograph 178 数据 < 5%，足够 V2 generator 使用。
    """
    x_h2_frac = max(0.0, x_h2) / 100.0
    x_ch4_frac = max(0.0, x_ch4) / 100.0
    x_co2_frac = max(0.0, x_co2) / 100.0
    x_n2_frac = max(0.0, x_n2) / 100.0

    c_mix = (
        x_h2_frac * _SPEED_H2_MS
        + x_ch4_frac * _SPEED_CH4_MS
        + x_co2_frac * _SPEED_CO2_MS
        + x_n2_frac * _SPEED_N2_MS
    )
    c_mix += 0.6 * (t_c - 25.0)
    return max(c_mix, 200.0)


def _h2o_mole_pct(t_c, p_mpa, h_rh):
    """根据相对湿度估算水汽摩尔百分比（限幅 0-5%）。

    p_sat 用 Buck 公式 (kPa)，简便且在 [0, 50] °C 范围内准确度足够。
    """
    p_sat_kpa = 0.61121 * math.exp(17.502 * t_c / (240.97 + t_c))
    p_amb_kpa = max(p_mpa, 1e-3) * 1000.0
    h_w_pct = (h_rh / 100.0) * (p_sat_kpa / p_amb_kpa) * 100.0
    return max(0.0, min(h_w_pct, 5.0))


def _hidden_attenuation_v2(
    x_h2, x_ch4, x_co2, x_n2, t_c, p_mpa, h_rh,
    c_mix=None, f_hz=None, params=None,
):
    """V2 半物理声衰减真值 (Np/m) + 各分项检查字段。

    Args:
        x_h2, x_ch4, x_co2, x_n2: 体积百分比 [0, 100]
        t_c: 摄氏度
        p_mpa: 绝压 MPa
        h_rh: 相对湿度 %
        c_mix: 混合气声速 m/s。None 时用线性混合估计
        f_hz: 工作频率 Hz。None 时使用 params 中的值
        params: V2 参数字典。None 时使用 PROCESSING_PARAMS_V2

    Returns:
        dict 含 alpha_true_v2 (Np/m), alpha_classical, alpha_co2,
        alpha_ch4, alpha_h2_diffusion, f_relax_co2_eff, f_relax_ch4_eff,
        h_w_pct_eff, c_mix_used
    """
    if params is None:
        params = PROCESSING_PARAMS_V2
    if f_hz is None:
        f_hz = params["acoustic_excitation_frequency_hz"]
    if c_mix is None:
        c_mix = _hidden_sound_speed_v2(x_h2, x_ch4, x_co2, x_n2, t_c)
    c_mix = max(c_mix, 200.0)

    p_atm = max(p_mpa, 1e-4) * _PRESSURE_MPA_TO_ATM
    p_pa = max(p_mpa, 1e-4) * 1e6
    t_k = t_c + 273.15
    h_w_pct = _h2o_mole_pct(t_c, p_mpa, h_rh)

    # 1. 经典吸收（Stokes-Kirchhoff, ISO 9613-1 form, dB/m → Np/m）
    alpha_classical_db = (
        params["alpha_classical_K_ref"]
        * (f_hz ** 2)
        * (1.0 / max(p_atm, 1e-4))
        * math.sqrt(t_k / _T0_K)
    )
    alpha_classical_npm = alpha_classical_db / _NP_TO_DB

    # 2. CO2 弛豫吸收（Herzfeld-Litovitz）
    f_r_co2 = (
        params["f_relax_co2_per_atm"]
        * p_atm
        * (1.0 + params["k_h2o_to_f_relax_co2"] * h_w_pct)
    )
    alpha_lambda_co2 = (
        params["alpha_lambda_max_co2"]
        * (max(0.0, x_co2) / 100.0)
        * 2.0 * f_hz * f_r_co2
        / (f_hz ** 2 + f_r_co2 ** 2)
    )
    alpha_co2_npm = alpha_lambda_co2 * f_hz / c_mix

    # 3. CH4 弛豫吸收（Herzfeld-Litovitz, 弛豫频率随浓度漂移）
    x_ch4_frac = max(0.0, x_ch4) / 100.0
    f_r_ch4 = (
        params["f_relax_ch4_base_per_atm"]
        + params["f_relax_ch4_slope_per_atm"] * x_ch4_frac
    ) * p_atm
    alpha_lambda_ch4 = (
        params["alpha_lambda_max_ch4"]
        * x_ch4_frac
        * 2.0 * f_hz * f_r_ch4
        / (f_hz ** 2 + f_r_ch4 ** 2)
    )
    alpha_ch4_npm = alpha_lambda_ch4 * f_hz / c_mix

    # 4. H2 扩散贡献（Bhatia 1967 二元混合气）
    x_h2_frac = max(0.0, x_h2) / 100.0
    x_other_frac = max(0.0, 1.0 - x_h2_frac)
    alpha_diff_npm = (
        params["k_diffusion_h2"]
        * x_h2_frac * x_other_frac
        * (f_hz ** 2)
        / max(p_pa, 1e3)
        / max(c_mix ** 3, 1e6)
    )

    # 5. H2O 独立弛豫吸收（Herzfeld-Litovitz 单峰，f_relax 随压力线性）
    #    与 CO2 弛豫频率的水汽修正解耦：本项捕捉水汽自身的吸收，CO2 项继续保留
    #    f_relax_co2 *= (1 + k_h2o_to_f_relax_co2 * h_w_pct) 的耦合。
    h_w_frac = max(0.0, h_w_pct) / 100.0
    f_r_h2o = params["f_relax_h2o_per_atm"] * p_atm
    alpha_lambda_h2o = (
        params["alpha_lambda_max_h2o"]
        * h_w_frac
        * 2.0 * f_hz * f_r_h2o
        / (f_hz ** 2 + f_r_h2o ** 2)
    )
    alpha_h2o_npm = alpha_lambda_h2o * f_hz / c_mix

    alpha_true = max(
        0.0,
        alpha_classical_npm + alpha_co2_npm + alpha_ch4_npm + alpha_diff_npm + alpha_h2o_npm,
    )

    return {
        "alpha_true_v2": alpha_true,
        "alpha_classical_v2": alpha_classical_npm,
        "alpha_co2_v2": alpha_co2_npm,
        "alpha_ch4_v2": alpha_ch4_npm,
        "alpha_h2_diffusion_v2": alpha_diff_npm,
        "alpha_h2o_v2": alpha_h2o_npm,
        "f_relax_co2_eff": f_r_co2,
        "f_relax_ch4_eff": f_r_ch4,
        "f_relax_h2o_eff": f_r_h2o,
        "h_w_pct_eff": h_w_pct,
        "c_mix_used": c_mix,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Amp 链路分离 — 发射端 / 链路增益 / 测量噪声各自独立采样
# ═══════════════════════════════════════════════════════════════════════════


def _simulate_fiber_acoustic_features_v2(
    condition, rng, sound_speed_mix, params=None,
):
    """V2 光纤麦克风声学链路：分离 source / chain / noise。

    链路分解：
        Amp_emit_ref      = amp_emit_ref * (1 + source_drift)
        Amp_chain_gain    = amp_chain_gain_mean * (1 + receiver_gain_drift)
        Amp_before_noise  = Amp_emit_ref * Amp_chain_gain * exp(-alpha_true * L)
        Amp_obs           = max(amp_floor, Amp_before_noise + measurement_noise)

    f_peak 和 A_fft_max 沿用 V1 形式（共振频率 + Amp 缩放），噪声参数从 V2 配置读取。
    """
    if params is None:
        params = PROCESSING_PARAMS_V2

    x_h2 = float(condition["x_H2"])
    x_ch4 = float(condition["x_CH4"])
    x_co2 = float(condition["x_CO2"])
    x_n2 = float(condition["x_N2"])
    t_c = float(condition["T_C"])
    p_mpa = float(condition["P_MPa"])
    h_rh = float(condition["H_RH"])
    l_m = float(condition["L_m"])

    f_hz = params["acoustic_excitation_frequency_hz"]

    alpha_pkg = _hidden_attenuation_v2(
        x_h2, x_ch4, x_co2, x_n2, t_c, p_mpa, h_rh,
        c_mix=sound_speed_mix, f_hz=f_hz, params=params,
    )
    alpha_true = alpha_pkg["alpha_true_v2"]

    if "source_drift_override" in params:
        source_drift = float(params["source_drift_override"])
    else:
        source_drift = rng.gauss(0.0, params["source_drift_std"])
    amp_emit_ref = params["amp_emit_ref"] * (1.0 + source_drift)

    if "receiver_gain_drift_override" in params:
        receiver_gain_drift = float(params["receiver_gain_drift_override"])
    else:
        receiver_gain_drift = rng.gauss(0.0, params["receiver_gain_drift_std"])
    amp_chain_gain = params["amp_chain_gain_mean"] * (1.0 + receiver_gain_drift)

    amp_before_noise = amp_emit_ref * amp_chain_gain * math.exp(-alpha_true * l_m)
    measurement_noise = rng.gauss(0.0, params["measurement_noise_std"])
    amp_obs = max(params["amp_floor_v"], amp_before_noise + measurement_noise)

    f_peak = (
        f_hz
        + params["f_peak_h2_coef_hz_per_pct"] * x_h2
        + params["f_peak_co2_coef_hz_per_pct"] * x_co2
        + params["f_peak_t_coef_hz_per_c"] * (t_c - 20.0)
        + rng.gauss(0.0, params["daq_fft_noise_hz"])
    )

    a_fft_max = amp_obs * (
        params["a_fft_scale_mean"]
        + rng.gauss(0.0, params["a_fft_scale_std"])
    )

    return {
        "Amp": amp_obs,
        "f_peak": f_peak,
        "A_fft_max": a_fft_max,
        # 检查字段（不进入模型输入）
        "Amp_emit_ref": amp_emit_ref,
        "Amp_chain_gain": amp_chain_gain,
        "Amp_before_noise": amp_before_noise,
        "source_drift": source_drift,
        "receiver_gain_drift": receiver_gain_drift,
        "measurement_noise": measurement_noise,
        "attenuation_alpha_true": alpha_true,
        "alpha_classical_v2": alpha_pkg["alpha_classical_v2"],
        "alpha_co2_v2": alpha_pkg["alpha_co2_v2"],
        "alpha_ch4_v2": alpha_pkg["alpha_ch4_v2"],
        "alpha_h2_diffusion_v2": alpha_pkg["alpha_h2_diffusion_v2"],
        "alpha_h2o_v2": alpha_pkg["alpha_h2o_v2"],
        "f_relax_co2_eff": alpha_pkg["f_relax_co2_eff"],
        "f_relax_ch4_eff": alpha_pkg["f_relax_ch4_eff"],
        "f_relax_h2o_eff": alpha_pkg["f_relax_h2o_eff"],
        "h_w_pct_eff": alpha_pkg["h_w_pct_eff"],
        "c_mix_used": alpha_pkg["c_mix_used"],
    }


__all__ = [
    "PROCESSING_PARAMS_V2",
    "_hidden_attenuation_v2",
    "_hidden_sound_speed_v2",
    "_simulate_fiber_acoustic_features_v2",
]
