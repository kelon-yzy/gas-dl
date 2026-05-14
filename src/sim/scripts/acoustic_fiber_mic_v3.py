from __future__ import annotations

import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.acoustic_v2 import _hidden_attenuation_v2, _hidden_sound_speed_v2
from scripts.acoustic_waveform_v3 import ADC_MAX_INT16, BURST_CYCLES, CENTER_FREQUENCY_HZ, SAMPLE_RATE_HZ, generate_burst_pulse


MEASUREMENT_WINDOW_S = 0.010
WAVEFORM_SAMPLES = int(round(SAMPLE_RATE_HZ * MEASUREMENT_WINDOW_S))
DEFAULT_NOISE_STD_V = 1e-3
DEFAULT_L_DIRECT_FACTOR = 0.5
DEFAULT_WALL_REFLECTION_COEF = 0.5
DEFAULT_MAX_REFLECTIONS = 15
CALIBRATION_STATUS = "pending"


@dataclass(frozen=True)
class FiberMicV3Spec:
    sample_rate_hz: int = SAMPLE_RATE_HZ
    center_frequency_hz: float = CENTER_FREQUENCY_HZ
    burst_cycles: int = BURST_CYCLES
    measurement_window_s: float = MEASUREMENT_WINDOW_S
    adc_max_int16: int = ADC_MAX_INT16
    noise_std_v: float = DEFAULT_NOISE_STD_V
    l_direct_factor: float = DEFAULT_L_DIRECT_FACTOR
    wall_reflection_coef: float = DEFAULT_WALL_REFLECTION_COEF
    max_reflections: int = DEFAULT_MAX_REFLECTIONS
    calibration_status: str = CALIBRATION_STATUS

    @property
    def waveform_samples(self) -> int:
        return int(round(self.sample_rate_hz * self.measurement_window_s))

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["waveform_samples"] = self.waveform_samples
        return payload


def _add_pulse(buffer: np.ndarray, pulse: np.ndarray, start: int, amplitude: float) -> None:
    if start < 0 or start >= buffer.shape[0]:
        return
    usable = min(pulse.shape[0], buffer.shape[0] - start)
    if usable <= 0:
        return
    buffer[start : start + usable] += pulse[:usable] * amplitude


def simulate_fiber_mic_measurement(
    x_h2,
    x_ch4,
    x_co2,
    x_n2,
    t_c,
    p_mpa,
    h_rh,
    l_m,
    seed,
    noise_std_v=DEFAULT_NOISE_STD_V,
    sample_rate_hz=SAMPLE_RATE_HZ,
    center_frequency_hz=CENTER_FREQUENCY_HZ,
    measurement_window_s=MEASUREMENT_WINDOW_S,
    spec: FiberMicV3Spec | None = None,
):
    if spec is None:
        spec = FiberMicV3Spec(
            sample_rate_hz=sample_rate_hz,
            center_frequency_hz=center_frequency_hz,
            burst_cycles=BURST_CYCLES,
            measurement_window_s=measurement_window_s,
            noise_std_v=noise_std_v,
        )
    if l_m <= 0.0:
        raise ValueError("l_m must be > 0")

    rng = random.Random(seed)
    c_sound = _hidden_sound_speed_v2(x_h2, x_ch4, x_co2, x_n2, t_c)
    if c_sound <= 0.0:
        raise ValueError(f"sound speed must be > 0, got {c_sound}")
    attenuation = _hidden_attenuation_v2(
        x_h2, x_ch4, x_co2, x_n2, t_c, p_mpa, h_rh, c_mix=c_sound, f_hz=spec.center_frequency_hz
    )
    alpha_true_npm = float(attenuation["alpha_true_v2"])

    l_direct = float(l_m) * float(spec.l_direct_factor)
    tof_direct_s = l_direct / c_sound
    t_round_s = (2.0 * float(l_m)) / c_sound
    tau_s = 1.0 / max(alpha_true_npm * c_sound, 1e-9)

    pulse = generate_burst_pulse(
        center_frequency_hz=spec.center_frequency_hz,
        burst_cycles=spec.burst_cycles,
        sample_rate_hz=spec.sample_rate_hz,
        amplitude_v=1.0,
    )
    waveform = np.zeros(spec.waveform_samples, dtype=np.float32)

    direct_index = int(round(tof_direct_s * spec.sample_rate_hz))
    direct_amp = math.exp(-alpha_true_npm * l_direct)
    _add_pulse(waveform, pulse, direct_index, direct_amp)

    reflection_count = 0
    reflection_amplitudes: list[float] = []
    reflection_indices: list[int] = []
    threshold = 3.0 * float(spec.noise_std_v)
    for reflection_idx in range(1, int(spec.max_reflections) + 1):
        path_length = l_direct + (2.0 * reflection_idx * float(l_m))
        amplitude = (float(spec.wall_reflection_coef) ** reflection_idx) * math.exp(-alpha_true_npm * path_length)
        if reflection_idx > 1 and amplitude < threshold:
            break
        start = int(round((tof_direct_s + reflection_idx * t_round_s) * spec.sample_rate_hz))
        if start >= waveform.shape[0]:
            break
        _add_pulse(waveform, pulse, start, amplitude)
        reflection_count += 1
        reflection_amplitudes.append(float(amplitude))
        reflection_indices.append(int(start))

    if spec.noise_std_v > 0.0:
        noise_rng = np.random.default_rng(rng.randrange(0, 2**32))
        waveform = waveform + noise_rng.normal(0.0, spec.noise_std_v, size=waveform.shape).astype(np.float32)

    peak_abs_v = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    if peak_abs_v <= 0.0:
        raise ValueError("peak_abs_v must be > 0")
    scale_factor = peak_abs_v / spec.adc_max_int16
    waveform_int16 = np.clip(np.round(waveform / scale_factor), -spec.adc_max_int16, spec.adc_max_int16).astype(np.int16)
    return {
        "waveform_float": waveform.astype(np.float32),
        "waveform_int16": waveform_int16,
        "scale_factor": float(scale_factor),
        "tof_direct_s": float(tof_direct_s),
        "t_round_s": float(t_round_s),
        "tau_s": float(tau_s),
        "direct_index": int(direct_index),
        "direct_amp": float(direct_amp),
        "reflection_count": int(reflection_count),
        "reflection_amplitudes": reflection_amplitudes,
        "reflection_indices": reflection_indices,
        "peak_abs_v": float(peak_abs_v),
        "alpha_true_npm": float(alpha_true_npm),
        "sound_speed_m_per_s": float(c_sound),
        "sample_rate_hz": int(spec.sample_rate_hz),
        "center_frequency_hz": float(spec.center_frequency_hz),
        "measurement_window_s": float(spec.measurement_window_s),
        "l_direct_factor": float(spec.l_direct_factor),
        "wall_reflection_coef": float(spec.wall_reflection_coef),
        "max_reflections": int(spec.max_reflections),
        "calibration_status": spec.calibration_status,
        "spec": spec.to_dict(),
    }


__all__ = [
    "CALIBRATION_STATUS",
    "DEFAULT_L_DIRECT_FACTOR",
    "DEFAULT_MAX_REFLECTIONS",
    "DEFAULT_NOISE_STD_V",
    "DEFAULT_WALL_REFLECTION_COEF",
    "FiberMicV3Spec",
    "MEASUREMENT_WINDOW_S",
    "WAVEFORM_SAMPLES",
    "simulate_fiber_mic_measurement",
]
