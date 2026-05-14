import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.acoustic_v2 import (
    PROCESSING_PARAMS_V2,
    _hidden_attenuation_v2,
    _hidden_sound_speed_v2,
)


CENTER_FREQUENCY_HZ = 40000.0
BURST_CYCLES = 8
SAMPLE_RATE_HZ = 200000
MEASUREMENT_WINDOW_S = 0.005
WAVEFORM_SAMPLES = int(round(SAMPLE_RATE_HZ * MEASUREMENT_WINDOW_S))
ADC_MAX_INT16 = 32767
DEFAULT_NOISE_STD_V = 1e-3
CALIBRATION_STATUS = "pending"


@dataclass(frozen=True)
class WaveformV3Spec:
    sample_rate_hz: int = SAMPLE_RATE_HZ
    center_frequency_hz: float = CENTER_FREQUENCY_HZ
    burst_cycles: int = BURST_CYCLES
    measurement_window_s: float = MEASUREMENT_WINDOW_S
    adc_max_int16: int = ADC_MAX_INT16
    noise_std_v: float = DEFAULT_NOISE_STD_V
    calibration_status: str = CALIBRATION_STATUS

    @property
    def waveform_samples(self) -> int:
        return int(round(self.sample_rate_hz * self.measurement_window_s))

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["waveform_samples"] = self.waveform_samples
        return payload


def generate_burst_pulse(
    center_frequency_hz=CENTER_FREQUENCY_HZ,
    burst_cycles=BURST_CYCLES,
    sample_rate_hz=SAMPLE_RATE_HZ,
    amplitude_v=1.0,
):
    sample_count = int(round(burst_cycles * sample_rate_hz / center_frequency_hz))
    t = np.arange(sample_count, dtype=np.float32) / float(sample_rate_hz)
    window = np.hanning(sample_count).astype(np.float32)
    pulse = amplitude_v * window * np.sin(2.0 * math.pi * center_frequency_hz * t)
    return pulse.astype(np.float32)


def simulate_waveform_measurement(
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
    spec: WaveformV3Spec | None = None,
):
    if spec is None:
        spec = WaveformV3Spec(
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
    tof_s = float(l_m) / c_sound
    pulse = generate_burst_pulse(
        center_frequency_hz=spec.center_frequency_hz,
        burst_cycles=spec.burst_cycles,
        sample_rate_hz=spec.sample_rate_hz,
        amplitude_v=1.0,
    )
    waveform = np.zeros(spec.waveform_samples, dtype=np.float32)
    peak_index = int(round(tof_s * spec.sample_rate_hz))
    start = peak_index
    if start < 0 or start >= waveform.shape[0]:
        raise ValueError("pulse start is outside measurement window")

    usable = min(pulse.shape[0], waveform.shape[0] - start)
    if usable <= 0:
        raise ValueError("pulse is outside measurement window")
    amp_scale = math.exp(-alpha_true_npm * float(l_m))
    waveform[start : start + usable] = pulse[:usable] * amp_scale
    if spec.noise_std_v > 0.0:
        noise_rng = np.random.default_rng(rng.randrange(0, 2**32))
        waveform = waveform + noise_rng.normal(0.0, spec.noise_std_v, size=waveform.shape).astype(np.float32)
    peak_abs_v = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    if peak_abs_v <= 0.0:
        raise ValueError("peak_abs_v must be > 0")
    scale_factor = peak_abs_v / spec.adc_max_int16
    waveform_int16 = np.clip(np.round(waveform / scale_factor), -spec.adc_max_int16, spec.adc_max_int16).astype(
        np.int16
    )
    return {
        "waveform_float": waveform.astype(np.float32),
        "waveform_int16": waveform_int16,
        "scale_factor": float(scale_factor),
        "tof_s": tof_s,
        "peak_index": peak_index,
        "peak_abs_v": peak_abs_v,
        "alpha_true_npm": alpha_true_npm,
        "sound_speed_m_per_s": float(c_sound),
        "sample_rate_hz": int(spec.sample_rate_hz),
        "center_frequency_hz": float(spec.center_frequency_hz),
        "measurement_window_s": float(spec.measurement_window_s),
        "calibration_status": spec.calibration_status,
        "spec": spec.to_dict(),
    }


__all__ = [
    "ADC_MAX_INT16",
    "BURST_CYCLES",
    "CALIBRATION_STATUS",
    "CENTER_FREQUENCY_HZ",
    "DEFAULT_NOISE_STD_V",
    "MEASUREMENT_WINDOW_S",
    "PROCESSING_PARAMS_V2",
    "SAMPLE_RATE_HZ",
    "WAVEFORM_SAMPLES",
    "WaveformV3Spec",
    "generate_burst_pulse",
    "simulate_waveform_measurement",
]
