"""V1-compatible helpers used by shared simulation modules.

This module intentionally lives under ``sim_common`` so shared V2/V3 code does
not import the ambiguous top-level ``scripts`` package.
"""

DISTANCE_STAGE_GROUP_SIZE = 5
DISTANCE_STAGE_PATHS_M = (0.2, 0.6, 1.0, 1.4, 1.8)
DEFAULT_SEED = 20260425

PRESSURE_MIN_MPA = 0.1
PRESSURE_MAX_MPA = 0.709
BASELINE_PRESSURE_MAX_MPA = 0.3

MEASUREMENT_N2_RANGE = (0.0, 20.0)
MEASUREMENT_CH4_MIN = 40.0

STAGE_WEIGHTS = {
    "baseline_stage": 0.10,
    "distance_stage": 0.40,
    "pressure_stage": 0.40,
    "purge_stage": 0.10,
}


def fmt(value, digits):
    return f"{value:.{digits}f}"


def _condition_rows(sample_count, rng):
    """Generate V1-compatible condition rows for shared V2/V3 sampling."""

    rows = []
    distance_group_base = None
    distance_group_index = 0
    distance_group_step = 0

    for index in range(sample_count):
        stage_id = _stage_for_index(index, sample_count)

        if stage_id == "distance_stage":
            if distance_group_base is None or distance_group_step >= DISTANCE_STAGE_GROUP_SIZE:
                distance_group_base = _sample_condition_base(stage_id, rng)
                distance_group_index += 1
                distance_group_step = 0

            l_m = DISTANCE_STAGE_PATHS_M[distance_group_step % len(DISTANCE_STAGE_PATHS_M)]
            rows.append(
                _build_condition_row(
                    index=index,
                    stage_id=stage_id,
                    base=distance_group_base,
                    l_m=l_m,
                    mixture_id=f"MD{distance_group_index:05d}",
                    repeat_id=distance_group_step + 1,
                )
            )
            distance_group_step += 1
        else:
            rows.append(_condition_row(index, rng, sample_count))

    return rows


def _condition_row(index, rng, sample_count):
    stage_id = _stage_for_index(index, sample_count)
    base = _sample_condition_base(stage_id, rng)
    l_m = _path_length_for_stage(stage_id, rng)
    return _build_condition_row(
        index=index,
        stage_id=stage_id,
        base=base,
        l_m=l_m,
        mixture_id=f"M{index + 1:05d}",
        repeat_id=(index % 3) + 1,
    )


def _sample_condition_base(stage_id, rng):
    is_control = stage_id in {"baseline_stage", "purge_stage"}

    if is_control:
        x_n2 = rng.uniform(95.0, 100.0)
        remainder = 100.0 - x_n2
        x_h2 = rng.uniform(0.0, min(1.0, remainder))
        x_co2 = rng.uniform(0.0, max(0.0, remainder - x_h2))
        x_ch4 = 100.0 - x_n2 - x_h2 - x_co2
    else:
        x_h2, x_ch4, x_co2, x_n2 = _sample_v3_synced_measurement_components(rng)

    x_h2, x_ch4, x_co2, x_n2 = _normalize_components(x_h2, x_ch4, x_co2, x_n2)

    return {
        "x_H2": x_h2,
        "x_CH4": x_ch4,
        "x_CO2": x_co2,
        "x_N2": x_n2,
        "T_C": rng.uniform(15.0, 35.0),
        "P_MPa": _pressure_for_stage(stage_id, rng),
        "H_RH": rng.uniform(20.0, 80.0),
    }


def _sample_v3_synced_measurement_components(rng):
    for _ in range(128):
        x_h2 = _sample_hydrogen_percent(rng)
        x_co2 = rng.uniform(0.0, 15.0)
        x_n2 = rng.uniform(*MEASUREMENT_N2_RANGE)
        x_ch4 = 100.0 - x_h2 - x_co2 - x_n2
        if x_ch4 >= MEASUREMENT_CH4_MIN:
            return x_h2, x_ch4, x_co2, x_n2

    x_h2 = _sample_hydrogen_percent(rng)
    x_co2 = rng.uniform(0.0, 15.0)
    max_n2 = min(MEASUREMENT_N2_RANGE[1], 100.0 - x_h2 - x_co2 - MEASUREMENT_CH4_MIN)
    x_n2 = max(MEASUREMENT_N2_RANGE[0], max_n2)
    x_ch4 = 100.0 - x_h2 - x_co2 - x_n2
    return x_h2, x_ch4, x_co2, x_n2


def _build_condition_row(index, stage_id, base, l_m, mixture_id, repeat_id, sample_id_offset=0):
    p_mpa = base["P_MPa"]
    pressure_band = "low" if p_mpa < 0.24 else "mid" if p_mpa < 0.47 else "high"
    distance_band = "short" if l_m < 0.7 else "mid" if l_m < 1.3 else "long"
    is_control = stage_id in {"baseline_stage", "purge_stage"}

    return {
        "sample_id": f"S{index + 1 + sample_id_offset:05d}",
        "mixture_id": mixture_id,
        "stage_id": stage_id,
        "x_H2": fmt(base["x_H2"], 6),
        "x_CH4": fmt(base["x_CH4"], 6),
        "x_CO2": fmt(base["x_CO2"], 6),
        "x_N2": fmt(base["x_N2"], 6),
        "T_C": fmt(base["T_C"], 4),
        "P_MPa": fmt(p_mpa, 4),
        "H_RH": fmt(base["H_RH"], 4),
        "L_m": fmt(l_m, 4),
        "piston_position_m": fmt(l_m, 4),
        "pressure_stage": pressure_band,
        "distance_stage": distance_band,
        "repeat_id": str(repeat_id),
        "status": "calibration_control" if is_control else "synthetic_measurement",
    }


def _normalize_components(x_h2, x_ch4, x_co2, x_n2):
    total = x_h2 + x_ch4 + x_co2 + x_n2
    x_h2, x_ch4, x_co2, x_n2 = [v * 100.0 / total for v in (x_h2, x_ch4, x_co2, x_n2)]
    x_h2 = round(x_h2, 6)
    x_co2 = round(x_co2, 6)
    x_n2 = round(x_n2, 6)
    x_ch4 = round(100.0 - x_h2 - x_co2 - x_n2, 6)
    return x_h2, x_ch4, x_co2, x_n2


def _stage_for_index(index, sample_count):
    ratio = index / sample_count
    if ratio < STAGE_WEIGHTS["baseline_stage"]:
        return "baseline_stage"
    if ratio < STAGE_WEIGHTS["baseline_stage"] + STAGE_WEIGHTS["distance_stage"]:
        return "distance_stage"
    if ratio < STAGE_WEIGHTS["baseline_stage"] + STAGE_WEIGHTS["distance_stage"] + STAGE_WEIGHTS["pressure_stage"]:
        return "pressure_stage"
    return "purge_stage"


def _sample_hydrogen_percent(rng):
    marker = rng.random()
    if marker < 0.15:
        return rng.uniform(0.0, 3.0)
    if marker > 0.85:
        return rng.uniform(25.0, 30.0)
    return rng.uniform(0.0, 30.0)


def _pressure_for_stage(stage_id, rng):
    if stage_id == "baseline_stage":
        return rng.uniform(PRESSURE_MIN_MPA, BASELINE_PRESSURE_MAX_MPA)
    return rng.uniform(PRESSURE_MIN_MPA, PRESSURE_MAX_MPA)


def _path_length_for_stage(stage_id, rng):
    if stage_id == "distance_stage":
        return rng.uniform(0.2, 1.8)
    return rng.uniform(0.4, 1.2)
