"""Feature profile definitions for traditional-model inputs."""

from __future__ import annotations

from patent_model.config import (
    ACOUSTIC_FEATURE_COLUMNS,
    ENVIRONMENT_FEATURE_COLUMNS,
    OPTICAL_FEATURE_COLUMNS,
    THERMAL_FEATURE_COLUMNS,
)


DUAL_WAVEFORM_ACOUSTIC_COLUMNS = (
    "TOF",
    "Amp",
    "f_peak",
    "A_fft_max",
)

DUAL_WAVEFORM_ACOUSTIC_ENV_COLUMNS = DUAL_WAVEFORM_ACOUSTIC_COLUMNS + (
    "T_C",
    "P_MPa",
    "H_RH",
    "T_K",
    "P_kPa",
    "p_H2O_kPa",
    "x_H2O",
    "AH_g_m3",
    "P_dry_kPa",
    "sound_speed",
    "attenuation_alpha",
    "c_sound",
    "c_T_norm",
    "delta_Amp",
)

DUAL_WAVEFORM_OPTICAL_COLUMNS = (
    "V_NDIR_CH4",
    "V_NDIR_CO2",
    "delta_I_CH4",
    "delta_I_CO2",
    "A_NDIR_CH4",
    "A_NDIR_CO2",
)

DUAL_WAVEFORM_OPTICAL_ENV_COLUMNS = (
    *DUAL_WAVEFORM_OPTICAL_COLUMNS,
    "T_C",
    "P_MPa",
    "H_RH",
    "T_K",
    "P_kPa",
    "p_H2O_kPa",
    "x_H2O",
    "AH_g_m3",
    "P_dry_kPa",
)

DUAL_WAVEFORM_THERMAL_ENV_COLUMNS = (
    "V_TCS",
    "T_C",
    "P_MPa",
    "H_RH",
    "T_K",
    "P_kPa",
    "p_H2O_kPa",
    "x_H2O",
    "AH_g_m3",
    "P_dry_kPa",
)


FEATURE_PROFILES = {
    "raw_no_env": {
        "acoustic_file": "training/train_acoustic.csv",
        "optical_file": "training/train_optical.csv",
        "thermal_file": "training/train_thermal.csv",
        "feature_table_file": "features/feature_table.csv",
        "acoustic_columns": ACOUSTIC_FEATURE_COLUMNS,
        "optical_columns": OPTICAL_FEATURE_COLUMNS,
        "thermal_columns": THERMAL_FEATURE_COLUMNS,
        "environment_columns": ENVIRONMENT_FEATURE_COLUMNS,
        "include_environment": False,
        "component_mode": "three",
    },
    "raw_tph": {
        "acoustic_file": "training/train_acoustic.csv",
        "optical_file": "training/train_optical.csv",
        "thermal_file": "training/train_thermal.csv",
        "feature_table_file": "features/feature_table.csv",
        "acoustic_columns": ACOUSTIC_FEATURE_COLUMNS,
        "optical_columns": OPTICAL_FEATURE_COLUMNS,
        "thermal_columns": THERMAL_FEATURE_COLUMNS,
        "environment_columns": ENVIRONMENT_FEATURE_COLUMNS,
        "include_environment": True,
        "component_mode": "three",
    },
    "derived_env": {
        "acoustic_file": "training/train_acoustic_env.csv",
        "optical_file": "training/train_optical_env.csv",
        "thermal_file": "training/train_thermal_env.csv",
        "feature_table_file": "features/feature_table_env.csv",
        "acoustic_columns": None,
        "optical_columns": None,
        "thermal_columns": None,
        "environment_columns": ENVIRONMENT_FEATURE_COLUMNS,
        "include_environment": False,
        "component_mode": "three",
    },
    "raw_no_env_four": {
        "acoustic_file": "training/train_acoustic.csv",
        "optical_file": "training/train_optical.csv",
        "thermal_file": "training/train_thermal.csv",
        "feature_table_file": "features/feature_table.csv",
        "acoustic_columns": ACOUSTIC_FEATURE_COLUMNS,
        "optical_columns": OPTICAL_FEATURE_COLUMNS,
        "thermal_columns": THERMAL_FEATURE_COLUMNS,
        "environment_columns": ENVIRONMENT_FEATURE_COLUMNS,
        "include_environment": False,
        "component_mode": "four",
    },
    "raw_tph_four": {
        "acoustic_file": "training/train_acoustic.csv",
        "optical_file": "training/train_optical.csv",
        "thermal_file": "training/train_thermal.csv",
        "feature_table_file": "features/feature_table.csv",
        "acoustic_columns": ACOUSTIC_FEATURE_COLUMNS,
        "optical_columns": OPTICAL_FEATURE_COLUMNS,
        "thermal_columns": THERMAL_FEATURE_COLUMNS,
        "environment_columns": ENVIRONMENT_FEATURE_COLUMNS,
        "include_environment": True,
        "component_mode": "four",
    },
    "derived_env_four": {
        "acoustic_file": "training/train_acoustic_env.csv",
        "optical_file": "training/train_optical_env.csv",
        "thermal_file": "training/train_thermal_env.csv",
        "feature_table_file": "features/feature_table_env.csv",
        "acoustic_columns": None,
        "optical_columns": None,
        "thermal_columns": None,
        "environment_columns": ENVIRONMENT_FEATURE_COLUMNS,
        "include_environment": False,
        "component_mode": "four",
    },
    "v3_waveform_dual_channel_four": {
        "acoustic_file": "training/train_acoustic.csv",
        "optical_file": "training/train_optical.csv",
        "thermal_file": "training/train_thermal.csv",
        "feature_table_file": "features/feature_table.csv",
        "acoustic_columns": DUAL_WAVEFORM_ACOUSTIC_COLUMNS,
        "optical_columns": DUAL_WAVEFORM_OPTICAL_COLUMNS,
        "thermal_columns": THERMAL_FEATURE_COLUMNS,
        "environment_columns": ENVIRONMENT_FEATURE_COLUMNS,
        "include_environment": False,
        "component_mode": "four",
    },
    "v3_waveform_dual_channel_tph_four": {
        "acoustic_file": "training/train_acoustic.csv",
        "optical_file": "training/train_optical.csv",
        "thermal_file": "training/train_thermal.csv",
        "feature_table_file": "features/feature_table.csv",
        "acoustic_columns": DUAL_WAVEFORM_ACOUSTIC_COLUMNS,
        "optical_columns": DUAL_WAVEFORM_OPTICAL_COLUMNS,
        "thermal_columns": THERMAL_FEATURE_COLUMNS,
        "environment_columns": ENVIRONMENT_FEATURE_COLUMNS,
        "include_environment": True,
        "component_mode": "four",
    },
    "v3_waveform_dual_channel_env_four": {
        "acoustic_file": "training/train_acoustic_env.csv",
        "optical_file": "training/train_optical_env.csv",
        "thermal_file": "training/train_thermal_env.csv",
        "feature_table_file": "features/feature_table_env.csv",
        "acoustic_columns": DUAL_WAVEFORM_ACOUSTIC_ENV_COLUMNS,
        "optical_columns": DUAL_WAVEFORM_OPTICAL_ENV_COLUMNS,
        "thermal_columns": DUAL_WAVEFORM_THERMAL_ENV_COLUMNS,
        "environment_columns": ENVIRONMENT_FEATURE_COLUMNS,
        "include_environment": False,
        "component_mode": "four",
    },
}


_EMBEDDED_ENV_COLUMNS = frozenset({
    "T_C", "P_MPa", "H_RH", "T_K", "P_kPa",
    "p_H2O_kPa", "x_H2O", "AH_g_m3", "P_dry_kPa",
})


def has_embedded_environment(profile: str) -> bool:
    """Check whether a feature profile embeds environment-derived columns in modality matrices.

    Returns True when any of acoustic/optical/thermal columns contain environment-
    derived column names, or when columns are None (meaning all CSV columns are used,
    which includes environment columns for *_env CSV files).
    """

    defn = FEATURE_PROFILES.get(profile)
    if defn is None:
        return False
    for key in ("acoustic_columns", "optical_columns", "thermal_columns"):
        columns = defn.get(key)
        if columns is None:
            return True
        if _EMBEDDED_ENV_COLUMNS & set(columns):
            return True
    return False


def get_feature_profile(name: str) -> dict[str, object]:
    if name not in FEATURE_PROFILES:
        raise ValueError(f"Unknown feature profile: {name}")
    return FEATURE_PROFILES[name]
