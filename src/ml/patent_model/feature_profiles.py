"""Feature profile definitions for V1 traditional-model inputs."""

from __future__ import annotations

from patent_model.config import (
    ACOUSTIC_FEATURE_COLUMNS,
    ENVIRONMENT_FEATURE_COLUMNS,
    OPTICAL_FEATURE_COLUMNS,
    THERMAL_FEATURE_COLUMNS,
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
}


def get_feature_profile(name: str) -> dict[str, object]:
    if name not in FEATURE_PROFILES:
        raise ValueError(f"Unknown feature profile: {name}")
    return FEATURE_PROFILES[name]
