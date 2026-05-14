"""Offline Monte Carlo augmentation for derived environment features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from patent_model.dataset import PatentDataset
from patent_model.robustness import add_profile_environment_noise


def _with_augmented_ids(dataset: PatentDataset, copy_index: int) -> PatentDataset:
    sample_ids = np.array([f"{sample_id}__mc{copy_index}" for sample_id in dataset.sample_ids], dtype=object)
    metadata = dataset.metadata.copy()
    metadata["sample_id"] = sample_ids
    metadata["env_augmentation"] = "mc"
    metadata["env_augmentation_index"] = copy_index
    return PatentDataset(
        sample_ids=sample_ids,
        acoustic=dataset.acoustic.copy(),
        optical=dataset.optical.copy(),
        thermal=dataset.thermal.copy(),
        environment=dataset.environment.copy(),
        targets=dataset.targets.copy(),
        component_names=dataset.component_names,
        metadata=metadata,
        acoustic_columns=dataset.acoustic_columns,
        optical_columns=dataset.optical_columns,
        thermal_columns=dataset.thermal_columns,
        environment_columns=dataset.environment_columns,
        provenance=dict(dataset.provenance),
        filter_report=dict(dataset.filter_report),
    )


def _concat_datasets(parts: list[PatentDataset]) -> PatentDataset:
    return PatentDataset(
        sample_ids=np.concatenate([part.sample_ids for part in parts]),
        acoustic=np.vstack([part.acoustic for part in parts]),
        optical=np.vstack([part.optical for part in parts]),
        thermal=np.vstack([part.thermal for part in parts]),
        environment=np.vstack([part.environment for part in parts]),
        targets=np.vstack([part.targets for part in parts]),
        component_names=parts[0].component_names,
        metadata=pd.concat([part.metadata for part in parts], ignore_index=True),
        acoustic_columns=parts[0].acoustic_columns,
        optical_columns=parts[0].optical_columns,
        thermal_columns=parts[0].thermal_columns,
        environment_columns=parts[0].environment_columns,
        provenance=dict(parts[0].provenance),
        filter_report=dict(parts[0].filter_report),
    )


def augment_derived_env_training_data(
    dataset: PatentDataset,
    mc_samples: int,
    sigma_t: float,
    sigma_p: float,
    sigma_h: float,
    seed: int,
    profile: str = "derived_env",
) -> PatentDataset:
    """Return original training rows plus MC noisy copies for derived_env inputs."""

    if mc_samples < 0:
        raise ValueError("mc_samples must be >= 0.")
    original_metadata = dataset.metadata.copy()
    original_metadata["env_augmentation"] = "original"
    original_metadata["env_augmentation_index"] = 0
    original_metadata["feature_profile"] = profile
    original = PatentDataset(
        sample_ids=dataset.sample_ids.copy(),
        acoustic=dataset.acoustic.copy(),
        optical=dataset.optical.copy(),
        thermal=dataset.thermal.copy(),
        environment=dataset.environment.copy(),
        targets=dataset.targets.copy(),
        component_names=dataset.component_names,
        metadata=original_metadata,
        acoustic_columns=dataset.acoustic_columns,
        optical_columns=dataset.optical_columns,
        thermal_columns=dataset.thermal_columns,
        environment_columns=dataset.environment_columns,
        provenance=dict(dataset.provenance),
        filter_report=dict(dataset.filter_report),
    )
    if mc_samples == 0:
        return original

    parts = [original]
    for copy_index in range(1, mc_samples + 1):
        noisy = add_profile_environment_noise(
            dataset,
            profile=profile,
            sigma_t=sigma_t,
            sigma_p=sigma_p,
            sigma_h=sigma_h,
            seed=seed + copy_index,
        )
        parts.append(_with_augmented_ids(noisy, copy_index))
    return _concat_datasets(parts)
