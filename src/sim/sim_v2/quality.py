# V2 数据包质量摘要：以 sim_common.build_common_summary 为骨架 + V2 扩展字段。

from sim_common.quality import build_common_summary

from .constants import (
    BASELINE_PATH_LMS,
    LABEL_FIELDS,
    MULTI_PATH_PHASE_BASELINE,
    MULTI_PATH_PHASE_OFF,
    MULTI_PATH_PHASE_STEADY,
    SEQUENCE_CHANNELS,
    normalize_multi_path_phase,
)


def quality_summary(
    sequence_ids,
    timesteps,
    split_distribution,
    split_warnings,
    label_fields=LABEL_FIELDS,
    dataset_version="V2 sequence",
    acoustic_version="v1",
    multi_path_phase=MULTI_PATH_PHASE_OFF,
    acoustic_summary=None,
):
    """V2 质量摘要：通用骨架 + V2 扩展字段。

    扩展字段：response/drift/noise model、acoustic_version、multi_path、
    acoustic_derived_note、derived_feature_files、label_note 等。
    """
    multi_path_phase = normalize_multi_path_phase(multi_path_phase)

    summary = build_common_summary(
        sequence_ids=sequence_ids,
        timesteps=timesteps,
        split_distribution=split_distribution,
        split_warnings=split_warnings,
        label_fields=label_fields,
        dataset_version=dataset_version,
        simulation_level="structure_level_dynamic_simulation",
        channel_names=SEQUENCE_CHANNELS,
        shape={
            "sequences": len(sequence_ids),
            "timesteps": timesteps,
            "channels": len(SEQUENCE_CHANNELS),
            "labels": len(label_fields),
        },
    )

    if multi_path_phase == MULTI_PATH_PHASE_BASELINE:
        multi_path_note = (
            "Multi-path scan in BASELINE phase (pure N2). attenuation_alpha_fit reflects "
            "N2 background attenuation, used for link calibration only. Acoustic-related channels "
            "(TOF/Amp/f_peak/A_fft_max/L_m/piston_position_m) follow the path schedule; "
            "other channels stay on default baseline."
        )
    elif multi_path_phase == MULTI_PATH_PHASE_STEADY:
        multi_path_note = (
            "Multi-path scan in STEADY phase (target mixture). attenuation_alpha_fit reflects "
            "actual mixture attenuation including CH4/CO2/H2O contributions. Acoustic-related "
            "channels follow the path schedule; other channels stay on default steady targets."
        )
    else:
        multi_path_note = "Multi-path scan disabled."

    summary.update({
        "derived_feature_files": {
            "acoustic_derived_sequence_long": "sequences/acoustic_derived_sequence_long.csv",
        },
        "response_model": "first_order_system_equivalent",
        "drift_model": "linear_plus_random_walk",
        "noise_model": "gaussian_channel_noise",
        "acoustic_version": acoustic_version,
        "multi_path": {
            "phase": multi_path_phase,
            "path_lms": list(BASELINE_PATH_LMS) if multi_path_phase != MULTI_PATH_PHASE_OFF else [],
            "note": multi_path_note,
        },
        # 兼容字段：旧消费方按 multi_path_baseline.enabled 读
        "multi_path_baseline": {
            "enabled": multi_path_phase == MULTI_PATH_PHASE_BASELINE,
            "path_lms": list(BASELINE_PATH_LMS) if multi_path_phase == MULTI_PATH_PHASE_BASELINE else [],
            "note": multi_path_note,
        },
        "acoustic_note": (
            "TOF/Amp/f_peak/A_fft_max are low-frequency window-level features extracted "
            "from high-frequency DAQ signals, not raw acoustic waveforms."
        ),
        "acoustic": acoustic_summary or {},
        "acoustic_derived_note": (
            "attenuation_alpha_rel uses each sequence baseline-phase mean Amp as the local "
            "reference. attenuation_alpha_calibrated first maps the N2 baseline Amp back to "
            "the source/link reference using the simulated N2 attenuation model. "
            "When multi_path scan is enabled, attenuation_alpha_fit and Amp_ref_fit are "
            "estimated from ln(Amp) vs L_m over the configured phase sub-segments. "
            "Reference for fit error: N2 alpha for baseline-phase scan, mixture alpha for steady-phase scan."
        ),
        "label_note": (
            "sequence labels represent target mixture composition, not instantaneous "
            "composition at every timestep."
        ),
    })
    return summary
