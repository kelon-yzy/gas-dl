EXPECTED_CHANNEL_NAMES = [
    "V_NDIR_CH4",
    "V_NDIR_CO2",
    "V_TCS",
    "T_C",
    "P_MPa",
    "H_RH",
    "L_m",
    "piston_position_m",
    "TOF",
    "Amp",
    "f_peak",
    "A_fft_max",
]

EXPECTED_LABEL_NAMES = ["x_H2", "x_CH4", "x_CO2", "x_N2"]

CHANNEL_GROUPS = {
    "optical": [0, 1],
    "thermal": [2],
    "environment": [3, 4, 5, 6, 7],
    "acoustic": [8, 9, 10, 11],
}

TIME_WINDOWS = {
    "all": None,
    "baseline": list(range(0, 20)),
    "exposure": list(range(20, 70)),
    "steady": list(range(70, 100)),
    "recovery": list(range(100, 120)),
    "baseline_exposure": list(range(0, 70)),
    "baseline+exposure": list(range(0, 70)),
    "no_baseline": list(range(20, 120)),
}


def resolve_channel_indices(channels):
    if channels is None or channels == "all":
        return None
    if isinstance(channels, str):
        if channels not in CHANNEL_GROUPS:
            raise ValueError(f"Unknown channel group: {channels}")
        return CHANNEL_GROUPS[channels]
    resolved = []
    for item in channels:
        if isinstance(item, int):
            resolved.append(item)
        elif isinstance(item, str) and item in CHANNEL_GROUPS:
            resolved.extend(CHANNEL_GROUPS[item])
        elif isinstance(item, str) and item in EXPECTED_CHANNEL_NAMES:
            resolved.append(EXPECTED_CHANNEL_NAMES.index(item))
        else:
            raise ValueError(f"Unknown channel selector: {item}")
    return sorted(dict.fromkeys(resolved))


def resolve_time_indices(time_window):
    if time_window is None or time_window == "all":
        return None
    if isinstance(time_window, str):
        if time_window not in TIME_WINDOWS:
            raise ValueError(f"Unknown time window: {time_window}")
        return TIME_WINDOWS[time_window]
    return list(time_window)
