# 通用划分：每条序列独立（不按 mixture_id 分组）+ V3 分层划分 / 外推保留。
#
# V2 仍可继续使用 `split_mixture_ids` 的简单随机 split。
# V3 waveform 正式数据默认改用
# `build_stratified_group_splits_with_extrapolation`。

from collections import Counter, defaultdict

import numpy as np

from .v1_helpers import DEFAULT_SEED


STRATIFIED_GROUP_SPLIT_POLICY = "stratified_independent_sequence_with_extrapolation_holdout"
SMALL_DATASET_FALLBACK_POLICY = "small_dataset_random_independent_sequence"
DEFAULT_STRATIFY_FIELDS = ("x_H2", "x_CO2", "x_N2", "P_MPa_base", "L_m_base")
CORE_SPLIT_NAMES = ("train", "val", "test")
ALL_SPLIT_NAMES = ("train", "val", "test", "extrapolation")


def split_mixture_ids(mixture_ids, train_ratio=0.70, val_ratio=0.15, seed=DEFAULT_SEED):
    """按 mixture_id 去重后随机分组划分 train/val/test。"""
    rng = np.random.default_rng(seed)
    ids = np.array(sorted(set(mixture_ids)))
    rng.shuffle(ids)

    n = len(ids)
    if n == 0:
        return set(), set(), set()
    if n == 1:
        return {str(ids[0])}, set(), set()
    if n == 2:
        return {str(ids[0])}, set(), {str(ids[1])}

    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1

    train_ids = {str(value) for value in ids[:n_train]}
    val_ids = {str(value) for value in ids[n_train : n_train + n_val]}
    test_ids = {str(value) for value in ids[n_train + n_val :]}
    return train_ids, val_ids, test_ids


def build_split_rows(conditions, train_ids, val_ids, test_ids):
    """兼容旧接口：根据 sequence_id 将序列条件分配到 train/val/test 三组。"""
    split_groups = {
        "train": set(train_ids),
        "val": set(val_ids),
        "test": set(test_ids),
    }
    return build_split_rows_from_group_sets(conditions, split_groups, group_field="sequence_id")


def build_split_rows_from_group_sets(conditions, split_groups, group_field="sequence_id"):
    """根据 group_field 将序列条件分配到多个 split。"""
    rows = {name: [] for name in split_groups}
    for condition in conditions:
        group_value = str(condition[group_field])
        split_name = _split_name_for_group(group_value, split_groups)
        rows[split_name].append(
            {
                "sequence_id": condition["sequence_id"],
                "mixture_id": condition["sequence_id"],
            }
        )
    return rows


def _split_name_for_group(group_value, split_groups):
    for split_name, group_set in split_groups.items():
        if group_value in group_set:
            return split_name
    raise ValueError(f"group {group_value!r} was not assigned to a split")


def build_stratified_group_splits_with_extrapolation(
    conditions,
    *,
    group_field="sequence_id",
    stratify_fields=DEFAULT_STRATIFY_FIELDS,
    extrapolation_ratio=0.15,
    boundary_quantile=0.10,
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=DEFAULT_SEED,
):
    """按 group_field 分组后做分层 train/val/test，并额外保留 extrapolation。"""
    if not conditions:
        raise ValueError("conditions must not be empty")
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
    if not (0.0 < extrapolation_ratio < 1.0):
        raise ValueError("extrapolation_ratio must be in (0, 1)")
    if not (0.0 < boundary_quantile < 0.5):
        raise ValueError("boundary_quantile must be in (0, 0.5)")

    group_rows = _group_conditions(conditions, group_field)
    if len(group_rows) < 3:
        return _build_small_dataset_split_result(
            conditions,
            group_field=group_field,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
        )
    total_sequences = len(conditions)
    rng = np.random.default_rng(seed)

    boundary_thresholds = _compute_two_sided_quantiles(conditions, stratify_fields, boundary_quantile)
    boundary_profiles = _build_boundary_profiles(group_rows, stratify_fields, boundary_thresholds)
    extrapolation_groups = _select_extrapolation_groups(
        boundary_profiles,
        target_sequences=max(1, int(round(total_sequences * extrapolation_ratio))),
        rng=rng,
    )

    regular_group_rows = {
        group_id: rows
        for group_id, rows in group_rows.items()
        if group_id not in extrapolation_groups
    }
    if len(regular_group_rows) < 3:
        return _build_small_dataset_split_result(
            conditions,
            group_field=group_field,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
        )

    regular_rows = []
    for rows in regular_group_rows.values():
        regular_rows.extend(rows)
    tertile_thresholds = _compute_tertiles(regular_rows, stratify_fields)
    regular_profiles = _build_regular_profiles(regular_group_rows, stratify_fields, tertile_thresholds, rng)

    split_groups = _assign_regular_groups(
        regular_profiles,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    split_groups["extrapolation"] = set(extrapolation_groups)

    split_rows = build_split_rows_from_group_sets(conditions, split_groups, group_field=group_field)
    split_summary = _build_split_summary(
        conditions=conditions,
        split_rows=split_rows,
        split_groups=split_groups,
        group_field=group_field,
        stratify_fields=stratify_fields,
        boundary_thresholds=boundary_thresholds,
        tertile_thresholds=tertile_thresholds,
        boundary_profiles=boundary_profiles,
        extrapolation_ratio=extrapolation_ratio,
        boundary_quantile=boundary_quantile,
        seed=seed,
    )
    return split_rows, split_summary


def _group_conditions(conditions, group_field):
    output = defaultdict(list)
    for row in conditions:
        output[str(row[group_field])].append(row)
    return dict(output)


def _build_small_dataset_split_result(conditions, *, group_field, train_ratio, val_ratio, seed):
    train_ids, val_ids, test_ids = split_mixture_ids(
        [str(row[group_field]) for row in conditions],
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )
    split_groups = {
        "train": set(train_ids),
        "val": set(val_ids),
        "test": set(test_ids),
        "extrapolation": set(),
    }
    split_rows = build_split_rows_from_group_sets(conditions, split_groups, group_field=group_field)
    split_summary = {
        "split_policy": SMALL_DATASET_FALLBACK_POLICY,
        "group_field": group_field,
        "seed": int(seed),
        "stratify_fields": list(DEFAULT_STRATIFY_FIELDS),
        "boundary_quantile": None,
        "extrapolation_ratio_target": 0.0,
        "boundary_thresholds": {},
        "regular_tertile_thresholds": {},
        "candidate_extrapolation_groups": 0,
        "selected_extrapolation_groups": 0,
        "candidate_extrapolation_sequences": 0,
        "selected_extrapolation_sequences": 0,
        "boundary_reason_totals": {},
        "boundary_reason_selected": {},
        "total_sequences": int(len(conditions)),
        "total_groups": int(len({str(row[group_field]) for row in conditions})),
        "splits": {
            name: {
                "sequence_count": int(len(rows)),
                "group_count": int(len(split_groups[name])),
                "group_coverage_ratio": 0.0,
                "sequence_coverage_ratio": 0.0,
            }
            for name, rows in split_rows.items()
        },
        "stratify_distribution": {
            name: _split_field_stats([], DEFAULT_STRATIFY_FIELDS)
            for name in split_rows
        },
    }
    return split_rows, split_summary


def _compute_two_sided_quantiles(rows, fields, quantile):
    thresholds = {}
    for field in fields:
        values = np.array([float(row[field]) for row in rows], dtype=np.float64)
        thresholds[field] = {
            "low": float(np.quantile(values, quantile)),
            "high": float(np.quantile(values, 1.0 - quantile)),
        }
    return thresholds


def _compute_tertiles(rows, fields):
    thresholds = {}
    for field in fields:
        values = np.array([float(row[field]) for row in rows], dtype=np.float64)
        thresholds[field] = {
            "q33": float(np.quantile(values, 1.0 / 3.0)),
            "q67": float(np.quantile(values, 2.0 / 3.0)),
        }
    return thresholds


def _build_boundary_profiles(group_rows, fields, thresholds):
    profiles = {}
    for group_id, rows in group_rows.items():
        reasons = set()
        for row in rows:
            for field in fields:
                value = float(row[field])
                if value <= thresholds[field]["low"]:
                    reasons.add(f"{field}:low")
                if value >= thresholds[field]["high"]:
                    reasons.add(f"{field}:high")
        profiles[group_id] = {
            "group_id": group_id,
            "sequence_count": len(rows),
            "reasons": tuple(sorted(reasons)),
            "is_candidate": bool(reasons),
        }
    return profiles


def _select_extrapolation_groups(boundary_profiles, target_sequences, rng):
    candidates = [profile for profile in boundary_profiles.values() if profile["is_candidate"]]
    if not candidates:
        return set()

    shuffled = list(candidates)
    rng.shuffle(shuffled)
    reason_counts = Counter()
    selected = set()
    selected_sequences = 0

    while shuffled and selected_sequences < target_sequences:
        remaining = target_sequences - selected_sequences
        best = None
        best_score = None
        for profile in shuffled:
            coverage_gain = sum(1.0 / (1.0 + reason_counts[reason]) for reason in profile["reasons"])
            size_penalty = abs(profile["sequence_count"] - remaining) / max(target_sequences, 1)
            overshoot_penalty = max(0, selected_sequences + profile["sequence_count"] - target_sequences) / max(target_sequences, 1)
            score = coverage_gain - size_penalty - overshoot_penalty
            tie_break = (-profile["sequence_count"], profile["group_id"])
            if best is None or score > best_score or (score == best_score and tie_break > best[2]):
                best = (profile, score, tie_break)
                best_score = score
        profile = best[0]
        selected.add(profile["group_id"])
        selected_sequences += profile["sequence_count"]
        for reason in profile["reasons"]:
            reason_counts[reason] += 1
        shuffled = [item for item in shuffled if item["group_id"] != profile["group_id"]]
    return selected


def _build_regular_profiles(group_rows, fields, tertile_thresholds, rng):
    profiles = []
    for group_id, rows in group_rows.items():
        bin_counts = Counter()
        mean_values = {}
        for row in rows:
            for field in fields:
                bin_counts[(field, _bin_name(float(row[field]), tertile_thresholds[field]["q33"], tertile_thresholds[field]["q67"]))] += 1
        for field in fields:
            mean_values[field] = float(sum(float(row[field]) for row in rows) / len(rows))
        stratum_key = tuple(
            _bin_name(mean_values[field], tertile_thresholds[field]["q33"], tertile_thresholds[field]["q67"])
            for field in fields
        )
        profiles.append(
            {
                "group_id": group_id,
                "sequence_count": len(rows),
                "bin_counts": dict(bin_counts),
                "bin_support": len(bin_counts),
                "mean_values": mean_values,
                "stratum_key": stratum_key,
                "shuffle_key": float(rng.random()),
            }
        )
    profiles.sort(key=lambda item: (-item["sequence_count"], -item["bin_support"], item["shuffle_key"]))
    return profiles


def _bin_name(value, q33, q67):
    if value <= q33:
        return "low"
    if value >= q67:
        return "high"
    return "mid"


def _assign_regular_groups(regular_profiles, *, train_ratio, val_ratio, test_ratio, seed):
    del seed
    split_groups = {name: set() for name in CORE_SPLIT_NAMES}
    split_sequence_counts = {name: 0 for name in CORE_SPLIT_NAMES}
    split_bin_counts = {name: Counter() for name in CORE_SPLIT_NAMES}

    total_sequences = sum(profile["sequence_count"] for profile in regular_profiles)
    total_bin_counts = Counter()
    for profile in regular_profiles:
        total_bin_counts.update(profile["bin_counts"])

    target_ratios = {
        "train": float(train_ratio),
        "val": float(val_ratio),
        "test": float(test_ratio),
    }
    target_sequences = {
        name: total_sequences * ratio
        for name, ratio in target_ratios.items()
    }
    target_bin_counts = {
        name: {
            key: total_bin_counts[key] * ratio
            for key in total_bin_counts
        }
        for name, ratio in target_ratios.items()
    }

    profiles_by_stratum = defaultdict(list)
    for profile in regular_profiles:
        profiles_by_stratum[profile["stratum_key"]].append(profile)

    stratum_sequence_totals = {
        key: sum(profile["sequence_count"] for profile in profiles)
        for key, profiles in profiles_by_stratum.items()
    }
    stratum_keys = sorted(
        profiles_by_stratum,
        key=lambda key: (-stratum_sequence_totals[key], key),
    )

    for stratum_key in stratum_keys:
        profiles = profiles_by_stratum[stratum_key]
        stratum_targets = {
            name: stratum_sequence_totals[stratum_key] * ratio
            for name, ratio in target_ratios.items()
        }
        stratum_counts = {name: 0 for name in CORE_SPLIT_NAMES}

        for local_index, profile in enumerate(profiles):
            remaining_groups = len(profiles) - local_index - 1
            best_split = None
            best_score = None
            for split_name in CORE_SPLIT_NAMES:
                score = _assignment_score(
                    split_name,
                    profile,
                    split_groups,
                    split_sequence_counts,
                    split_bin_counts,
                    target_sequences,
                    target_bin_counts,
                    remaining_groups,
                    stratum_counts,
                    stratum_targets,
                )
                if best_split is None or score < best_score or (score == best_score and split_name < best_split):
                    best_split = split_name
                    best_score = score
            split_groups[best_split].add(profile["group_id"])
            split_sequence_counts[best_split] += profile["sequence_count"]
            split_bin_counts[best_split].update(profile["bin_counts"])
            stratum_counts[best_split] += profile["sequence_count"]
    return split_groups


def _assignment_score(
    split_name,
    profile,
    split_groups,
    split_sequence_counts,
    split_bin_counts,
    target_sequences,
    target_bin_counts,
    remaining_groups,
    stratum_counts,
    stratum_targets,
):
    empty_splits = [name for name in CORE_SPLIT_NAMES if not split_groups[name]]
    if split_name not in empty_splits and empty_splits and remaining_groups < len(empty_splits):
        return float("inf")

    seq_after = split_sequence_counts[split_name] + profile["sequence_count"]
    global_score = abs(seq_after - target_sequences[split_name]) / max(target_sequences[split_name], 1.0)

    stratum_after = stratum_counts[split_name] + profile["sequence_count"]
    stratum_score = abs(stratum_after - stratum_targets[split_name]) / max(stratum_targets[split_name], 1.0)
    if stratum_after > stratum_targets[split_name]:
        stratum_score += (stratum_after - stratum_targets[split_name]) / max(stratum_targets[split_name], 1.0) * 4.0

    bin_score = 0.0
    for key, count in profile["bin_counts"].items():
        after = split_bin_counts[split_name][key] + count
        target = target_bin_counts[split_name][key]
        bin_score += abs(after - target) / max(target, 1.0)
    bin_score = bin_score / max(len(profile["bin_counts"]), 1)
    return global_score * 4.0 + stratum_score * 8.0 + bin_score


def _build_split_summary(
    *,
    conditions,
    split_rows,
    split_groups,
    group_field,
    stratify_fields,
    boundary_thresholds,
    tertile_thresholds,
    boundary_profiles,
    extrapolation_ratio,
    boundary_quantile,
    seed,
):
    by_sequence_id = {row["sequence_id"]: row for row in conditions}
    total_sequences = len(conditions)
    total_groups = len({str(row[group_field]) for row in conditions})
    candidate_groups = [profile for profile in boundary_profiles.values() if profile["is_candidate"]]
    selected_profiles = [boundary_profiles[group_id] for group_id in split_groups["extrapolation"]]
    reason_totals = Counter()
    reason_selected = Counter()
    for profile in candidate_groups:
        reason_totals.update(profile["reasons"])
    for profile in selected_profiles:
        reason_selected.update(profile["reasons"])

    return {
        "split_policy": STRATIFIED_GROUP_SPLIT_POLICY,
        "group_field": group_field,
        "seed": int(seed),
        "stratify_fields": list(stratify_fields),
        "boundary_quantile": float(boundary_quantile),
        "extrapolation_ratio_target": float(extrapolation_ratio),
        "boundary_thresholds": boundary_thresholds,
        "regular_tertile_thresholds": tertile_thresholds,
        "candidate_extrapolation_groups": int(len(candidate_groups)),
        "selected_extrapolation_groups": int(len(split_groups["extrapolation"])),
        "candidate_extrapolation_sequences": int(sum(profile["sequence_count"] for profile in candidate_groups)),
        "selected_extrapolation_sequences": int(sum(profile["sequence_count"] for profile in selected_profiles)),
        "boundary_reason_totals": dict(reason_totals),
        "boundary_reason_selected": dict(reason_selected),
        "total_sequences": int(total_sequences),
        "total_groups": int(total_groups),
        "splits": {
            name: {
                "sequence_count": int(len(split_rows[name])),
                "group_count": int(len(split_groups[name])),
                "group_coverage_ratio": float(len(split_groups[name]) / max(total_groups, 1)),
                "sequence_coverage_ratio": float(len(split_rows[name]) / max(total_sequences, 1)),
            }
            for name in split_rows
        },
        "stratify_distribution": {
            name: _split_field_stats([by_sequence_id[row["sequence_id"]] for row in split_rows[name]], stratify_fields)
            for name in split_rows
        },
    }


def _split_field_stats(rows, fields):
    stats = {}
    for field in fields:
        values = [float(row[field]) for row in rows]
        stats[field] = _value_stats(values)
    return stats


def compute_split_distribution(conditions, split_rows, label_fields):
    """统计每个 split 的序列数、mixture 数和标签分布（min/max/mean）。"""
    by_sequence_id = {row["sequence_id"]: row for row in conditions}
    distribution = {}
    for split_name, rows in split_rows.items():
        split_conditions = [by_sequence_id[row["sequence_id"]] for row in rows]
        distribution[split_name] = {
            "sequence_count": len(split_conditions),
            "mixture_count": len({row["sequence_id"] for row in split_conditions}),
        }
        for label_name in label_fields:
            values = [float(row[label_name]) for row in split_conditions]
            distribution[split_name][label_name] = _value_stats(values)
    return distribution


def _value_stats(values):
    if not values:
        return {"min": None, "max": None, "mean": None}
    return {
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def collect_split_warnings(distribution, label_fields):
    """检查 split 分布是否存在问题（空 split、某 split 缺少标签变化等）。"""
    warnings = []
    for split_name, stats in distribution.items():
        if stats["sequence_count"] == 0:
            warnings.append(f"{split_name} split has no sequences.")
        for label_name in label_fields:
            label_stats = stats[label_name]
            if label_stats["min"] is None or label_stats["max"] is None:
                continue
            if label_stats["min"] == label_stats["max"]:
                warnings.append(f"{split_name} split has no {label_name} range.")
    return warnings
