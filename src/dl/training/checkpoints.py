from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


REQUIRED_CHECKPOINT_KEYS = [
    "format_version",
    "epoch",
    "total_epochs",
    "model_state_dict",
    "optimizer_state_dict",
    "amp_scaler_state_dict",
    "early_stopping",
    "log_rows",
    "rng_state",
]
CHECKPOINT_DATA_KEYS = [
    "dataset_type",
    "npz_path",
    "index_path",
    "split_dir",
    "split_strategy",
    "input_format",
    "time_window",
    "channels",
    "acoustic_feature_path",
    "acoustic_features",
]


@dataclass
class CheckpointBuildContext:
    model: Any
    optimizer: Any
    scaler: Any
    stopper: Any
    epoch: int
    total_epochs: int
    log_rows: list
    best_path: Path
    config_to_write: dict
    status: str = "running"
    label_names: list | None = None
    scheduler: Any = None


def _capture_rng_state() -> dict:
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    numpy_state = np.random.get_state()
    return {
        "torch": torch.get_rng_state(),
        "cuda": cuda_state,
        "numpy": {
            "bit_generator": numpy_state[0],
            "state": torch.from_numpy(numpy_state[1].copy()),
            "pos": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
    }


def _restore_rng_state(state: dict) -> None:
    t_state = state["torch"]
    if t_state.device.type != "cpu":
        t_state = t_state.cpu()
    torch.set_rng_state(t_state.to(torch.uint8))
    if state.get("cuda") and torch.cuda.is_available():
        cuda_state = state["cuda"]
        if isinstance(cuda_state, list):
            cuda_state = [s.cpu().to(torch.uint8) for s in cuda_state]
        torch.cuda.set_rng_state_all(cuda_state)
    numpy_data = state.get("numpy")
    if numpy_data is None:
        return
    if isinstance(numpy_data, tuple):
        np.random.set_state(numpy_data)
        return
    numpy_values = numpy_data["state"]
    if torch.is_tensor(numpy_values):
        if numpy_values.device.type != "cpu":
            numpy_values = numpy_values.cpu()
        numpy_values = numpy_values.numpy()
    np.random.set_state(
        (
            numpy_data["bit_generator"],
            numpy_values,
            numpy_data["pos"],
            numpy_data["has_gauss"],
            numpy_data["cached_gaussian"],
        )
    )


def _state_dict_to_cpu(state_dict: dict) -> dict:
    output = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            output[key] = value.cpu()
        elif isinstance(value, dict):
            output[key] = _state_dict_to_cpu(value)
        else:
            output[key] = value
    return output


def _checkpoint_payload(context: CheckpointBuildContext) -> dict:
    model_state = _state_dict_to_cpu(context.model.state_dict())
    optimizer_state = _state_dict_to_cpu(context.optimizer.state_dict())
    return {
        "format_version": 1,
        "status": context.status,
        "epoch": context.epoch,
        "total_epochs": context.total_epochs,
        "config": context.config_to_write,
        "model_name": context.config_to_write.get("model", {}).get("name"),
        "label_names": context.label_names,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer_state,
        "amp_scaler_state_dict": context.scaler.state_dict(),
        "scheduler_state_dict": context.scheduler.state_dict() if context.scheduler is not None else None,
        "early_stopping": {
            "best": context.stopper.best,
            "bad_epochs": context.stopper.bad_epochs,
            "patience": context.stopper.patience,
            "mode": context.stopper.mode,
        },
        "log_rows": context.log_rows,
        "best_metric": context.stopper.best,
        "best_model_path": str(context.best_path),
        "rng_state": _capture_rng_state(),
    }


def _save_checkpoint(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def _load_checkpoint(path: Path, device: torch.device) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
    except Exception as exc:
        raise ValueError(
            f"Checkpoint cannot be loaded safely with weights_only=True: {path}. "
            "Only load checkpoints produced by the current safe checkpoint format."
        ) from exc
    if not isinstance(ckpt, dict):
        raise ValueError(f"Checkpoint is not a dict: {type(ckpt)}")
    if ckpt.get("format_version") != 1:
        raise ValueError(f"Unsupported checkpoint format_version: {ckpt.get('format_version')}")
    return ckpt


def _restore_early_stopping(stopper, state: dict) -> None:
    stopper.best = state.get("best")
    stopper.bad_epochs = state.get("bad_epochs", 0)
    if "patience" in state:
        stopper.patience = state["patience"]
    if "mode" in state:
        stopper.mode = state["mode"]


def _compare_config_key(mismatches: list[str], ckpt_section: dict, cur_section: dict, prefix: str, key: str) -> None:
    sentinel = object()
    ckpt_value = ckpt_section.get(key, sentinel)
    cur_value = cur_section.get(key, sentinel)
    if ckpt_value == cur_value:
        return
    ckpt_display = "<missing>" if ckpt_value is sentinel else ckpt_value
    cur_display = "<missing>" if cur_value is sentinel else cur_value
    mismatches.append(f"{prefix}.{key}: checkpoint={ckpt_display}, config={cur_display}")


def _validate_model_name_match(ckpt: dict, config: dict) -> None:
    ckpt_model = ckpt.get("model_name")
    cur_model = config.get("model", {}).get("name")
    if ckpt_model and cur_model and ckpt_model != cur_model:
        raise ValueError(f"Model mismatch: checkpoint uses '{ckpt_model}', config uses '{cur_model}'")


def _validate_required_checkpoint_keys(ckpt: dict) -> None:
    missing = [key for key in REQUIRED_CHECKPOINT_KEYS if key not in ckpt]
    if missing:
        raise ValueError(f"Checkpoint missing required keys: {missing}")


def _collect_data_config_mismatches(mismatches: list[str], ckpt_config: dict, config: dict) -> None:
    ckpt_data = ckpt_config.get("data", {})
    cur_data = config.get("data", {})
    for key in CHECKPOINT_DATA_KEYS:
        _compare_config_key(mismatches, ckpt_data, cur_data, "data", key)


def _collect_model_config_mismatches(mismatches: list[str], ckpt_config: dict, config: dict) -> None:
    ckpt_model_config = ckpt_config.get("model", {})
    cur_model_config = config.get("model", {})
    model_keys = sorted(set(ckpt_model_config).union(cur_model_config).difference({"name"}))
    for key in model_keys:
        _compare_config_key(mismatches, ckpt_model_config, cur_model_config, "model", key)


def _collect_label_mismatches(mismatches: list[str], ckpt: dict, label_names: list | None) -> None:
    ckpt_labels = ckpt.get("label_names")
    if ckpt_labels is None or label_names is None:
        return
    if len(ckpt_labels) != len(label_names):
        mismatches.append(f"out_dim: checkpoint={len(ckpt_labels)}, config={len(label_names)}")
    elif ckpt_labels != label_names:
        mismatches.append(f"label_names 顺序不一致: checkpoint={ckpt_labels}, config={label_names}")


def _collect_scheduler_mismatches(mismatches: list[str], ckpt_config: dict, config: dict) -> None:
    ckpt_sched = ckpt_config.get("training", {}).get("lr_scheduler")
    cur_sched = config.get("training", {}).get("lr_scheduler")
    if ckpt_sched == cur_sched:
        return
    ckpt_st = ckpt_sched.get("type") if ckpt_sched else None
    cur_st = cur_sched.get("type") if cur_sched else None
    mismatches.append(f"lr_scheduler.type: checkpoint={ckpt_st}, config={cur_st}")


def _raise_checkpoint_mismatches(mismatches: list[str]) -> None:
    if mismatches:
        raise ValueError(
            "Checkpoint/config 不兼容。checkpoint 使用了不同的训练设置，"
            "恢复训练将导致实验结果不可比。请使用匹配的配置文件或从头训练。\n"
            "差异项:\n" + "\n".join(f"  - {m}" for m in mismatches)
        )


def _validate_model_architecture_compat(ckpt: dict, model_state_keys: set[str]) -> None:
    """校验 checkpoint 中的 model_state_dict 键集合与当前模型是否一致。

    架构升级（如新增 BN、投影头参数）后，旧 checkpoint 的键集合会与当前模型不匹配，
    load_state_dict 会因键名/数量不一致报错。此函数在恢复前提前检测，给出更明确的提示。
    """
    ckpt_state = ckpt.get("model_state_dict")
    if ckpt_state is None:
        return
    ckpt_keys = set(ckpt_state.keys())
    if ckpt_keys == model_state_keys:
        return
    only_in_ckpt = ckpt_keys - model_state_keys
    only_in_model = model_state_keys - ckpt_keys
    parts: list[str] = ["Checkpoint model_state_dict 与当前模型架构不兼容。"]
    if only_in_ckpt:
        parts.append(f"  checkpoint 独有（当前模型已移除）: {sorted(only_in_ckpt)}")
    if only_in_model:
        parts.append(f"  当前模型独有（checkpoint 中缺失）: {sorted(only_in_model)}")
    parts.append("模型架构已变更，请从头训练或使用匹配的 checkpoint。")
    raise ValueError("\n".join(parts))


def _validate_checkpoint_compat(
    ckpt: dict, config: dict, label_names: list | None = None, model_state_keys: set[str] | None = None
) -> None:
    _validate_model_name_match(ckpt, config)
    _validate_required_checkpoint_keys(ckpt)
    if model_state_keys is not None:
        _validate_model_architecture_compat(ckpt, model_state_keys)
    ckpt_config = ckpt.get("config", {})
    if not ckpt_config:
        return
    mismatches: list[str] = []
    _collect_data_config_mismatches(mismatches, ckpt_config, config)
    _collect_model_config_mismatches(mismatches, ckpt_config, config)
    _collect_label_mismatches(mismatches, ckpt, label_names)
    _collect_scheduler_mismatches(mismatches, ckpt_config, config)
    _raise_checkpoint_mismatches(mismatches)
