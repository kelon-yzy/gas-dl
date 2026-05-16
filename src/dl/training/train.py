from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.channel_groups import CHANNEL_GROUPS, EXPECTED_CHANNEL_NAMES, resolve_channel_indices, resolve_time_indices
from data.dataset_v2 import (
    V2SequenceDataset,
    load_acoustic_feature_array,
    load_sequence_metadata,
    load_v2_npz,
    resolve_dataset_channel_indices,
)
from data.dataset_waveform import WaveformSequenceDataset, load_waveform_package
from data.scaler_utils import load_or_fit_scaler
from data.split_utils import generate_group_splits, load_existing_splits, split_indices_from_frames
from models.registry import build_model
from training.early_stopping import EarlyStopping
from training.losses import build_loss
from training.metrics import regression_metrics, _display_labels
from training.seed import set_seed


def resolve_path(value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _ensure_scaler_path(data_config: dict, output_dir: Path) -> None:
    if data_config.get("scaler_path"):
        return
    filename = "scaler_slow_sequence.json" if data_config.get("dataset_type") == "waveform_v3" else "scaler_sequence.json"
    data_config["scaler_path"] = str(output_dir / filename)


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def make_loader(dataset, batch_size: int, shuffle: bool, device: torch.device | None = None, num_workers: int = 0):
    pin_memory = device is not None and device.type == "cuda"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )


def _is_waveform_batch(batch) -> bool:
    return isinstance(batch, dict)


def _move_tensor(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    return tensor.to(device, non_blocking=device.type == "cuda")


def _use_amp(training_config: dict, device: torch.device) -> bool:
    return bool(training_config.get("amp", False)) and device.type == "cuda"


def _amp_context(device: torch.device, enabled: bool):
    if enabled:
        return torch.amp.autocast(device_type=device.type)
    return nullcontext()


def _move_waveform_batch(batch: dict, device: torch.device) -> dict:
    return {
        "ultrasonic": _move_tensor(batch["ultrasonic"], device),
        "ultrasonic_scale": _move_tensor(batch["ultrasonic_scale"], device),
        "fiber_mic": _move_tensor(batch["fiber_mic"], device),
        "fiber_mic_scale": _move_tensor(batch["fiber_mic_scale"], device),
        "slow": _move_tensor(batch["slow"], device),
        "target": _move_tensor(batch["target"], device),
        "meta": batch["meta"],
    }


def _forward_batch(model, batch, device):
    if _is_waveform_batch(batch):
        moved = _move_waveform_batch(batch, device)
        if getattr(model, "use_waveform", False):
            pred = model(moved["ultrasonic"], moved["ultrasonic_scale"], moved["fiber_mic"], moved["fiber_mic_scale"], moved["slow"])
        else:
            slow_input = moved["slow"]
            if getattr(model, "input_format", "NTC").upper() == "NCT":
                slow_input = slow_input.transpose(1, 2)
            pred = model(slow_input)
        return pred, moved["target"], moved["meta"]
    x, y, meta = batch
    x = _move_tensor(x, device)
    y = _move_tensor(y, device)
    pred = model(x)
    return pred, y, meta


def _metadata_to_frame(meta) -> pd.DataFrame:
    normalized = {}
    for key, value in meta.items():
        if torch.is_tensor(value):
            normalized[key] = value.cpu().numpy().tolist()
        else:
            normalized[key] = list(value)
    return pd.DataFrame(normalized)


def apply_environment_augmentation(x: torch.Tensor, input_format: str, sigma: float) -> torch.Tensor:
    if sigma <= 0.0:
        return x
    out = x.clone()
    env_indices = CHANNEL_GROUPS["environment"]
    input_format = input_format.upper()
    if input_format == "NTC":
        out[:, :, env_indices] = out[:, :, env_indices] + torch.randn_like(out[:, :, env_indices]) * sigma
        return out
    if input_format == "NCT":
        out[:, env_indices, :] = out[:, env_indices, :] + torch.randn_like(out[:, env_indices, :]) * sigma
        return out
    raise ValueError(f"Unknown input_format: {input_format}")


def predict(model, loader, device):
    model.eval()
    y_true, y_pred, frames = [], [], []
    with torch.no_grad():
        for batch in loader:
            pred, y, meta = _forward_batch(model, batch, device)
            pred = pred.cpu().numpy()
            y_pred.append(pred)
            y_true.append(y.cpu().numpy())
            frames.append(_metadata_to_frame(meta))
    return PredictionBundle(
        y_true=np.vstack(y_true),
        y_pred=np.vstack(y_pred),
        meta=pd.concat(frames, ignore_index=True),
    )


def evaluate_with_predictions(model, loader, loss_fn, device) -> tuple[float, PredictionBundle]:
    model.eval()
    losses = []
    y_true, y_pred, frames = [], [], []
    with torch.no_grad():
        for batch in loader:
            pred, y, meta = _forward_batch(model, batch, device)
            losses.append(float(loss_fn(pred, y).item()))
            y_pred.append(pred.cpu().numpy())
            y_true.append(y.cpu().numpy())
            frames.append(_metadata_to_frame(meta))
    return (
        float(np.mean(losses)),
        PredictionBundle(
            y_true=np.vstack(y_true),
            y_pred=np.vstack(y_pred),
            meta=pd.concat(frames, ignore_index=True),
        ),
    )


def evaluate_loss(model, loader, loss_fn, device) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            pred, y, _ = _forward_batch(model, batch, device)
            losses.append(float(loss_fn(pred, y).item()))
    return float(np.mean(losses))


@dataclass(frozen=True)
class PredictionBundle:
    y_true: np.ndarray
    y_pred: np.ndarray
    meta: pd.DataFrame


def _build_waveform_datasets(data_config: dict, seed: int):
    npz_path = resolve_path(data_config["npz_path"])
    index_path = resolve_path(data_config.get("index_path", "../simulation-data/output_sequence/sequence_index.csv"))
    split_dir = resolve_path(data_config.get("split_dir", "../simulation-data/output_sequence/splits"))
    scaler_path = resolve_path(data_config.get("scaler_path"))
    time_indices = resolve_time_indices(data_config.get("time_window", "all"))

    data = load_waveform_package(npz_path)
    slow = data["slow"].astype(np.float32)
    sequence_ids = [str(v) for v in data.get("sequence_ids", np.arange(len(slow)))]
    metadata = load_sequence_metadata(index_path, sequence_ids)

    splits = None
    if data_config.get("split_strategy", "existing_or_group_mixture") == "existing_or_group_mixture":
        splits = load_existing_splits(split_dir) if split_dir is not None else None
    if splits is None:
        splits = generate_group_splits(metadata, seed=seed)
    split_indices = split_indices_from_frames(splits, sequence_ids)

    slow_train = slow[split_indices["train"]]
    if time_indices is not None:
        slow_train = slow_train[:, time_indices, :]
    slow_scaler = load_or_fit_scaler(scaler_path, slow_train, channel_names=list(data["slow_channel_names"]))

    common = {
        "npz_path": npz_path,
        "slow_scaler": slow_scaler,
        "time_indices": time_indices,
        "index_path": index_path,
        "preloaded_data": data,
    }
    datasets = {name: WaveformSequenceDataset(indices=indices, **common) for name, indices in split_indices.items()}
    return datasets, splits


def _build_v2_datasets(data_config: dict, seed: int):
    npz_path = resolve_path(data_config["npz_path"])
    index_path = resolve_path(data_config.get("index_path", "../simulation-data/output_sequence/sequence_index.csv"))
    split_dir = resolve_path(data_config.get("split_dir", "../simulation-data/output_sequence/splits"))
    scaler_path = resolve_path(data_config.get("scaler_path"))
    input_format = data_config.get("input_format", "NTC")
    time_indices = resolve_time_indices(data_config.get("time_window", "all"))
    acoustic_feature_path = resolve_path(data_config.get("acoustic_feature_path"))
    acoustic_features = data_config.get("acoustic_features") or []

    data = load_v2_npz(npz_path)
    X = data["X"].astype(np.float32)
    sequence_ids = [str(v) for v in data.get("sequence_ids", np.arange(len(X)))]
    channel_names = list(EXPECTED_CHANNEL_NAMES)
    if acoustic_features:
        acoustic = load_acoustic_feature_array(acoustic_feature_path, sequence_ids, X.shape[1], list(acoustic_features))
        X = np.concatenate([X, acoustic], axis=2)
        channel_names.extend(acoustic_features)
    channel_indices = resolve_dataset_channel_indices(data_config.get("channels", "all"), channel_names)
    metadata = load_sequence_metadata(index_path, sequence_ids)

    splits = None
    if data_config.get("split_strategy", "existing_or_group_mixture") == "existing_or_group_mixture":
        splits = load_existing_splits(split_dir)
    if splits is None:
        splits = generate_group_splits(metadata, seed=seed)
    split_indices = split_indices_from_frames(splits, sequence_ids)

    X_train = X[split_indices["train"]]
    if time_indices is not None:
        X_train = X_train[:, time_indices, :]
    scaler = load_or_fit_scaler(scaler_path, X_train, channel_names=channel_names)
    dataset_scaler = scaler.subset(channel_indices)

    common = {
        "npz_path": npz_path,
        "scaler": dataset_scaler,
        "input_format": input_format,
        "channel_indices": channel_indices,
        "time_indices": time_indices,
        "index_path": index_path,
        "acoustic_feature_path": acoustic_feature_path,
        "acoustic_features": acoustic_features,
        "preloaded_data": data,
    }
    datasets = {name: V2SequenceDataset(indices=indices, **common) for name, indices in split_indices.items()}
    return datasets, splits


def build_datasets(config: dict):
    data_config = config["data"]
    dataset_type = data_config.get("dataset_type", "v2")
    seed = int(config["run"].get("seed", 42))
    if dataset_type == "waveform_v3":
        return _build_waveform_datasets(data_config, seed)
    return _build_v2_datasets(data_config, seed)


def _load_label_names(config: dict):
    data_config = config["data"]
    npz_path = resolve_path(data_config["npz_path"])
    dataset_type = data_config.get("dataset_type", "v2")
    if dataset_type == "waveform_v3":
        data_preview = load_waveform_package(npz_path)
    else:
        data_preview = load_v2_npz(npz_path)
    return [str(value) for value in data_preview["label_names"]]


def _train_one_epoch(
    model,
    loader,
    loss_fn,
    optimizer,
    device,
    dataset,
    env_aug_sigma: float,
    amp_enabled: bool = False,
    scaler: torch.amp.GradScaler | None = None,
):
    model.train()
    losses = []
    if scaler is None:
        scaler = torch.amp.GradScaler("cuda", enabled=False)
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        if not _is_waveform_batch(batch) and env_aug_sigma > 0.0:
            x, y, _ = batch
            x = _move_tensor(x, device)
            y = _move_tensor(y, device)
            x = apply_environment_augmentation(x, dataset.input_format, env_aug_sigma)
            with _amp_context(device, amp_enabled):
                pred = model(x)
                loss = loss_fn(pred, y)
        else:
            with _amp_context(device, amp_enabled):
                pred, y, _ = _forward_batch(model, batch, device)
                loss = loss_fn(pred, y)
        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses))


def save_predictions(path: Path, meta: pd.DataFrame, y_true, y_pred, split: str, label_names) -> None:
    out = pd.DataFrame()
    out["sample_id"] = meta.get("sample_id", meta.get("sequence_id")).astype(str)
    out["mixture_id"] = meta.get("mixture_id", out["sample_id"]).astype(str)
    out["split"] = split
    labels = _display_labels(label_names)
    for i, label in enumerate(labels):
        out[f"y_true_{label}"] = y_true[:, i]
    for i, label in enumerate(labels):
        out[f"y_pred_{label}"] = y_pred[:, i]
    out["sum_true"] = y_true.sum(axis=1)
    out["sum_pred"] = y_pred.sum(axis=1)
    out["abs_sum_error"] = (out["sum_pred"] - out["sum_true"]).abs()
    out.to_csv(path, index=False)


# ── checkpoint helpers ──────────────────────────────────────────

def _capture_rng_state() -> dict:
    """捕获当前所有随机数生成器状态（weights_only 兼容格式：仅含 Tensor 与基础类型）"""
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
    """恢复随机数生成器状态（兼容旧版 tuple 格式和新版 dict 格式）"""
    t_state = state["torch"]
    if t_state.device.type != "cpu":
        t_state = t_state.cpu()
    torch.set_rng_state(t_state)
    if state.get("cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    numpy_data = state.get("numpy")
    if numpy_data is None:
        return
    if isinstance(numpy_data, tuple):
        # 旧版格式：直接是 (str, ndarray, int, int, float)
        np.random.set_state(numpy_data)
    else:
        # 新版格式：dict with tensor
        numpy_values = numpy_data["state"]
        if torch.is_tensor(numpy_values):
            if numpy_values.device.type != "cpu":
                numpy_values = numpy_values.cpu()
            numpy_values = numpy_values.numpy()
        np.random.set_state((
            numpy_data["bit_generator"],
            numpy_values,
            numpy_data["pos"],
            numpy_data["has_gauss"],
            numpy_data["cached_gaussian"],
        ))


def _checkpoint_payload(
    model,
    optimizer,
    scaler,
    stopper,
    epoch: int,
    total_epochs: int,
    log_rows: list,
    best_path: Path,
    config_to_write: dict,
    status: str = "running",
    label_names: list | None = None,
) -> dict:
    """组装 checkpoint dict，模型参数移至 CPU 以支持跨设备恢复"""
    model_state = {}
    for k, v in model.state_dict().items():
        model_state[k] = v.cpu()
    optim_state = {}
    for k, v in optimizer.state_dict().items():
        if isinstance(v, torch.Tensor):
            optim_state[k] = v.cpu()
        elif isinstance(v, dict):
            optim_state[k] = {kk: vv.cpu() if isinstance(vv, torch.Tensor) else vv for kk, vv in v.items()}
        else:
            optim_state[k] = v
    return {
        "format_version": 1,
        "status": status,
        "epoch": epoch,
        "total_epochs": total_epochs,
        "config": config_to_write,
        "model_name": config_to_write.get("model", {}).get("name"),
        "label_names": label_names,
        "model_state_dict": model_state,
        "optimizer_state_dict": optim_state,
        "amp_scaler_state_dict": scaler.state_dict(),
        "early_stopping": {
            "best": stopper.best,
            "bad_epochs": stopper.bad_epochs,
            "patience": stopper.patience,
            "mode": stopper.mode,
        },
        "log_rows": log_rows,
        "best_metric": stopper.best,
        "best_model_path": str(best_path),
        "rng_state": _capture_rng_state(),
    }


def _save_checkpoint(path: Path, payload: dict) -> None:
    """原子保存：先写临时文件再 rename，避免写入中断产生半文件"""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def _load_checkpoint(path: Path, device: torch.device) -> dict:
    """安全加载 checkpoint；拒绝需要 pickle 反序列化的旧格式。"""
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


def _restore_early_stopping(stopper: "EarlyStopping", state: dict) -> None:
    """恢复 EarlyStopping 内部状态"""
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


def _validate_checkpoint_compat(ckpt: dict, config: dict, label_names: list | None = None) -> None:
    """校验 checkpoint 与当前配置的兼容性：模型名、数据集类型、时间窗口、波形模式、输出维度、标签名"""
    ckpt_model = ckpt.get("model_name")
    cur_model = config.get("model", {}).get("name")
    if ckpt_model and cur_model and ckpt_model != cur_model:
        raise ValueError(
            f"Model mismatch: checkpoint uses '{ckpt_model}', config uses '{cur_model}'"
        )
    required = [
        "format_version", "epoch", "total_epochs", "model_state_dict",
        "optimizer_state_dict", "amp_scaler_state_dict", "early_stopping",
        "log_rows", "rng_state",
    ]
    missing = [k for k in required if k not in ckpt]
    if missing:
        raise ValueError(f"Checkpoint missing required keys: {missing}")

    # ── 深度兼容性校验：防止静默污染实验结果 ──
    ckpt_config = ckpt.get("config", {})
    if not ckpt_config:
        return  # 旧版 checkpoint 无 config，跳过深度校验

    mismatches = []

    ckpt_data = ckpt_config.get("data", {})
    cur_data = config.get("data", {})
    for key in (
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
    ):
        _compare_config_key(mismatches, ckpt_data, cur_data, "data", key)

    ckpt_model_config = ckpt_config.get("model", {})
    cur_model_config = config.get("model", {})
    model_keys = sorted(set(ckpt_model_config).union(cur_model_config).difference({"name"}))
    for key in model_keys:
        _compare_config_key(mismatches, ckpt_model_config, cur_model_config, "model", key)

    # 输出维度（通过 label_names 长度）
    ckpt_labels = ckpt.get("label_names")
    if ckpt_labels is not None and label_names is not None:
        if len(ckpt_labels) != len(label_names):
            mismatches.append(f"out_dim: checkpoint={len(ckpt_labels)}, config={len(label_names)}")
        elif ckpt_labels != label_names:
            mismatches.append(f"label_names 顺序不一致: checkpoint={ckpt_labels}, config={label_names}")

    if mismatches:
        raise ValueError(
            "Checkpoint/config 不兼容。checkpoint 使用了不同的训练设置，"
            "恢复训练将导致实验结果不可比。请使用匹配的配置文件或从头训练。\n"
            "差异项:\n" + "\n".join(f"  - {m}" for m in mismatches)
        )


# ── main training ──────────────────────────────────────────────

def train_config(
    config: dict,
    epochs_override: int | None = None,
    resume_path: str | Path | None = None,
    checkpoint_every: int = 0,
    restore_rng: bool = True,
    stop_after_epoch: int | None = None,
) -> dict:
    if epochs_override is not None:
        config["training"]["epochs"] = epochs_override

    set_seed(int(config["run"].get("seed", 42)))
    output_dir = resolve_path(config["run"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    _ensure_scaler_path(config["data"], output_dir)

    label_names = _load_label_names(config)
    datasets, splits = build_datasets(config)
    batch_size = int(config["training"].get("batch_size", 64))
    device = select_device(config["training"].get("device", "auto"))
    num_workers = int(config["training"].get("num_workers", 0))
    eval_num_workers = int(config["training"].get("eval_num_workers", 0))
    amp_enabled = _use_amp(config["training"], device)
    loaders = {
        "train": make_loader(datasets["train"], batch_size, shuffle=True, device=device, num_workers=num_workers),
        "val": make_loader(datasets["val"], batch_size, shuffle=False, device=device, num_workers=eval_num_workers),
        "test": make_loader(datasets["test"], batch_size, shuffle=False, device=device, num_workers=eval_num_workers),
    }

    model = build_model(config["model"]).to(device)
    loss_fn = build_loss(
        config["training"].get("loss", "mse"),
        sum_constraint=config["training"].get("sum_constraint"),
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["training"].get("learning_rate", 1e-3)),
        weight_decay=float(config["training"].get("weight_decay", 0.0)),
    )
    stopper = EarlyStopping(patience=int(config["training"].get("early_stopping_patience", 25)), mode="min")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    total_epochs = int(config["training"].get("epochs", 200))

    # ── 恢复训练 ──
    start_epoch = 1
    log_rows = []
    best_path = output_dir / "best_model.pt"
    last_ckpt_path = output_dir / "last_checkpoint.pt"
    best_ckpt_path = output_dir / "best_checkpoint.pt"
    paused_ckpt_path = output_dir / "paused_checkpoint.pt"
    initial_ckpt_path = output_dir / "initial_checkpoint.pt"
    resumed_from = None

    progress = config.get("_cli_progress")
    if resume_path:
        resume_path = Path(resume_path)
        ckpt = _load_checkpoint(resume_path, device)
        _validate_checkpoint_compat(ckpt, config, label_names)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scaler.load_state_dict(ckpt["amp_scaler_state_dict"])
        _restore_early_stopping(stopper, ckpt.get("early_stopping", {}))
        if restore_rng:
            _restore_rng_state(ckpt["rng_state"])
        log_rows = ckpt.get("log_rows", [])
        start_epoch = ckpt["epoch"] + 1
        resumed_from = str(resume_path)
        if start_epoch > total_epochs:
            if progress is not None:
                progress.log_message(f"checkpoint epoch {ckpt['epoch']} >= target {total_epochs}, 跳过训练直接评估")
            start_epoch = total_epochs + 1

    if progress is not None:
        progress.start_run(mode="deep", title=config["run"]["name"], seed=int(config["run"].get("seed", 42)), stage="setup")
        progress.update_metric(
            model=config["model"]["name"],
            device=str(device),
            batch_size=batch_size,
            n_train=len(datasets["train"]),
            n_val=len(datasets["val"]),
            n_test=len(datasets["test"]),
        )

    config_to_write = {k: v for k, v in config.items() if not k.startswith("_")}
    training_status = "completed"
    paused = False

    # 训练开始前保存初始状态快照，供首 epoch Ctrl+C 回滚使用
    run_start_epoch = start_epoch - 1
    run_start_log_rows = list(log_rows)
    _save_checkpoint(initial_ckpt_path, _checkpoint_payload(
        model, optimizer, scaler, stopper, run_start_epoch, total_epochs,
        run_start_log_rows, best_path, config_to_write, status="initial",
        label_names=label_names,
    ))
    current_run_last_checkpoint_written = False

    try:
        for epoch in range(start_epoch, total_epochs + 1):
            epoch_started = perf_counter()
            train_loss = _train_one_epoch(
                model,
                loaders["train"],
                loss_fn,
                optimizer,
                device,
                datasets["train"],
                float(config["training"].get("environment_augmentation_sigma", 0.0)),
                amp_enabled=amp_enabled,
                scaler=scaler,
            )

            val_loss, val_bundle = evaluate_with_predictions(model, loaders["val"], loss_fn, device)
            val_summary, _ = regression_metrics(val_bundle.y_true, val_bundle.y_pred, label_names=label_names)
            monitor_value = val_summary["macro_RMSE"]
            improved = stopper.step(monitor_value)

            # 先构造并 append row，确保后续 checkpoint 包含当前 epoch 日志
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_macro_RMSE": monitor_value,
                "val_macro_MAE": val_summary["macro_MAE"],
                "improved": improved,
            }
            log_rows.append(row)

            if improved:
                torch.save(model.state_dict(), best_path)
                _save_checkpoint(best_ckpt_path, _checkpoint_payload(
                    model, optimizer, scaler, stopper, epoch, total_epochs,
                    log_rows, best_path, config_to_write, status="running",
                    label_names=label_names,
                ))

            if progress is not None:
                progress.update_stage(stage="epoch", current_task=f"epoch={epoch}/{total_epochs}", completed=epoch, total=total_epochs)
                progress.update_metric(
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    val_macro_RMSE=monitor_value,
                    improved=improved,
                    best=stopper.best,
                    bad_epochs=stopper.bad_epochs,
                    patience=stopper.patience,
                    epoch_seconds=perf_counter() - epoch_started,
                )

            # 每个 epoch 后保存 last_checkpoint
            _save_checkpoint(last_ckpt_path, _checkpoint_payload(
                model, optimizer, scaler, stopper, epoch, total_epochs,
                log_rows, best_path, config_to_write, status="running",
                label_names=label_names,
            ))
            current_run_last_checkpoint_written = True

            # 周期快照 epoch_XXXX.pt
            if checkpoint_every > 0 and epoch % checkpoint_every == 0:
                _save_checkpoint(output_dir / f"epoch_{epoch:04d}.pt", _checkpoint_payload(
                    model, optimizer, scaler, stopper, epoch, total_epochs,
                    log_rows, best_path, config_to_write, status="running",
                    label_names=label_names,
                ))

            # 测试专用：训练到指定 epoch 后主动暂停
            if stop_after_epoch is not None and epoch >= stop_after_epoch:
                training_status = "paused"
                paused = True
                if progress is not None:
                    progress.log_message(f"stop_after_epoch={stop_after_epoch}, 暂停于 epoch {epoch}")
                break

            if stopper.should_stop:
                if progress is not None:
                    progress.log_message(f"early stop at epoch {epoch}")
                break

    except KeyboardInterrupt:
        training_status = "paused"
        paused = True
        if progress is not None:
            progress.log_message("用户中断 (Ctrl+C), 回退到最近完整 epoch 的干净权重")
        # 回退到最近完整 epoch 的干净权重，避免保存半轮训练状态
        # 优先 last_checkpoint，不存在则降级到 initial_checkpoint
        restore_source = None
        if current_run_last_checkpoint_written and last_ckpt_path.exists():
            restore_source = last_ckpt_path
        elif initial_ckpt_path.exists():
            restore_source = initial_ckpt_path
        if restore_source is not None:
            ckpt_clean = _load_checkpoint(restore_source, device)
            model.load_state_dict(ckpt_clean["model_state_dict"])
            optimizer.load_state_dict(ckpt_clean["optimizer_state_dict"])
            scaler.load_state_dict(ckpt_clean["amp_scaler_state_dict"])

    # ── 暂停退出：写部分输出，不跑测试集 ──
    if paused:
        last_epoch = log_rows[-1]["epoch"] if log_rows else (start_epoch - 1)
        _save_checkpoint(paused_ckpt_path, _checkpoint_payload(
            model, optimizer, scaler, stopper, last_epoch, total_epochs,
            log_rows, best_path, config_to_write, status="paused",
            label_names=label_names,
        ))
        summary = {
            "run_name": config["run"]["name"],
            "model": config["model"]["name"],
            "label_names": label_names,
            "seed": int(config["run"].get("seed", 42)),
            "training_status": "paused",
            "resumed_from": resumed_from,
            "last_checkpoint": str(last_ckpt_path),
            "best_checkpoint": str(best_ckpt_path),
            "epochs_trained": int(log_rows[-1]["epoch"]) if log_rows else 0,
        }
        if progress is not None:
            progress.finish_run(status="paused", epochs_trained=summary["epochs_trained"])
        (output_dir / "config.json").write_text(json.dumps(config_to_write, indent=2, ensure_ascii=False), encoding="utf-8")
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        pd.DataFrame(log_rows).to_csv(output_dir / "train_log.csv", index=False)
        return summary

    # ── 正常完成 ──
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))

    test_bundle = predict(model, loaders["test"], device)
    summary, component_metrics = regression_metrics(test_bundle.y_true, test_bundle.y_pred, label_names=label_names)
    summary.update(
        {
            "run_name": config["run"]["name"],
            "model": config["model"]["name"],
            "label_names": label_names,
            "seed": int(config["run"].get("seed", 42)),
            "device": str(device),
            "amp": bool(amp_enabled),
            "batch_size": batch_size,
            "num_workers": num_workers,
            "eval_num_workers": eval_num_workers,
            "epochs_trained": int(log_rows[-1]["epoch"]) if log_rows else 0,
            "n_train": int(len(datasets["train"])),
            "n_val": int(len(datasets["val"])),
            "n_test": int(len(datasets["test"])),
            "training_status": "completed",
            "resumed_from": resumed_from,
            "last_checkpoint": str(last_ckpt_path),
            "best_checkpoint": str(best_ckpt_path),
        }
    )
    if "use_waveform" in config.get("model", {}):
        summary["use_waveform"] = bool(config["model"]["use_waveform"])

    if progress is not None:
        progress.finish_run(status="done", macro_RMSE=summary["macro_RMSE"], epochs_trained=summary["epochs_trained"])

    # 正常完成后保存 completed 状态的 last_checkpoint
    final_epoch = int(log_rows[-1]["epoch"]) if log_rows else 0
    _save_checkpoint(last_ckpt_path, _checkpoint_payload(
        model, optimizer, scaler, stopper, final_epoch, total_epochs,
        log_rows, best_path, config_to_write, status="completed",
        label_names=label_names,
    ))

    (output_dir / "config.json").write_text(json.dumps(config_to_write, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    component_metrics.to_csv(output_dir / "component_metrics.csv", index=False)
    pd.DataFrame(log_rows).to_csv(output_dir / "train_log.csv", index=False)
    save_predictions(
        output_dir / "predictions.csv",
        test_bundle.meta,
        test_bundle.y_true,
        test_bundle.y_pred,
        split="test",
        label_names=label_names,
    )
    return summary


def train_one(
    config_path: str | Path,
    epochs_override: int | None = None,
    progress=None,
    resume_path: str | Path | None = None,
    checkpoint_every: int = 0,
    restore_rng: bool = True,
    stop_after_epoch: int | None = None,
) -> dict:
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    if progress is not None:
        config["_cli_progress"] = progress
    return train_config(
        config,
        epochs_override=epochs_override,
        resume_path=resume_path,
        checkpoint_every=checkpoint_every,
        restore_rng=restore_rng,
        stop_after_epoch=stop_after_epoch,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    summary = train_one(args.config, epochs_override=args.epochs)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
