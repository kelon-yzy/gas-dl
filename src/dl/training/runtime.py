from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data.channel_groups import CHANNEL_GROUPS
from training.metrics import _display_labels


@dataclass(frozen=True)
class PredictionBundle:
    y_true: np.ndarray
    y_pred: np.ndarray
    meta: pd.DataFrame


@dataclass
class TrainEpochRequest:
    model: Any
    loader: Any
    loss_fn: Any
    optimizer: Any
    device: torch.device
    dataset: Any
    env_aug_sigma: float
    amp_enabled: bool = False
    scaler: torch.amp.GradScaler | None = None
    grad_clip_norm: float = 0.0


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def configure_cudnn(training_config: dict, device: torch.device) -> None:
    """按配置开启 cuDNN benchmark。模型输入形状固定时可加速 5-15%。"""
    if device.type != "cuda":
        return
    if bool(training_config.get("cudnn_benchmark", False)):
        torch.backends.cudnn.benchmark = True


def make_loader(
    dataset,
    batch_size: int,
    shuffle: bool,
    device: torch.device | None = None,
    num_workers: int = 0,
    prefetch_factor: int | None = None,
):
    pin_memory = device is not None and device.type == "cuda"
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }
    # prefetch_factor 在 num_workers=0 下传入会报错，仅多 worker 场景启用
    if num_workers > 0 and prefetch_factor is not None and prefetch_factor > 0:
        kwargs["prefetch_factor"] = int(prefetch_factor)
    return DataLoader(dataset, **kwargs)


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
    moved = {
        "ultrasonic": _move_tensor(batch["ultrasonic"], device),
        "ultrasonic_scale": _move_tensor(batch["ultrasonic_scale"], device),
        "fiber_mic": _move_tensor(batch["fiber_mic"], device),
        "fiber_mic_scale": _move_tensor(batch["fiber_mic_scale"], device),
        "slow": _move_tensor(batch["slow"], device),
        "target": _move_tensor(batch["target"], device),
        "meta": batch["meta"],
    }
    return moved


def _forward_batch(model, batch, device):
    if _is_waveform_batch(batch):
        moved = _move_waveform_batch(batch, device)
        if getattr(model, "use_waveform", False):
            pred = model(
                moved["ultrasonic"],
                moved["ultrasonic_scale"],
                moved["fiber_mic"],
                moved["fiber_mic_scale"],
                moved["slow"],
            )
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


def _mean_loss(total_loss: float, total_samples: int) -> float:
    if total_samples == 0:
        return float("nan")
    return float(total_loss / total_samples)


def _optimizer_parameters(optimizer) -> list[torch.nn.Parameter]:
    params = []
    seen = set()
    for group in optimizer.param_groups:
        for param in group["params"]:
            ident = id(param)
            if ident in seen:
                continue
            seen.add(ident)
            params.append(param)
    return params


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


def predict(model, loader, device) -> PredictionBundle:
    model.eval()
    preds_gpu, targets_gpu, frames = [], [], []
    with torch.no_grad():
        for batch in loader:
            pred, y, meta = _forward_batch(model, batch, device)
            preds_gpu.append(pred)
            targets_gpu.append(y)
            frames.append(_metadata_to_frame(meta))
    all_preds = torch.cat(preds_gpu, dim=0).cpu().numpy()
    all_targets = torch.cat(targets_gpu, dim=0).cpu().numpy()
    return PredictionBundle(
        y_true=all_targets,
        y_pred=all_preds,
        meta=pd.concat(frames, ignore_index=True),
    )


def evaluate_with_predictions(model, loader, loss_fn, device) -> tuple[float, PredictionBundle]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    preds_gpu, targets_gpu, frames = [], [], []
    with torch.no_grad():
        for batch in loader:
            pred, y, meta = _forward_batch(model, batch, device)
            batch_size = int(y.shape[0])
            total_loss += float(loss_fn(pred, y).item()) * batch_size
            total_samples += batch_size
            preds_gpu.append(pred)
            targets_gpu.append(y)
            frames.append(_metadata_to_frame(meta))
    all_preds = torch.cat(preds_gpu, dim=0).cpu().numpy()
    all_targets = torch.cat(targets_gpu, dim=0).cpu().numpy()
    return (
        _mean_loss(total_loss, total_samples),
        PredictionBundle(
            y_true=all_targets,
            y_pred=all_preds,
            meta=pd.concat(frames, ignore_index=True),
        ),
    )


def evaluate_loss(model, loader, loss_fn, device) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch in loader:
            pred, y, _ = _forward_batch(model, batch, device)
            batch_size = int(y.shape[0])
            total_loss += float(loss_fn(pred, y).item()) * batch_size
            total_samples += batch_size
    return _mean_loss(total_loss, total_samples)


def _train_one_epoch(request: TrainEpochRequest) -> float:
    request.model.train()
    total_loss = 0.0
    total_samples = 0
    scaler = request.scaler
    if scaler is None:
        scaler = torch.amp.GradScaler("cuda", enabled=False)
    for batch in request.loader:
        request.optimizer.zero_grad(set_to_none=True)
        if not _is_waveform_batch(batch) and request.env_aug_sigma > 0.0:
            x, y, _ = batch
            x = _move_tensor(x, request.device)
            y = _move_tensor(y, request.device)
            x = apply_environment_augmentation(x, request.dataset.input_format, request.env_aug_sigma)
            with _amp_context(request.device, request.amp_enabled):
                pred = request.model(x)
                loss = request.loss_fn(pred, y)
        else:
            with _amp_context(request.device, request.amp_enabled):
                pred, y, _ = _forward_batch(request.model, batch, request.device)
                loss = request.loss_fn(pred, y)
        if request.amp_enabled:
            scaler.scale(loss).backward()
            if request.grad_clip_norm > 0:
                scaler.unscale_(request.optimizer)
                torch.nn.utils.clip_grad_norm_(_optimizer_parameters(request.optimizer), request.grad_clip_norm)
            scaler.step(request.optimizer)
            scaler.update()
        else:
            loss.backward()
            if request.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(_optimizer_parameters(request.optimizer), request.grad_clip_norm)
            request.optimizer.step()
        batch_size = int(y.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
    return _mean_loss(total_loss, total_samples)


def save_predictions(path: Path, bundle: PredictionBundle, split: str, label_names) -> None:
    out = pd.DataFrame()
    out["sample_id"] = bundle.meta.get("sample_id", bundle.meta.get("sequence_id")).astype(str)
    out["mixture_id"] = bundle.meta.get("mixture_id", out["sample_id"]).astype(str)
    out["split"] = split
    labels = _display_labels(label_names)
    for i, label in enumerate(labels):
        out[f"y_true_{label}"] = bundle.y_true[:, i]
    for i, label in enumerate(labels):
        out[f"y_pred_{label}"] = bundle.y_pred[:, i]
    out["sum_true"] = bundle.y_true.sum(axis=1)
    out["sum_pred"] = bundle.y_pred.sum(axis=1)
    out["abs_sum_error"] = (out["sum_pred"] - out["sum_true"]).abs()
    out.to_csv(path, index=False)
