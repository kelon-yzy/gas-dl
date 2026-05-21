from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import torch
from models.registry import build_model
from training.checkpoints import CheckpointBuildContext, _checkpoint_payload, _load_checkpoint, _restore_early_stopping, _restore_rng_state, _save_checkpoint, _validate_checkpoint_compat
from training.data_setup import _ensure_scaler_path, _load_label_names, build_datasets, load_config, resolve_path
from training.early_stopping import EarlyStopping
from training.losses import build_loss
from training.metrics import regression_metrics
from training.runtime import TrainEpochRequest, _train_one_epoch, _use_amp, configure_cudnn, evaluate_with_predictions, make_loader, predict, save_predictions, select_device
from training.seed import set_seed


@dataclass
class TrainRunOptions:
    epochs_override: int | None = None
    resume_path: str | Path | None = None
    checkpoint_every: int = 0
    restore_rng: bool = True
    stop_after_epoch: int | None = None
    progress: Any = None


@dataclass(frozen=True)
class TrainDependencies:
    ensure_scaler_path: Any = None
    load_label_names: Any = None
    build_datasets: Any = None
    predict: Any = predict
    evaluate_with_predictions: Any = evaluate_with_predictions
    train_one_epoch: Any = _train_one_epoch
    load_checkpoint: Any = _load_checkpoint
    validate_checkpoint_compat: Any = _validate_checkpoint_compat
    restore_early_stopping: Any = _restore_early_stopping
    restore_rng_state: Any = _restore_rng_state
    checkpoint_payload: Any = _checkpoint_payload
    save_checkpoint: Any = _save_checkpoint
    save_predictions: Any = save_predictions


@dataclass
class PreparedTrainingResources:
    output_dir: Path
    label_names: list[str]
    datasets: dict
    loaders: dict
    batch_size: int
    device: torch.device
    num_workers: int
    eval_num_workers: int
    amp_enabled: bool
    model: Any
    loss_fn: Any
    optimizer: Any
    stopper: Any
    scaler: Any
    total_epochs: int
    grad_clip_norm: float
    scheduler: Any


@dataclass
class TrainExecutionContext:
    config: dict
    options: TrainRunOptions
    dependencies: TrainDependencies
    output_dir: Path
    label_names: list[str]
    datasets: dict
    loaders: dict
    batch_size: int
    device: torch.device
    num_workers: int
    eval_num_workers: int
    amp_enabled: bool
    model: Any
    loss_fn: Any
    optimizer: Any
    stopper: Any
    scaler: Any
    total_epochs: int
    grad_clip_norm: float
    scheduler: Any
    progress: Any
    config_to_write: dict
    best_path: Path
    last_ckpt_path: Path
    best_ckpt_path: Path
    paused_ckpt_path: Path
    initial_ckpt_path: Path
    start_epoch: int = 1
    log_rows: list = field(default_factory=list)
    resumed_from: str | None = None
    current_run_last_checkpoint_written: bool = False


@dataclass(frozen=True)
class EpochResult:
    train_loss: float
    val_loss: float
    monitor_value: float
    val_macro_mae: float
    val_mean_pred_sum: float
    val_mean_abs_sum_error: float
    val_std_pred_sum: float
    improved: bool
    epoch_seconds: float
    val_skipped: bool = False


def _default_dependencies() -> TrainDependencies:
    return TrainDependencies(
        ensure_scaler_path=_ensure_scaler_path,
        load_label_names=_load_label_names,
        build_datasets=build_datasets,
    )


def _normalize_training_request(config: dict, options: TrainRunOptions | None) -> TrainRunOptions:
    resolved = options or TrainRunOptions()
    if resolved.progress is None:
        resolved.progress = config.get("_cli_progress")
    if resolved.epochs_override is not None:
        config["training"]["epochs"] = resolved.epochs_override
    return resolved


def _build_optimizer(model, training_config: dict):
    lr = float(training_config.get("learning_rate", 1e-3))
    wd = float(training_config.get("weight_decay", 0.01))
    optimizer_name = training_config.get("optimizer", "adam").lower()
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)


def _build_optimizer_with_loss(model, loss_fn, training_config: dict):
    """构建优化器，若 loss_fn 含可学参数（如 UncertaintyWeightedLoss），一并加入优化。

    loss 参数使用独立学习率（默认为模型lr的10倍），不施加 weight_decay。
    """
    lr = float(training_config.get("learning_rate", 1e-3))
    wd = float(training_config.get("weight_decay", 0.01))
    optimizer_name = training_config.get("optimizer", "adam").lower()

    # 收集 loss_fn 中的可学参数
    loss_params = list(loss_fn.parameters())
    if not loss_params:
        # 无可学参数，退回普通 optimizer
        return _build_optimizer(model, training_config)

    # loss 参数使用较高学习率（Kendall论文建议），不加 weight_decay
    loss_lr_multiplier = float(training_config.get("uncertainty_weighted", {}).get("lr_multiplier", 10.0))
    param_groups = [
        {"params": model.parameters(), "lr": lr, "weight_decay": wd},
        {"params": loss_params, "lr": lr * loss_lr_multiplier, "weight_decay": 0.0},
    ]

    if optimizer_name == "adamw":
        return torch.optim.AdamW(param_groups)
    return torch.optim.Adam(param_groups)


def _prepare_scheduler(optimizer, total_epochs: int, training_config: dict):
    scheduler_cfg = training_config.get("lr_scheduler")
    if scheduler_cfg is None:
        return None
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

    lr = float(training_config.get("learning_rate", 1e-3))
    stype = scheduler_cfg.get("type", "cosine_warmup")
    if stype == "cosine_warmup":
        warmup_epochs = int(scheduler_cfg.get("warmup_epochs", 5))
        eta_min = float(scheduler_cfg.get("eta_min", lr * 0.1))
        warmup_start = eta_min / lr
        warmup = LinearLR(optimizer, start_factor=warmup_start, end_factor=1.0, total_iters=warmup_epochs)
        cosine = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs, eta_min=eta_min)
        return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
    if stype == "cosine":
        eta_min = float(scheduler_cfg.get("eta_min", lr * 0.1))
        return CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=eta_min)
    if stype == "multistep":
        milestones = [int(value) for value in scheduler_cfg.get("milestones", [80, 140])]
        gamma = float(scheduler_cfg.get("gamma", 0.1))
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    if stype == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(scheduler_cfg.get("gamma", 0.1)),
            patience=int(scheduler_cfg.get("patience", 10)),
        )
    raise ValueError(f"Unknown lr_scheduler type: {stype}")


def _prepare_loader_resources(config: dict, dependencies: TrainDependencies, output_dir: Path) -> tuple:
    dependencies.ensure_scaler_path(config["data"], output_dir)
    label_names = dependencies.load_label_names(config)
    datasets, _ = dependencies.build_datasets(config)
    batch_size = int(config["training"].get("batch_size", 64))
    device = select_device(config["training"].get("device", "auto"))
    configure_cudnn(config["training"], device)
    num_workers = int(config["training"].get("num_workers", 0))
    eval_num_workers = int(config["training"].get("eval_num_workers", 0))
    prefetch_factor = config["training"].get("prefetch_factor")
    amp_enabled = _use_amp(config["training"], device)
    loaders = {
        "train": make_loader(
            datasets["train"], batch_size, shuffle=True, device=device,
            num_workers=num_workers, prefetch_factor=prefetch_factor,
        ),
        "val": make_loader(
            datasets["val"], batch_size, shuffle=False, device=device,
            num_workers=eval_num_workers, prefetch_factor=prefetch_factor,
        ),
        "test": make_loader(
            datasets["test"], batch_size, shuffle=False, device=device,
            num_workers=eval_num_workers, prefetch_factor=prefetch_factor,
        ),
    }
    return label_names, datasets, loaders, batch_size, device, num_workers, eval_num_workers, amp_enabled


def _prepare_model_resources(config: dict, device: torch.device, amp_enabled: bool, label_weights: torch.Tensor | None = None) -> tuple:
    model = build_model(config["model"]).to(device)
    # B3: torch.compile 支持（默认关闭，需 PyTorch 2.0+；Windows 上需 triton 后端）
    compile_cfg = config["training"].get("compile", False)
    if compile_cfg and hasattr(torch, "compile"):
        compile_mode = config["training"].get("compile_mode", "reduce-overhead")
        try:
            model = torch.compile(model, mode=compile_mode)
        except Exception as exc:  # noqa: BLE001 - compile 失败不应阻断训练
            import logging
            logging.getLogger(__name__).warning("torch.compile 失败，回退到 eager 模式: %s", exc)
    loss_fn = build_loss(
        config["training"].get("loss", "mse"),
        sum_constraint=config["training"].get("sum_constraint"),
        label_weights=label_weights,
        uncertainty_weighted=config["training"].get("uncertainty_weighted"),
        loss_columns=config["training"].get("loss_columns"),
    )
    # 将 loss_fn 移到设备上（UncertaintyWeightedLoss 含可学参数）
    loss_fn = loss_fn.to(device)
    optimizer = _build_optimizer_with_loss(model, loss_fn, config["training"])
    stopper = EarlyStopping(patience=int(config["training"].get("early_stopping_patience", 25)), mode="min")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    total_epochs = int(config["training"].get("epochs", 200))
    scheduler = _prepare_scheduler(optimizer, total_epochs, config["training"])
    grad_clip_norm = float(config["training"].get("grad_clip_norm", 0.0))
    return model, loss_fn, optimizer, stopper, scaler, total_epochs, grad_clip_norm, scheduler


def _resolve_label_weights(training_config: dict, datasets: dict) -> torch.Tensor | None:
    """读取 train dataset 上挂载的 label_std, 转成 1/var 权重传给 loss。

    返回 None 时表示退回原始等权 MSE/MAE。
    """
    if not training_config.get("label_balanced_loss", False):
        return None
    train_ds = datasets.get("train")
    label_std = getattr(train_ds, "label_std", None)
    if label_std is None:
        return None
    std_arr = np.asarray(label_std, dtype=np.float32)
    std_arr = np.where(std_arr < 1e-6, np.float32(1.0), std_arr)
    weights = 1.0 / (std_arr ** 2)
    return torch.from_numpy(weights.astype(np.float32))


def _prepare_training_resources(config: dict, dependencies: TrainDependencies) -> PreparedTrainingResources:
    set_seed(int(config["run"].get("seed", 42)))
    output_dir = resolve_path(config["run"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    label_names, datasets, loaders, batch_size, device, num_workers, eval_num_workers, amp_enabled = _prepare_loader_resources(
        config,
        dependencies,
        output_dir,
    )
    label_weights = _resolve_label_weights(config["training"], datasets)
    model, loss_fn, optimizer, stopper, scaler, total_epochs, grad_clip_norm, scheduler = _prepare_model_resources(
        config,
        device,
        amp_enabled,
        label_weights=label_weights,
    )
    return PreparedTrainingResources(
        output_dir=output_dir,
        label_names=label_names,
        datasets=datasets,
        loaders=loaders,
        batch_size=batch_size,
        device=device,
        num_workers=num_workers,
        eval_num_workers=eval_num_workers,
        amp_enabled=amp_enabled,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        stopper=stopper,
        scaler=scaler,
        total_epochs=total_epochs,
        grad_clip_norm=grad_clip_norm,
        scheduler=scheduler,
    )


def _prepare_training_setup(config: dict, options: TrainRunOptions, dependencies: TrainDependencies) -> TrainExecutionContext:
    resources = _prepare_training_resources(config, dependencies)
    return TrainExecutionContext(
        config=config,
        options=options,
        dependencies=dependencies,
        output_dir=resources.output_dir,
        label_names=resources.label_names,
        datasets=resources.datasets,
        loaders=resources.loaders,
        batch_size=resources.batch_size,
        device=resources.device,
        num_workers=resources.num_workers,
        eval_num_workers=resources.eval_num_workers,
        amp_enabled=resources.amp_enabled,
        model=resources.model,
        loss_fn=resources.loss_fn,
        optimizer=resources.optimizer,
        stopper=resources.stopper,
        scaler=resources.scaler,
        total_epochs=resources.total_epochs,
        grad_clip_norm=resources.grad_clip_norm,
        scheduler=resources.scheduler,
        progress=options.progress,
        config_to_write={key: value for key, value in config.items() if not key.startswith("_")},
        best_path=resources.output_dir / "best_model.pt",
        last_ckpt_path=resources.output_dir / "last_checkpoint.pt",
        best_ckpt_path=resources.output_dir / "best_checkpoint.pt",
        paused_ckpt_path=resources.output_dir / "paused_checkpoint.pt",
        initial_ckpt_path=resources.output_dir / "initial_checkpoint.pt",
    )


def _checkpoint_context(ctx: TrainExecutionContext, epoch: int, log_rows: list, status: str) -> CheckpointBuildContext:
    return CheckpointBuildContext(
        model=ctx.model,
        optimizer=ctx.optimizer,
        scaler=ctx.scaler,
        stopper=ctx.stopper,
        epoch=epoch,
        total_epochs=ctx.total_epochs,
        log_rows=log_rows,
        best_path=ctx.best_path,
        config_to_write=ctx.config_to_write,
        status=status,
        label_names=ctx.label_names,
        scheduler=ctx.scheduler,
        loss_fn=ctx.loss_fn,
    )


def _save_run_checkpoint(ctx: TrainExecutionContext, path: Path, epoch: int, log_rows: list, status: str) -> None:
    payload = ctx.dependencies.checkpoint_payload(context=_checkpoint_context(ctx, epoch, log_rows, status))
    ctx.dependencies.save_checkpoint(path, payload)


def _restore_run_state_if_needed(ctx: TrainExecutionContext) -> None:
    resume_path = ctx.options.resume_path
    if not resume_path:
        return
    ckpt = ctx.dependencies.load_checkpoint(Path(resume_path), ctx.device)
    ctx.dependencies.validate_checkpoint_compat(ckpt, ctx.config, ctx.label_names, model_state_keys=set(ctx.model.state_dict().keys()))
    ctx.model.load_state_dict(ckpt["model_state_dict"])
    ctx.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    ctx.scaler.load_state_dict(ckpt["amp_scaler_state_dict"])
    # 恢复 loss_fn 状态（UncertaintyWeightedLoss 的 log_sigmas）
    loss_fn_state = ckpt.get("loss_fn_state_dict")
    if loss_fn_state is not None and hasattr(ctx.loss_fn, "load_state_dict"):
        ctx.loss_fn.load_state_dict(loss_fn_state)
    ctx.dependencies.restore_early_stopping(ctx.stopper, ckpt.get("early_stopping", {}))
    if ctx.scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        ctx.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if ctx.options.restore_rng:
        ctx.dependencies.restore_rng_state(ckpt["rng_state"])
    ctx.log_rows = ckpt.get("log_rows", [])
    ctx.start_epoch = ckpt["epoch"] + 1
    ctx.resumed_from = str(resume_path)
    if ctx.start_epoch > ctx.total_epochs and ctx.progress is not None:
        ctx.progress.log_message(f"checkpoint epoch {ckpt['epoch']} >= target {ctx.total_epochs}, 跳过训练直接评估")
        ctx.start_epoch = ctx.total_epochs + 1


def _start_progress(ctx: TrainExecutionContext) -> None:
    if ctx.progress is None:
        return
    ctx.progress.start_run(
        mode="deep",
        title=ctx.config["run"]["name"],
        seed=int(ctx.config["run"].get("seed", 42)),
        stage="setup",
    )
    ctx.progress.update_metric(
        model=ctx.config["model"]["name"],
        device=str(ctx.device),
        batch_size=ctx.batch_size,
        n_train=len(ctx.datasets["train"]),
        n_val=len(ctx.datasets["val"]),
        n_test=len(ctx.datasets["test"]),
    )


def _save_initial_checkpoint(ctx: TrainExecutionContext) -> None:
    _save_run_checkpoint(ctx, ctx.initial_ckpt_path, ctx.start_epoch - 1, list(ctx.log_rows), "initial")


def _step_scheduler(scheduler, monitor_value: float) -> None:
    if scheduler is None:
        return
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(monitor_value)
    else:
        scheduler.step()


def _restore_clean_state_after_interrupt(ctx: TrainExecutionContext) -> None:
    if ctx.progress is not None:
        ctx.progress.log_message("用户中断 (Ctrl+C), 回退到最近完整 epoch 的干净权重")
    restore_source = None
    if ctx.current_run_last_checkpoint_written and ctx.last_ckpt_path.exists():
        restore_source = ctx.last_ckpt_path
    elif ctx.initial_ckpt_path.exists():
        restore_source = ctx.initial_ckpt_path
    if restore_source is None:
        return
    ckpt_clean = ctx.dependencies.load_checkpoint(restore_source, ctx.device)
    ctx.model.load_state_dict(ckpt_clean["model_state_dict"])
    ctx.optimizer.load_state_dict(ckpt_clean["optimizer_state_dict"])
    ctx.scaler.load_state_dict(ckpt_clean["amp_scaler_state_dict"])
    loss_fn_state = ckpt_clean.get("loss_fn_state_dict")
    if loss_fn_state is not None and hasattr(ctx.loss_fn, "load_state_dict"):
        ctx.loss_fn.load_state_dict(loss_fn_state)


def _build_epoch_request(ctx: TrainExecutionContext) -> TrainEpochRequest:
    return TrainEpochRequest(
        model=ctx.model,
        loader=ctx.loaders["train"],
        loss_fn=ctx.loss_fn,
        optimizer=ctx.optimizer,
        device=ctx.device,
        dataset=ctx.datasets["train"],
        env_aug_sigma=float(ctx.config["training"].get("environment_augmentation_sigma", 0.0)),
        amp_enabled=ctx.amp_enabled,
        scaler=ctx.scaler,
        grad_clip_norm=ctx.grad_clip_norm,
    )


def _run_training_epoch(ctx: TrainExecutionContext, do_validation: bool) -> tuple[float, float | None, dict | None]:
    train_loss = ctx.dependencies.train_one_epoch(request=_build_epoch_request(ctx))
    if not do_validation:
        return train_loss, None, None
    val_loss, val_bundle = ctx.dependencies.evaluate_with_predictions(ctx.model, ctx.loaders["val"], ctx.loss_fn, ctx.device)
    val_summary, _ = regression_metrics(val_bundle.y_true, val_bundle.y_pred, label_names=ctx.label_names)
    return train_loss, val_loss, val_summary


def _should_validate_this_epoch(ctx: TrainExecutionContext, epoch: int) -> bool:
    """val_every>1 时按间隔验证；epoch=1 / 最后 epoch 强制验证，保证有起止 baseline。"""
    val_every = int(ctx.config["training"].get("val_every", 1))
    if val_every <= 1:
        return True
    if epoch == 1 or epoch == ctx.total_epochs:
        return True
    return (epoch % val_every) == 0


def _should_write_last_ckpt(ctx: TrainExecutionContext, epoch: int) -> bool:
    """last_checkpoint_every>1 时按间隔写 last_checkpoint.pt；最后 epoch 强制写。"""
    interval = int(ctx.config["training"].get("last_checkpoint_every", 1))
    if interval <= 1:
        return True
    if epoch == ctx.total_epochs:
        return True
    return (epoch % interval) == 0


def _get_uncertainty_info(loss_fn) -> dict:
    """从 loss_fn 中提取 UncertaintyWeightedLoss 的当前权重信息。"""
    from training.losses import UncertaintyWeightedLoss
    # 可能直接是 UncertaintyWeightedLoss，也可能被 SumConstraintLoss 包裹
    uw_loss = None
    if isinstance(loss_fn, UncertaintyWeightedLoss):
        uw_loss = loss_fn
    elif hasattr(loss_fn, "base_loss") and isinstance(loss_fn.base_loss, UncertaintyWeightedLoss):
        uw_loss = loss_fn.base_loss
    if uw_loss is None:
        return {}
    sigmas = uw_loss.get_sigmas().cpu().numpy()
    weights = uw_loss.get_weights().cpu().numpy()
    return {
        "uw_sigmas": [float(s) for s in sigmas],
        "uw_weights": [float(w) for w in weights],
    }


def _record_epoch_result(ctx: TrainExecutionContext, epoch: int, result: EpochResult) -> None:
    lr = ctx.optimizer.param_groups[0]["lr"]
    row = {
        "epoch": epoch,
        "train_loss": result.train_loss,
        "val_loss": result.val_loss,
        "val_macro_RMSE": result.monitor_value,
        "val_macro_MAE": result.val_macro_mae,
        "val_mean_pred_sum": result.val_mean_pred_sum,
        "val_mean_abs_sum_error": result.val_mean_abs_sum_error,
        "val_std_pred_sum": result.val_std_pred_sum,
        "improved": result.improved,
        "lr": lr,
        "val_skipped": result.val_skipped,
    }
    # 附加 uncertainty weighting 监控信息
    uw_info = _get_uncertainty_info(ctx.loss_fn)
    row.update(uw_info)
    ctx.log_rows.append(row)
    if result.improved and not result.val_skipped:
        torch.save(ctx.model.state_dict(), ctx.best_path)
        _save_run_checkpoint(ctx, ctx.best_ckpt_path, epoch, ctx.log_rows, "running")
    if ctx.progress is not None:
        ctx.progress.update_stage(stage="epoch", current_task=f"epoch={epoch}/{ctx.total_epochs}", completed=epoch, total=ctx.total_epochs)
        ctx.progress.update_metric(
            epoch=epoch,
            train_loss=result.train_loss,
            val_loss=result.val_loss,
            val_macro_RMSE=result.monitor_value,
            val_mean_pred_sum=result.val_mean_pred_sum,
            val_mean_abs_sum_error=result.val_mean_abs_sum_error,
            improved=result.improved,
            best=ctx.stopper.best,
            bad_epochs=ctx.stopper.bad_epochs,
            patience=ctx.stopper.patience,
            epoch_seconds=result.epoch_seconds,
        )
    if _should_write_last_ckpt(ctx, epoch):
        _save_run_checkpoint(ctx, ctx.last_ckpt_path, epoch, ctx.log_rows, "running")
        ctx.current_run_last_checkpoint_written = True
    if ctx.options.checkpoint_every > 0 and epoch % ctx.options.checkpoint_every == 0:
        _save_run_checkpoint(ctx, ctx.output_dir / f"epoch_{epoch:04d}.pt", epoch, ctx.log_rows, "running")
    # 跳过 val 的 epoch 用 monitor=NaN，plateau scheduler 会异常，需在配置中避开此组合
    _step_scheduler(ctx.scheduler, result.monitor_value)


def _should_pause_after_epoch(ctx: TrainExecutionContext, epoch: int) -> bool:
    if ctx.options.stop_after_epoch is not None and epoch >= ctx.options.stop_after_epoch:
        if ctx.progress is not None:
            ctx.progress.log_message(f"stop_after_epoch={ctx.options.stop_after_epoch}, 暂停于 epoch {epoch}")
        return True
    if ctx.stopper.should_stop and ctx.progress is not None:
        ctx.progress.log_message(f"early stop at epoch {epoch}")
    return ctx.stopper.should_stop


def _run_training_epochs(ctx: TrainExecutionContext) -> bool:
    try:
        for epoch in range(ctx.start_epoch, ctx.total_epochs + 1):
            epoch_started = perf_counter()
            do_val = _should_validate_this_epoch(ctx, epoch)
            train_loss, val_loss, val_summary = _run_training_epoch(ctx, do_val)
            if do_val and val_summary is not None:
                monitor_value = val_summary["macro_RMSE"]
                result = EpochResult(
                    train_loss=train_loss,
                    val_loss=val_loss,
                    monitor_value=monitor_value,
                    val_macro_mae=val_summary["macro_MAE"],
                    val_mean_pred_sum=val_summary["mean_pred_sum"],
                    val_mean_abs_sum_error=val_summary["mean_abs_sum_error"],
                    val_std_pred_sum=val_summary["std_pred_sum"],
                    improved=ctx.stopper.step(monitor_value),
                    epoch_seconds=perf_counter() - epoch_started,
                    val_skipped=False,
                )
            else:
                # 跳过 val 的轮次：metric 写 NaN，stopper 不 step，bad_epochs 不累加；
                # scheduler 仍 step（适配 cosine/cosine_warmup，plateau 不可用）
                nan = float("nan")
                result = EpochResult(
                    train_loss=train_loss,
                    val_loss=nan,
                    monitor_value=nan,
                    val_macro_mae=nan,
                    val_mean_pred_sum=nan,
                    val_mean_abs_sum_error=nan,
                    val_std_pred_sum=nan,
                    improved=False,
                    epoch_seconds=perf_counter() - epoch_started,
                    val_skipped=True,
                )
            _record_epoch_result(ctx, epoch, result)
            if _should_pause_after_epoch(ctx, epoch):
                return ctx.options.stop_after_epoch is not None and epoch >= ctx.options.stop_after_epoch
    except KeyboardInterrupt:
        _restore_clean_state_after_interrupt(ctx)
        return True
    return False


def _finalize_paused_run(ctx: TrainExecutionContext) -> dict:
    last_epoch = ctx.log_rows[-1]["epoch"] if ctx.log_rows else (ctx.start_epoch - 1)
    _save_run_checkpoint(ctx, ctx.paused_ckpt_path, last_epoch, ctx.log_rows, "paused")
    summary = {
        "run_name": ctx.config["run"]["name"],
        "model": ctx.config["model"]["name"],
        "label_names": ctx.label_names,
        "seed": int(ctx.config["run"].get("seed", 42)),
        "training_status": "paused",
        "resumed_from": ctx.resumed_from,
        "last_checkpoint": str(ctx.last_ckpt_path),
        "best_checkpoint": str(ctx.best_ckpt_path),
        "epochs_trained": int(ctx.log_rows[-1]["epoch"]) if ctx.log_rows else 0,
    }
    if ctx.progress is not None:
        ctx.progress.finish_run(status="paused", epochs_trained=summary["epochs_trained"])
    (ctx.output_dir / "config.json").write_text(json.dumps(ctx.config_to_write, indent=2, ensure_ascii=False), encoding="utf-8")
    (ctx.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(ctx.log_rows).to_csv(ctx.output_dir / "train_log.csv", index=False)
    return summary


def _finalize_completed_run(ctx: TrainExecutionContext) -> dict:
    if ctx.best_path.exists():
        ctx.model.load_state_dict(torch.load(ctx.best_path, map_location=ctx.device, weights_only=True))
    test_bundle = ctx.dependencies.predict(ctx.model, ctx.loaders["test"], ctx.device)
    summary, component_metrics = regression_metrics(test_bundle.y_true, test_bundle.y_pred, label_names=ctx.label_names)
    summary.update(
        {
            "run_name": ctx.config["run"]["name"],
            "model": ctx.config["model"]["name"],
            "label_names": ctx.label_names,
            "seed": int(ctx.config["run"].get("seed", 42)),
            "device": str(ctx.device),
            "amp": bool(ctx.amp_enabled),
            "batch_size": ctx.batch_size,
            "num_workers": ctx.num_workers,
            "eval_num_workers": ctx.eval_num_workers,
            "epochs_trained": int(ctx.log_rows[-1]["epoch"]) if ctx.log_rows else 0,
            "n_train": int(len(ctx.datasets["train"])),
            "n_val": int(len(ctx.datasets["val"])),
            "n_test": int(len(ctx.datasets["test"])),
            "training_status": "completed",
            "resumed_from": ctx.resumed_from,
            "last_checkpoint": str(ctx.last_ckpt_path),
            "best_checkpoint": str(ctx.best_ckpt_path),
        }
    )
    if "use_waveform" in ctx.config.get("model", {}):
        summary["use_waveform"] = bool(ctx.config["model"]["use_waveform"])
    if ctx.progress is not None:
        ctx.progress.finish_run(status="done", macro_RMSE=summary["macro_RMSE"], epochs_trained=summary["epochs_trained"])

    final_epoch = int(ctx.log_rows[-1]["epoch"]) if ctx.log_rows else 0
    _save_run_checkpoint(ctx, ctx.last_ckpt_path, final_epoch, ctx.log_rows, "completed")

    (ctx.output_dir / "config.json").write_text(json.dumps(ctx.config_to_write, indent=2, ensure_ascii=False), encoding="utf-8")
    (ctx.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    component_metrics.to_csv(ctx.output_dir / "component_metrics.csv", index=False)
    pd.DataFrame(ctx.log_rows).to_csv(ctx.output_dir / "train_log.csv", index=False)
    ctx.dependencies.save_predictions(ctx.output_dir / "predictions.csv", test_bundle, split="test", label_names=ctx.label_names)
    return summary


def train_config(config: dict, options: TrainRunOptions | None = None, dependencies: TrainDependencies | None = None) -> dict:
    deps = dependencies or _default_dependencies()
    resolved_options = _normalize_training_request(config, options)
    ctx = _prepare_training_setup(config, resolved_options, deps)
    _restore_run_state_if_needed(ctx)
    _start_progress(ctx)
    _save_initial_checkpoint(ctx)
    paused = _run_training_epochs(ctx)
    if paused:
        return _finalize_paused_run(ctx)
    return _finalize_completed_run(ctx)


def train_one(config_path: str | Path, options: TrainRunOptions | None = None, dependencies: TrainDependencies | None = None) -> dict:
    config = load_config(Path(config_path).resolve())
    if options is not None and options.progress is not None:
        config["_cli_progress"] = options.progress
    return train_config(config, options=options, dependencies=dependencies)
