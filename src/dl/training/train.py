from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

from training.checkpoints import (
    CheckpointBuildContext,
    _capture_rng_state,
    _checkpoint_payload as _checkpoint_payload_impl,
    _load_checkpoint,
    _restore_early_stopping,
    _restore_rng_state,
    _save_checkpoint,
    _validate_checkpoint_compat,
    _validate_model_architecture_compat,
)
from training.orchestrator import (
    TrainDependencies,
    TrainRunOptions,
    _ensure_scaler_path,
    _load_label_names,
    build_datasets,
    load_config,
    resolve_path,
    train_config as _train_config_impl,
    train_one as _train_one_impl,
)
from training.runtime import (
    PredictionBundle,
    TrainEpochRequest,
    _amp_context,
    _forward_batch,
    _move_tensor,
    _move_waveform_batch,
    _train_one_epoch as _train_one_epoch_impl,
    _use_amp,
    apply_environment_augmentation,
    evaluate_loss,
    evaluate_with_predictions,
    make_loader,
    predict,
    save_predictions,
    select_device,
)


_TRAIN_CONFIG_OPTION_KEYS = {
    "epochs_override",
    "resume_path",
    "checkpoint_every",
    "restore_rng",
    "stop_after_epoch",
}
_TRAIN_ONE_OPTION_KEYS = _TRAIN_CONFIG_OPTION_KEYS | {"progress"}
_EPOCH_REQUEST_KEYS = {field.name for field in fields(TrainEpochRequest)}
_CHECKPOINT_CONTEXT_KEYS = {field.name for field in fields(CheckpointBuildContext)}
_EPOCH_REQUEST_ORDER = [field.name for field in fields(TrainEpochRequest)]
_CHECKPOINT_CONTEXT_ORDER = [field.name for field in fields(CheckpointBuildContext)]


def _validate_legacy_kwargs(legacy_kwargs: dict, allowed_keys: set[str], target: str) -> None:
    unknown = sorted(set(legacy_kwargs).difference(allowed_keys))
    if unknown:
        raise TypeError(f"{target} got unexpected keyword arguments: {unknown}")


def _merge_positional_legacy_args(args: tuple, legacy_kwargs: dict, field_order: list[str], target: str) -> dict:
    if len(args) > len(field_order):
        raise TypeError(f"{target} takes at most {len(field_order)} positional arguments ({len(args)} given)")
    merged = dict(legacy_kwargs)
    for name, value in zip(field_order, args):
        if name in merged:
            raise TypeError(f"{target} got multiple values for argument '{name}'")
        merged[name] = value
    return merged


def _build_train_options(options: TrainRunOptions | None, legacy_kwargs: dict, allow_progress: bool) -> TrainRunOptions:
    allowed_keys = _TRAIN_ONE_OPTION_KEYS if allow_progress else _TRAIN_CONFIG_OPTION_KEYS
    _validate_legacy_kwargs(legacy_kwargs, allowed_keys, "TrainRunOptions")
    if options is not None and legacy_kwargs:
        raise TypeError("cannot mix options object with legacy keyword arguments")
    if options is not None:
        return options
    return TrainRunOptions(**legacy_kwargs)


def _build_epoch_request(request: TrainEpochRequest | None, args: tuple, legacy_kwargs: dict) -> TrainEpochRequest:
    merged_kwargs = _merge_positional_legacy_args(args, legacy_kwargs, _EPOCH_REQUEST_ORDER, "_train_one_epoch")
    _validate_legacy_kwargs(merged_kwargs, _EPOCH_REQUEST_KEYS, "TrainEpochRequest")
    if request is not None and merged_kwargs:
        raise TypeError("cannot mix request object with legacy keyword arguments")
    if request is not None:
        return request
    return TrainEpochRequest(**merged_kwargs)


def _build_checkpoint_context(context: CheckpointBuildContext | None, args: tuple, legacy_kwargs: dict) -> CheckpointBuildContext:
    merged_kwargs = _merge_positional_legacy_args(args, legacy_kwargs, _CHECKPOINT_CONTEXT_ORDER, "_checkpoint_payload")
    _validate_legacy_kwargs(merged_kwargs, _CHECKPOINT_CONTEXT_KEYS, "CheckpointBuildContext")
    if context is not None and merged_kwargs:
        raise TypeError("cannot mix context object with legacy keyword arguments")
    if context is not None:
        return context
    return CheckpointBuildContext(**merged_kwargs)


def _build_train_dependencies() -> TrainDependencies:
    return TrainDependencies(
        ensure_scaler_path=_ensure_scaler_path,
        load_label_names=_load_label_names,
        build_datasets=build_datasets,
        predict=predict,
        evaluate_with_predictions=evaluate_with_predictions,
        train_one_epoch=_train_one_epoch,
        load_checkpoint=_load_checkpoint,
        validate_checkpoint_compat=_validate_checkpoint_compat,
        restore_early_stopping=_restore_early_stopping,
        restore_rng_state=_restore_rng_state,
        checkpoint_payload=_checkpoint_payload,
        save_checkpoint=_save_checkpoint,
        save_predictions=save_predictions,
    )


def _train_one_epoch(*args, request: TrainEpochRequest | None = None, **legacy_kwargs) -> float:
    resolved_request = _build_epoch_request(request, args, legacy_kwargs)
    return _train_one_epoch_impl(resolved_request)


def _checkpoint_payload(*args, context: CheckpointBuildContext | None = None, **legacy_kwargs) -> dict:
    resolved_context = _build_checkpoint_context(context, args, legacy_kwargs)
    return _checkpoint_payload_impl(resolved_context)


def train_config(config: dict, options: TrainRunOptions | None = None, **legacy_kwargs) -> dict:
    resolved_options = _build_train_options(options, legacy_kwargs, allow_progress=False)
    return _train_config_impl(config, options=resolved_options, dependencies=_build_train_dependencies())


def train_one(config_path: str | Path, options: TrainRunOptions | None = None, **legacy_kwargs) -> dict:
    resolved_options = _build_train_options(options, legacy_kwargs, allow_progress=True)
    return _train_one_impl(config_path, options=resolved_options, dependencies=_build_train_dependencies())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    summary = train_one(args.config, epochs_override=args.epochs)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
