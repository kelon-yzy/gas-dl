from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
DL_ROOT = SRC_ROOT / "dl"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(DL_ROOT) not in sys.path:
    sys.path.insert(0, str(DL_ROOT))

from training.data_setup import _ensure_scaler_path, _load_label_names, build_datasets, resolve_path
from training.metrics import regression_metrics
from training.runtime import _use_amp, configure_cudnn, make_loader, predict, save_predictions, select_device
from models.registry import build_model


def _load_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)


def _resolve_run_dir(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def _build_test_loader(config: dict):
    output_dir = resolve_path(config["run"]["output_dir"])
    _ensure_scaler_path(config["data"], output_dir)
    label_names = _load_label_names(config)
    datasets, _ = build_datasets(config)
    batch_size = int(config["training"].get("batch_size", 64))
    device = select_device(config["training"].get("device", "auto"))
    configure_cudnn(config["training"], device)
    eval_num_workers = int(config["training"].get("eval_num_workers", 0))
    prefetch_factor = config["training"].get("prefetch_factor")
    loader = make_loader(
        datasets["test"],
        batch_size,
        shuffle=False,
        device=device,
        num_workers=eval_num_workers,
        prefetch_factor=prefetch_factor,
    )
    amp_enabled = _use_amp(config["training"], device)
    return datasets, loader, label_names, device, batch_size, eval_num_workers, amp_enabled


def recover_run(run_dir: Path) -> dict:
    last_ckpt_path = run_dir / "last_checkpoint.pt"
    best_model_path = run_dir / "best_model.pt"
    best_ckpt_path = run_dir / "best_checkpoint.pt"
    if not best_ckpt_path.exists():
        alt_best_ckpts = sorted(run_dir.glob("best_checkpoint*.pt"))
        if not alt_best_ckpts:
            raise FileNotFoundError(f"未找到 best checkpoint: {run_dir}")
        best_ckpt_path = alt_best_ckpts[0]

    last_ckpt = _load_checkpoint(last_ckpt_path)
    best_ckpt = _load_checkpoint(best_ckpt_path)
    config = last_ckpt["config"]
    config["run"]["output_dir"] = str(run_dir.resolve())

    pd.DataFrame(last_ckpt.get("log_rows", [])).to_csv(run_dir / "train_log.csv", index=False)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    datasets, test_loader, label_names, device, batch_size, eval_num_workers, amp_enabled = _build_test_loader(config)
    model = build_model(config["model"]).to(device)
    state_dict = torch.load(best_model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    test_bundle = predict(model, test_loader, device)
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
            "eval_num_workers": eval_num_workers,
            "epochs_trained": int(last_ckpt["epoch"]),
            "n_train": int(len(datasets["train"])),
            "n_val": int(len(datasets["val"])),
            "n_test": int(len(datasets["test"])),
            "training_status": "recovered_from_checkpoint",
            "resumed_from": None,
            "last_checkpoint": str(last_ckpt_path.resolve()),
            "best_checkpoint": str(best_ckpt_path.resolve()),
            "best_metric": float(best_ckpt.get("best_metric", float("nan"))),
            "checkpoint_status": last_ckpt.get("status"),
        }
    )
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    component_metrics.to_csv(run_dir / "component_metrics.csv", index=False)
    save_predictions(run_dir / "predictions.csv", test_bundle, split="test", label_names=label_names)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="从 deep learning checkpoint 补回 run 产物。")
    parser.add_argument("--run-dir", required=True, help="run 输出目录，内含 last_checkpoint.pt / best_model.pt")
    args = parser.parse_args()
    summary = recover_run(_resolve_run_dir(args.run_dir))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
