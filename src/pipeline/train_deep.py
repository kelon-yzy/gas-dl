from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
DL_ROOT = SRC_ROOT / "dl"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(DL_ROOT) not in sys.path:
    sys.path.insert(0, str(DL_ROOT))

from pipeline.cli_progress import build_cli_progress
from training.train import load_config, train_config


def _resolve_project_output_dir(value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((ROOT / path).resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a deep-learning config for V3.1 dual-channel data.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--ui", action="store_true", help="优先启用 CLI UI；若终端不兼容则自动退回普通日志")
    parser.add_argument("--no-ui", action="store_true", help="禁用 CLI UI，始终使用普通日志输出")
    parser.add_argument("--resume", default=None, help="从指定 checkpoint 恢复训练")
    parser.add_argument("--checkpoint-every", type=int, default=0, help="每 N 个 epoch 保存一次 epoch_XXXX.pt，0=不保存")
    parser.add_argument("--no-resume-rng", dest="restore_rng", action="store_false", default=True, help="恢复时不还原随机数状态")
    parser.add_argument("--stop-after-epoch", type=int, default=None, help="训练到指定 epoch 后保存 checkpoint 并退出（测试用）")
    args = parser.parse_args()
    progress = build_cli_progress(force=args.ui, disable=args.no_ui)
    config = load_config(Path(args.config))
    config["run"]["output_dir"] = _resolve_project_output_dir(config["run"]["output_dir"])
    if args.seed is not None:
        output_dir = Path(config["run"]["output_dir"])
        config["run"]["seed"] = args.seed
        config["run"]["name"] = re.sub(r"_seed\d+$", f"_seed{args.seed}", config["run"]["name"])
        config["run"]["output_dir"] = str(output_dir.parent / config["run"]["name"])
    if args.output_root is not None:
        output_root = Path(args.output_root)
        if not output_root.is_absolute():
            output_root = ROOT / output_root
        config["run"]["output_dir"] = str(output_root / config["run"]["name"])
    config["_cli_progress"] = progress
    summary = train_config(
        config,
        epochs_override=args.epochs,
        resume_path=args.resume,
        checkpoint_every=args.checkpoint_every,
        restore_rng=args.restore_rng,
        stop_after_epoch=args.stop_after_epoch,
    )
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(json.dumps({"finished_at": stamp, "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

