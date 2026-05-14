from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.train import load_config, train_config


EXPERIMENTS = [
    {
        "experiment_id": "V3-WF",
        "notes": "waveform + slow channels",
        "use_waveform": True,
    },
    {
        "experiment_id": "V3-SLOW",
        "notes": "slow-only ablation without waveform encoder",
        "use_waveform": False,
    },
]


def run_experiment(base_config: dict, experiment: dict, epochs: int | None) -> dict:
    config = copy.deepcopy(base_config)
    seed = int(config["run"].get("seed", 42))
    config["run"]["name"] = f"{experiment['experiment_id'].lower()}_seed{seed}"
    config["run"]["output_dir"] = str(ROOT / "outputs" / config["run"]["name"])
    config["model"]["use_waveform"] = experiment["use_waveform"]
    summary = train_config(config, epochs_override=epochs)
    summary.update(
        {
            "experiment_id": experiment["experiment_id"],
            "use_waveform": experiment["use_waveform"],
            "notes": experiment["notes"],
        }
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output-name", default=str(ROOT / "outputs" / "waveform_baseline_summary.csv"))
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds. Defaults to the seed in base config.")
    args = parser.parse_args()

    base = load_config(Path(args.base_config))
    if args.seeds is None:
        seeds = [int(base["run"].get("seed", 42))]
    else:
        seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]

    rows = []
    for seed in seeds:
        seeded_base = copy.deepcopy(base)
        seeded_base["run"]["seed"] = seed
        for experiment in EXPERIMENTS:
            rows.append(run_experiment(seeded_base, experiment, args.epochs))
    output = pd.DataFrame(rows)
    columns = [
        "experiment_id",
        "model",
        "seed",
        "use_waveform",
        "device",
        "macro_RMSE",
        "macro_MAE",
        "epochs_trained",
        "notes",
    ]
    output = output[[col for col in columns if col in output.columns]]
    output_path = Path(args.output_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(output.to_string(index=False))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
