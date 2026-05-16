from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
STATUS_PATH = ROOT / "outputs" / "STATUS.tsv"
STATUS_FIELDS = ("exp_id", "model", "seed", "status", "started", "finished", "macro_RMSE", "notes")


def _sanitize_text(value: str | None) -> str:
    if value is None:
        return ""
    return str(value).replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()


def _timestamp(value: str | None = None) -> str:
    if value is not None:
        return value
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_rows(status_path: Path = STATUS_PATH) -> list[dict[str, str]]:
    if not status_path.exists():
        return []
    with status_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    normalized = []
    for row in rows:
        normalized.append({field: row.get(field, "") for field in STATUS_FIELDS})
    return normalized


def save_rows(rows: list[dict[str, str]], status_path: Path = STATUS_PATH) -> None:
    status_path.parent.mkdir(parents=True, exist_ok=True)
    with status_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=STATUS_FIELDS, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in STATUS_FIELDS})


def _find_row(rows: list[dict[str, str]], exp_id: str, model: str, seed: int) -> dict[str, str] | None:
    seed_text = str(seed)
    for row in rows:
        if row["exp_id"] == exp_id and row["model"] == model and row["seed"] == seed_text:
            return row
    return None


def best_macro_rmse_from_grid_summary(summary_csv: Path) -> float | None:
    frame = pd.read_csv(summary_csv)
    if frame.empty or "macro_RMSE_pp" not in frame.columns:
        return None
    return float(frame["macro_RMSE_pp"].min())


def upsert_status(
    *,
    exp_id: str,
    model: str,
    seed: int,
    status: str,
    notes: str | None = None,
    started: str | None = None,
    finished: str | None = None,
    macro_rmse: str | None = None,
    status_path: Path = STATUS_PATH,
) -> dict[str, str]:
    rows = load_rows(status_path)
    row = _find_row(rows, exp_id, model, seed)
    if row is None:
        row = {field: "" for field in STATUS_FIELDS}
        row["exp_id"] = exp_id
        row["model"] = model
        row["seed"] = str(seed)
        rows.append(row)
    row["status"] = status
    if started is not None:
        row["started"] = started
    if finished is not None:
        row["finished"] = finished
    if macro_rmse is not None:
        row["macro_RMSE"] = macro_rmse
    if notes is not None:
        row["notes"] = _sanitize_text(notes)
    save_rows(rows, status_path)
    return row


def mark_running(
    *,
    exp_id: str,
    model: str,
    seed: int,
    notes: str,
    status_path: Path = STATUS_PATH,
    started_at: str | None = None,
) -> dict[str, str]:
    return upsert_status(
        exp_id=exp_id,
        model=model,
        seed=seed,
        status="running",
        notes=notes,
        started=_timestamp(started_at),
        finished="",
        macro_rmse="",
        status_path=status_path,
    )


def mark_finished(
    *,
    exp_id: str,
    model: str,
    seed: int,
    notes: str,
    summary_csv: Path | None = None,
    status_path: Path = STATUS_PATH,
    finished_at: str | None = None,
) -> dict[str, str]:
    macro_rmse = None
    if summary_csv is not None:
        best = best_macro_rmse_from_grid_summary(summary_csv)
        if best is not None:
            macro_rmse = f"{best:.4f}"
    return upsert_status(
        exp_id=exp_id,
        model=model,
        seed=seed,
        status="success",
        notes=notes,
        finished=_timestamp(finished_at),
        macro_rmse=macro_rmse,
        status_path=status_path,
    )


def mark_failed(
    *,
    exp_id: str,
    model: str,
    seed: int,
    notes: str,
    status_path: Path = STATUS_PATH,
    finished_at: str | None = None,
) -> dict[str, str]:
    return upsert_status(
        exp_id=exp_id,
        model=model,
        seed=seed,
        status="failed",
        notes=notes,
        finished=_timestamp(finished_at),
        status_path=status_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Update outputs/STATUS.tsv.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_arguments(target: argparse.ArgumentParser) -> None:
        target.add_argument("--exp-id", required=True)
        target.add_argument("--model", required=True)
        target.add_argument("--seed", type=int, required=True)
        target.add_argument("--notes", default="")
        target.add_argument("--status-path", default=str(STATUS_PATH))

    start_parser = subparsers.add_parser("start")
    add_common_arguments(start_parser)
    start_parser.add_argument("--started-at", default=None)

    finish_parser = subparsers.add_parser("finish")
    add_common_arguments(finish_parser)
    finish_parser.add_argument("--summary-csv", default=None)
    finish_parser.add_argument("--finished-at", default=None)

    fail_parser = subparsers.add_parser("fail")
    add_common_arguments(fail_parser)
    fail_parser.add_argument("--finished-at", default=None)

    args = parser.parse_args()
    status_path = Path(args.status_path)

    if args.command == "start":
        mark_running(
            exp_id=args.exp_id,
            model=args.model,
            seed=args.seed,
            notes=args.notes,
            status_path=status_path,
            started_at=args.started_at,
        )
        return
    if args.command == "finish":
        summary_csv = Path(args.summary_csv) if args.summary_csv else None
        mark_finished(
            exp_id=args.exp_id,
            model=args.model,
            seed=args.seed,
            notes=args.notes,
            summary_csv=summary_csv,
            status_path=status_path,
            finished_at=args.finished_at,
        )
        return
    mark_failed(
        exp_id=args.exp_id,
        model=args.model,
        seed=args.seed,
        notes=args.notes,
        status_path=status_path,
        finished_at=args.finished_at,
    )


if __name__ == "__main__":
    main()
