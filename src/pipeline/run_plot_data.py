from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED_LOG_COLUMNS = ("epoch", "train_loss", "val_loss", "val_macro_RMSE", "val_macro_MAE")
REQUIRED_COMPONENT_METRIC_COLUMNS = ("component", "RMSE", "MAE", "R2")


@dataclass(frozen=True)
class TrainingRunLog:
    run_dir: Path
    relative_name: str
    display_name: str
    frame: pd.DataFrame
    summary: dict[str, Any] | None
    config: dict[str, Any] | None
    best_epoch: int
    best_val_macro_rmse: float


@dataclass(frozen=True)
class RunAnalysisBundle:
    run_dir: Path
    relative_name: str
    display_name: str
    run_name: str
    model_name: str | None
    training_log: pd.DataFrame
    predictions: pd.DataFrame
    component_metrics: pd.DataFrame
    summary: dict[str, Any]
    config: dict[str, Any] | None
    components: tuple[str, ...]
    component_display_names: dict[str, str]
    best_epoch: int
    best_val_macro_rmse: float

    @property
    def frame(self) -> pd.DataFrame:
        return self.training_log


def resolve_cli_path(value: str, root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def safe_slug(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", value.strip())
    return cleaned.strip("_") or "run"


def run_artifact_name(root: Path, run_dir: Path) -> str:
    relative = run_dir.relative_to(root)
    parts = [safe_slug(part) for part in relative.parts if part not in ("", ".")]
    if not parts:
        return safe_slug(run_dir.name)
    return "__".join(parts)


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_warning(run_dir: Path, reason: str) -> str:
    return f"跳过 {run_dir}: {reason}"


def load_training_run_log(root: Path, run_dir: Path) -> tuple[TrainingRunLog | None, str | None]:
    log_path = run_dir / "train_log.csv"
    try:
        frame = pd.read_csv(log_path)
    except pd.errors.EmptyDataError:
        return None, build_warning(run_dir, "train_log.csv 为空")
    if frame.empty:
        return None, build_warning(run_dir, "train_log.csv 没有有效行")

    missing = [column for column in REQUIRED_LOG_COLUMNS if column not in frame.columns]
    if missing:
        return None, build_warning(run_dir, f"缺少必要列: {', '.join(missing)}")

    ordered = frame.sort_values("epoch").reset_index(drop=True)
    best_idx = ordered["val_macro_RMSE"].idxmin()
    best_epoch = int(ordered.loc[best_idx, "epoch"])
    best_val_macro_rmse = float(ordered.loc[best_idx, "val_macro_RMSE"])
    return TrainingRunLog(
        run_dir=run_dir,
        relative_name=run_artifact_name(root, run_dir),
        display_name=run_dir.name,
        frame=ordered,
        summary=load_json_if_exists(run_dir / "summary.json"),
        config=load_json_if_exists(run_dir / "config.json"),
        best_epoch=best_epoch,
        best_val_macro_rmse=best_val_macro_rmse,
    ), None


def _require_file(run_dir: Path, filename: str) -> Path:
    path = run_dir / filename
    if not path.exists():
        raise ValueError(f"缺少必需文件: {filename}")
    return path


def _normalize_label_name(value: object) -> str:
    text = str(value)
    if text.startswith("x_"):
        return text[2:]
    return text


def _infer_components_from_predictions(predictions: pd.DataFrame) -> tuple[str, ...]:
    components: list[str] = []
    for column in predictions.columns:
        if not column.startswith("y_true_"):
            continue
        component = column[len("y_true_") :]
        pred_column = f"y_pred_{component}"
        if pred_column not in predictions.columns:
            raise ValueError(f"predictions.csv 缺少配对列: {pred_column}")
        components.append(component)
    if not components:
        raise ValueError("predictions.csv 未找到 y_true_*/y_pred_* 组件列")
    return tuple(components)


def _build_component_display_names(summary: dict[str, Any], components: tuple[str, ...]) -> dict[str, str]:
    raw_labels = summary.get("label_names")
    if isinstance(raw_labels, list) and len(raw_labels) == len(components):
        return {
            component: _normalize_label_name(label)
            for component, label in zip(components, raw_labels, strict=False)
        }
    return {component: component for component in components}


def _order_component_metrics(frame: pd.DataFrame, components: tuple[str, ...]) -> pd.DataFrame:
    missing_columns = [column for column in REQUIRED_COMPONENT_METRIC_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"component_metrics.csv 缺少必要列: {', '.join(missing_columns)}")

    ordered = frame.copy()
    ordered["component"] = ordered["component"].astype(str)
    missing_components = [component for component in components if component not in set(ordered["component"])]
    if missing_components:
        raise ValueError(f"component_metrics.csv 缺少组件行: {', '.join(missing_components)}")

    component_order = {component: idx for idx, component in enumerate(components)}
    ordered = ordered.loc[ordered["component"].isin(component_order)].copy()
    ordered["_plot_order"] = ordered["component"].map(component_order)
    ordered = ordered.sort_values("_plot_order").drop(columns="_plot_order").reset_index(drop=True)
    return ordered


def load_run_analysis_bundle(run_dir: Path, require_report_files: bool = True) -> RunAnalysisBundle:
    training_run, warning = load_training_run_log(run_dir.parent, run_dir)
    if warning is not None or training_run is None:
        raise ValueError(warning or f"无法加载训练日志: {run_dir}")

    predictions_path = _require_file(run_dir, "predictions.csv")
    component_metrics_path = _require_file(run_dir, "component_metrics.csv") if require_report_files else run_dir / "component_metrics.csv"
    summary_path = _require_file(run_dir, "summary.json") if require_report_files else run_dir / "summary.json"

    predictions = pd.read_csv(predictions_path)
    components = _infer_components_from_predictions(predictions)

    summary = load_json_if_exists(summary_path)
    if summary is None:
        raise ValueError("缺少必需文件: summary.json")

    component_metrics = pd.read_csv(component_metrics_path)
    component_metrics = _order_component_metrics(component_metrics, components)
    component_display_names = _build_component_display_names(summary, components)

    model_name = None
    if summary.get("model"):
        model_name = str(summary["model"])
    elif training_run.config and training_run.config.get("model", {}).get("name"):
        model_name = str(training_run.config["model"]["name"])

    run_name = str(summary.get("run_name") or training_run.display_name)

    return RunAnalysisBundle(
        run_dir=run_dir,
        relative_name=training_run.relative_name,
        display_name=training_run.display_name,
        run_name=run_name,
        model_name=model_name,
        training_log=training_run.frame,
        predictions=predictions,
        component_metrics=component_metrics,
        summary=summary,
        config=training_run.config,
        components=components,
        component_display_names=component_display_names,
        best_epoch=training_run.best_epoch,
        best_val_macro_rmse=training_run.best_val_macro_rmse,
    )
