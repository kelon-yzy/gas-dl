from __future__ import annotations

import numpy as np
import pandas as pd

LABELS = ["H2", "CH4", "CO2", "N2"]


def _display_labels(label_names):
    if label_names is None:
        return LABELS
    return [str(name).replace("x_", "") for name in label_names]


def regression_metrics(y_true, y_pred, label_names=None) -> tuple[dict, pd.DataFrame]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}")
    labels = _display_labels(label_names)
    if y_true.shape[1] != len(labels):
        raise ValueError(f"Label count mismatch: y.shape[1]={y_true.shape[1]}, labels={labels}")

    rows = []
    summary = {}
    for i, label in enumerate(labels):
        err = y_pred[:, i] - y_true[:, i]
        rmse = float(np.sqrt(np.mean(err**2)))
        mae = float(np.mean(np.abs(err)))
        denom = float(np.sum((y_true[:, i] - np.mean(y_true[:, i])) ** 2))
        r2 = float(1.0 - np.sum(err**2) / denom) if denom > 1e-12 else float("nan")
        rows.append(
            {
                "component": label,
                "RMSE": rmse,
                "MAE": mae,
                "R2": r2,
            }
        )
        summary[f"{label}_RMSE"] = rmse
        summary[f"{label}_MAE"] = mae
        summary[f"{label}_R2"] = r2

    summary["macro_RMSE"] = float(np.mean([row["RMSE"] for row in rows]))
    summary["macro_MAE"] = float(np.mean([row["MAE"] for row in rows]))

    true_sum = y_true.sum(axis=1)
    pred_sum = y_pred.sum(axis=1)
    abs_sum_error = np.abs(pred_sum - true_sum)
    summary["mean_true_sum"] = float(np.mean(true_sum))
    summary["mean_pred_sum"] = float(np.mean(pred_sum))
    summary["std_pred_sum"] = float(np.std(pred_sum))
    summary["mean_abs_sum_error"] = float(np.mean(abs_sum_error))
    summary["max_abs_sum_error"] = float(np.max(abs_sum_error))
    return summary, pd.DataFrame(rows)
