from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
)


LABEL_COLUMNS = ("cleanliness", "location", "luxury", "family_friendly")


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_json_ready(payload), indent=2), encoding="utf-8")


def classification_metrics(y_true: Any, y_pred: Any, label_columns: tuple[str, ...] = LABEL_COLUMNS) -> dict[str, Any]:
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_array,
        y_pred_array,
        average=None,
        zero_division=0,
    )
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        y_true_array,
        y_pred_array,
        average="micro",
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true_array,
        y_pred_array,
        average="macro",
        zero_division=0,
    )

    per_label = {
        label: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(label_columns)
    }

    return {
        "micro_precision": float(micro_precision),
        "micro_recall": float(micro_recall),
        "micro_f1": float(micro_f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "subset_accuracy": float(accuracy_score(y_true_array, y_pred_array)),
        "per_label": per_label,
    }


def regression_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)
    mse = mean_squared_error(y_true_array, y_pred_array)

    return {
        "mae": float(mean_absolute_error(y_true_array, y_pred_array)),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true_array, y_pred_array)),
    }
