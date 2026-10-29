"""Classification metrics for beat- and record-level tasks."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def sensitivity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(recall_score(y_true, y_pred, zero_division=0))


def specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return float(tn / (tn + fp + 1e-8))


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, *, binary: bool = True) -> dict[str, float]:
    if binary:
        y_pred = (y_score >= 0.5).astype(int)
        return {
            "sensitivity": sensitivity(y_true, y_pred),
            "specificity": specificity(y_true, y_pred),
            "ppv": float(precision_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "auroc": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else float("nan"),
            "auprc": float(average_precision_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else float("nan"),
        }
    y_pred = y_score.argmax(axis=1)
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy": float((y_true == y_pred).mean()),
    }
