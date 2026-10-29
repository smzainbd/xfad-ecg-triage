"""Patient-level bootstrap confidence intervals."""

from __future__ import annotations

from typing import Callable

import numpy as np


def patient_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    patient_ids: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    patients = np.unique(patient_ids)
    patient_to_idx = {p: np.where(patient_ids == p)[0] for p in patients}
    scores = []
    for _ in range(n_boot):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        idx = np.concatenate([patient_to_idx[p] for p in sampled])
        scores.append(metric_fn(y_true[idx], y_pred[idx]))
    scores = np.sort(scores)
    lo = scores[int((alpha / 2) * n_boot)]
    hi = scores[int((1 - alpha / 2) * n_boot)]
    return float(np.mean(scores)), float(lo), float(hi)
