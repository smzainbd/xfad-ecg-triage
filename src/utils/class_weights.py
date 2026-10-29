"""Inverse-frequency class weights for imbalanced ECG labels."""

from __future__ import annotations

import numpy as np
import torch


def class_weights_from_labels(labels, num_classes: int) -> torch.Tensor:
    """Balanced weights: n_samples / (n_classes * count_c), with a floor of one sample per class."""
    arr = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(arr, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = arr.size / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)
