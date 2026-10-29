"""Resampling and normalisation helpers."""

from __future__ import annotations

import numpy as np
from scipy.signal import resample_poly


def resample_signal(x: np.ndarray, orig_fs: float, target_fs: float) -> np.ndarray:
    """Resample (channels, length) signal to target_fs."""
    if orig_fs == target_fs:
        return x.astype(np.float32)
    x = np.asarray(x, dtype=np.float64)
    ratio = target_fs / orig_fs
    # Use integer ratio approximation
    up = int(round(target_fs))
    down = int(round(orig_fs))
    from math import gcd
    g = gcd(up, down)
    up //= g
    down //= g
    if x.ndim == 1:
        return resample_poly(x, up, down).astype(np.float32)
    return np.stack([resample_poly(ch, up, down) for ch in x], axis=0).astype(np.float32)


def zscore_per_channel(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return (x - mean) / (std + eps)
