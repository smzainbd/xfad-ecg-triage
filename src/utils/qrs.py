"""Lightweight QRS detection for saliency-mass overlap (Pan-Tompkins inspired)."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


def detect_qrs_windows(x: np.ndarray, fs: float, min_distance_ms: float = 300.0) -> list[tuple[int, int]]:
    """Return list of (start, end) sample indices for QRS complexes."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=0)

    nyq = fs / 2.0
    b, a = butter(2, [5 / nyq, 15 / nyq], btype="band")
    filtered = filtfilt(b, a, x)
    diff = np.diff(filtered, prepend=filtered[0])
    squared = diff ** 2

    min_dist = max(1, int(min_distance_ms * fs / 1000.0))
    peaks, _ = find_peaks(squared, distance=min_dist, height=np.percentile(squared, 90))

    half_width = int(0.08 * fs)  # ~80 ms each side
    windows = []
    for p in peaks:
        start = max(0, int(p) - half_width)
        end = min(len(x), int(p) + half_width)
        if end > start:
            windows.append((start, end))
    return windows
