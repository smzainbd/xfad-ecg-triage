"""Signal quality index (SQI) for gate Stage A."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt


def bandpower(x: np.ndarray, fs: float, low: float, high: float) -> float:
    nyq = fs / 2.0
    low_n = max(low / nyq, 1e-6)
    high_n = min(high / nyq, 0.999)
    if low_n >= high_n:
        return 0.0
    b, a = butter(2, [low_n, high_n], btype="band")
    filtered = filtfilt(b, a, x.astype(np.float64))
    return float(np.mean(filtered ** 2))


def compute_sqi(x: np.ndarray, fs: float) -> float:
    """SQI = 1 - clip(high_freq_power / ecg_band_power, 0, 1)."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=0)
    ecg_power = bandpower(x, fs, 0.5, 40.0)
    high_power = bandpower(x, fs, 40.0, fs / 2.0 - 1.0)
    if ecg_power <= 0:
        return 0.0
    ratio = np.clip(high_power / ecg_power, 0.0, 1.0)
    return float(1.0 - ratio)
