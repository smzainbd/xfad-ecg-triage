"""MIT-BIH beat dataset with optional real wfdb loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.config import AAMI_CLASS_TO_IDX, AAMI_MAP, COMMON_FS, MITBIH_FS
from src.data.preprocessing import resample_signal, zscore_per_channel
from src.utils.io import resolve_mitbih_raw_dir


def load_mitbih_samples(
    record_ids: list[int],
    *,
    beat_window_sec: float = 1.0,
    dry_run: bool = False,
    max_beats_per_record: int | None = None,
    with_groups: bool = False,
):
    """Load and preprocess MIT-BIH beats from raw wfdb files.

    With ``with_groups=True`` also returns the source record id of every beat, which
    patient-level bootstrap resampling needs.
    """
    raw_dir = resolve_mitbih_raw_dir()
    if dry_run or not any(raw_dir.glob("*.dat")):
        n = 20 if dry_run else 200
        win = int(COMMON_FS * beat_window_sec)
        synthetic = [
            (np.random.randn(2, win).astype(np.float32), np.random.randint(0, 5))
            for _ in range(n)
        ]
        if with_groups:
            ids = [str(r) for r in record_ids] or ["synthetic"]
            return synthetic, [ids[i % len(ids)] for i in range(n)]
        return synthetic

    import wfdb

    samples: list[tuple[np.ndarray, int]] = []
    groups: list[str] = []
    win = int(COMMON_FS * beat_window_sec)
    half = win // 2
    for rec in record_ids:
        base = raw_dir / str(rec)
        if not (base.with_suffix(".dat").exists() and base.with_suffix(".atr").exists()):
            continue
        sig, _ = wfdb.rdsamp(str(base))
        ann = wfdb.rdann(str(base), "atr")
        sig = sig.T
        sig = resample_signal(sig, MITBIH_FS, COMMON_FS)
        beats_this_record = 0
        for i, sym in enumerate(ann.symbol):
            if sym not in AAMI_MAP:
                continue
            center = int(ann.sample[i] * COMMON_FS / MITBIH_FS)
            start, end = center - half, center + half
            if start < 0 or end > sig.shape[1]:
                continue
            beat = zscore_per_channel(sig[:, start:end])
            label = AAMI_CLASS_TO_IDX[AAMI_MAP[sym]]
            samples.append((beat, label))
            groups.append(str(rec))
            beats_this_record += 1
            if max_beats_per_record is not None and beats_this_record >= max_beats_per_record:
                break

    if not samples:
        raise RuntimeError(f"No MIT-BIH beats loaded from {raw_dir}")
    if with_groups:
        return samples, groups
    return samples


class MITBIHDataset(Dataset):
    """Beat-level MIT-BIH dataset. Uses preprocessed cache when available."""

    def __init__(
        self,
        record_ids: list[int],
        *,
        beat_window_sec: float = 1.0,
        dry_run: bool = False,
        max_beats_per_record: int | None = None,
        use_cache: bool = True,
    ):
        self.record_ids = list(record_ids)
        self.beat_window_sec = beat_window_sec
        self.dry_run = dry_run
        self.max_beats_per_record = max_beats_per_record
        self.samples: list[tuple[np.ndarray, int]] = []
        self.groups: list[str] = []  # source record id per beat; empty for pre-group caches
        self._load(use_cache=use_cache)

    def _load(self, *, use_cache: bool) -> None:
        if use_cache and not self.dry_run:
            from src.data.cache import (
                load_samples_npz,
                mitbih_cache_name,
                resolve_mitbih_processed_dir,
            )

            cache_file = resolve_mitbih_processed_dir() / mitbih_cache_name(self.record_ids)
            if cache_file.exists():
                self.samples, self.groups = load_samples_npz(cache_file, with_groups=True)
                return

        self.samples, self.groups = load_mitbih_samples(
            self.record_ids,
            beat_window_sec=self.beat_window_sec,
            dry_run=self.dry_run,
            max_beats_per_record=self.max_beats_per_record,
            with_groups=True,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        x, y = self.samples[idx]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)
