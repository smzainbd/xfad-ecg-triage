"""PTB-XL 12-class rhythm dataset (single-label, not multi-label)."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config import (
    COMMON_FS,
    PTBXL_CLASS_TO_IDX,
    PTBXL_FS,
    PTBXL_NORMAL_CLASS,
    PTBXL_RAW,
    PTBXL_RHYTHM_CLASSES,
)
from src.data.preprocessing import resample_signal, zscore_per_channel


def _parse_scp_codes(val) -> dict:
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        return ast.literal_eval(val)
    return {}


def _primary_rhythm_label(scp_codes: dict) -> str | None:
    for cls in PTBXL_RHYTHM_CLASSES:
        if cls in scp_codes:
            return cls
    return None


def load_ptbxl_samples(
    folds: list[int],
    *,
    record_sec: float = 10.0,
    dry_run: bool = False,
    with_groups: bool = False,
):
    """Load and preprocess PTB-XL records for the requested folds.

    With ``with_groups=True`` also returns the patient id of every record, so bootstrap
    resampling can respect patient boundaries.
    """
    csv_path = PTBXL_RAW / "ptbxl_database.csv"
    if dry_run or not csv_path.exists():
        n = 20 if dry_run else 200
        win = int(COMMON_FS * record_sec)
        synthetic = [
            (np.random.randn(12, win).astype(np.float32), np.random.randint(0, len(PTBXL_RHYTHM_CLASSES)))
            for _ in range(n)
        ]
        if with_groups:
            return synthetic, [f"synthetic_{i % 10}" for i in range(n)]
        return synthetic

    import wfdb

    df = pd.read_csv(csv_path, index_col="ecg_id")
    df = df[df["strat_fold"].isin(folds)]
    samples: list[tuple[np.ndarray, int]] = []
    groups: list[str] = []
    for ecg_id, row in df.iterrows():
        label = _primary_rhythm_label(_parse_scp_codes(row["scp_codes"]))
        if label is None:
            continue
        hea = PTBXL_RAW / row["filename_hr"]
        if not hea.with_suffix(".dat").exists():
            continue
        sig, _ = wfdb.rdsamp(str(hea))
        sig = sig.T
        sig = resample_signal(sig, PTBXL_FS, COMMON_FS)
        sig = zscore_per_channel(sig)
        y = PTBXL_CLASS_TO_IDX[label]
        samples.append((sig, y))
        patient = row.get("patient_id", ecg_id)
        groups.append(str(int(patient)) if pd.notna(patient) else str(ecg_id))

    if not samples:
        raise RuntimeError(f"No PTB-XL records loaded for folds {folds}")
    if with_groups:
        return samples, groups
    return samples


class PTBXLDataset(Dataset):
    """Record-level PTB-XL rhythm classification. Uses preprocessed cache when available."""

    def __init__(
        self,
        folds: list[int],
        *,
        dry_run: bool = False,
        record_sec: float = 10.0,
        use_cache: bool = True,
    ):
        self.folds = list(folds)
        self.dry_run = dry_run
        self.record_sec = record_sec
        self.samples: list[tuple[np.ndarray, int]] = []
        self.groups: list[str] = []  # patient id per record; empty for pre-group caches
        self._load(use_cache=use_cache)

    def _load(self, *, use_cache: bool) -> None:
        if use_cache and not self.dry_run:
            from src.data.cache import (
                load_samples_npz,
                ptbxl_cache_name,
                resolve_ptbxl_processed_dir,
            )

            cache_file = resolve_ptbxl_processed_dir() / ptbxl_cache_name(self.folds)
            if cache_file.exists():
                self.samples, self.groups = load_samples_npz(cache_file, with_groups=True)
                return

        self.samples, self.groups = load_ptbxl_samples(
            self.folds,
            record_sec=self.record_sec,
            dry_run=self.dry_run,
            with_groups=True,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        x, y = self.samples[idx]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)

    @staticmethod
    def normal_class_idx() -> int:
        return PTBXL_CLASS_TO_IDX[PTBXL_NORMAL_CLASS]
