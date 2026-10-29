"""Fast DataLoader helpers for training notebooks."""

from __future__ import annotations

import os

import torch
from torch.utils.data import DataLoader, Dataset


def make_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool = False,
    dry_run: bool = False,
) -> DataLoader:
    """Create a DataLoader with GPU-friendly settings when training for real."""
    kwargs: dict = {
        "batch_size": batch_size,
        "shuffle": shuffle,
    }
    if not dry_run and torch.cuda.is_available():
        num_workers = int(os.environ.get("XFAD_NUM_WORKERS", "2"))
        kwargs.update(
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )
    return DataLoader(dataset, **kwargs)
