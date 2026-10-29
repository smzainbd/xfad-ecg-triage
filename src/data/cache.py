"""Preprocessed tensor caches for fast notebook training."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np

from src.config import (
    LOCAL_DATA,
    MITBIH_DS1_TRAIN,
    MITBIH_DS1_VAL,
    MITBIH_DS2,
    MITBIH_PROCESSED,
    PTBXL_PROCESSED,
    PTBXL_TEST_FOLD,
    PTBXL_TRAIN_FOLDS,
    PTBXL_VAL_FOLD,
)
from src.data.mitbih import load_mitbih_samples
from src.data.ptbxl import load_ptbxl_samples


def _local_processed_root() -> Path:
    return Path(os.environ.get("XFAD_LOCAL_DATA", LOCAL_DATA)) / "processed"


def resolve_mitbih_processed_dir() -> Path:
    """Prefer Colab-local processed cache when present."""
    local = _local_processed_root() / "mitbih"
    if any(local.glob("*.npz")):
        return local
    return MITBIH_PROCESSED


def resolve_ptbxl_processed_dir() -> Path:
    local = _local_processed_root() / "ptbxl"
    if any(local.glob("*.npz")):
        return local
    return PTBXL_PROCESSED


def mitbih_cache_name(record_ids: Iterable[int]) -> str:
    ids = sorted(set(record_ids))
    if tuple(ids) == tuple(sorted(MITBIH_DS1_TRAIN)):
        return "ds1_train.npz"
    if tuple(ids) == tuple(sorted(MITBIH_DS1_VAL)):
        return "ds1_val.npz"
    if tuple(ids) == tuple(sorted(MITBIH_DS2)):
        return "ds2_test.npz"
    return f"records_{'-'.join(map(str, ids))}.npz"


def ptbxl_cache_name(folds: Iterable[int]) -> str:
    fold_list = sorted(set(folds))
    if fold_list == sorted(PTBXL_TRAIN_FOLDS):
        return "train_folds_1-8.npz"
    if fold_list == [PTBXL_VAL_FOLD]:
        return f"fold_{PTBXL_VAL_FOLD}_val.npz"
    if fold_list == [PTBXL_TEST_FOLD]:
        return f"fold_{PTBXL_TEST_FOLD}_test.npz"
    return f"folds_{'-'.join(map(str, fold_list))}.npz"


def cache_path(processed_dir: Path, name: str) -> Path:
    return processed_dir / name


def save_samples_npz(
    path: Path,
    samples: list[tuple[np.ndarray, int]],
    groups: list[str] | None = None,
) -> dict:
    """Cache samples, storing per-sample record/patient ids when supplied."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not samples:
        raise ValueError(f"No samples to cache at {path}")
    x = np.stack([s[0] for s in samples]).astype(np.float32)
    y = np.array([s[1] for s in samples], dtype=np.int64)
    arrays = {"X": x, "y": y}
    if groups is not None:
        if len(groups) != len(samples):
            raise ValueError(f"groups length {len(groups)} != samples length {len(samples)}")
        arrays["groups"] = np.array([str(g) for g in groups])
    np.savez_compressed(path, **arrays)
    return {
        "path": str(path),
        "n_samples": int(len(samples)),
        "X_shape": list(x.shape),
        "has_groups": groups is not None,
    }


def load_samples_npz(path: Path, *, with_groups: bool = False):
    """Load a cache; older caches without group ids yield an empty group list."""
    data = np.load(path)
    x = data["X"]
    y = data["y"]
    samples = [(x[i], int(y[i])) for i in range(len(y))]
    if not with_groups:
        return samples
    groups = [str(g) for g in data["groups"]] if "groups" in data.files else []
    return samples, groups


def cache_has_groups(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        return "groups" in np.load(path).files
    except (OSError, ValueError):
        return False


def build_mitbih_cache(
    record_ids: Iterable[int],
    *,
    processed_dir: Path | None = None,
    dry_run: bool = False,
    require_groups: bool = True,
) -> dict:
    processed_dir = processed_dir or resolve_mitbih_processed_dir()
    name = mitbih_cache_name(record_ids)
    out = cache_path(processed_dir, name)
    if out.exists() and not dry_run and (cache_has_groups(out) or not require_groups):
        data = np.load(out)
        return {
            "status": "present",
            "path": str(out),
            "n_samples": int(len(data["y"])),
            "has_groups": "groups" in data.files,
        }
    samples, groups = load_mitbih_samples(record_ids, dry_run=dry_run, with_groups=True)
    meta = save_samples_npz(out, samples, groups)
    return {"status": "built", **meta}


def build_ptbxl_cache(
    folds: Iterable[int],
    *,
    processed_dir: Path | None = None,
    dry_run: bool = False,
    require_groups: bool = True,
) -> dict:
    processed_dir = processed_dir or resolve_ptbxl_processed_dir()
    name = ptbxl_cache_name(folds)
    out = cache_path(processed_dir, name)
    if out.exists() and not dry_run and (cache_has_groups(out) or not require_groups):
        data = np.load(out)
        return {
            "status": "present",
            "path": str(out),
            "n_samples": int(len(data["y"])),
            "has_groups": "groups" in data.files,
        }
    samples, groups = load_ptbxl_samples(folds, dry_run=dry_run, with_groups=True)
    meta = save_samples_npz(out, samples, groups)
    return {"status": "built", **meta}


def build_nb01_caches(*, dry_run: bool = False, local_first: bool = True) -> dict:
    """Build caches used by NB01 (train + val splits)."""
    if local_first and Path("/content").exists():
        mit_dir = _local_processed_root() / "mitbih"
        ptb_dir = _local_processed_root() / "ptbxl"
    else:
        mit_dir = MITBIH_PROCESSED
        ptb_dir = PTBXL_PROCESSED

    mit_dir.mkdir(parents=True, exist_ok=True)
    ptb_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "mitbih": {
            "train": build_mitbih_cache(MITBIH_DS1_TRAIN, processed_dir=mit_dir, dry_run=dry_run),
            "val": build_mitbih_cache(MITBIH_DS1_VAL, processed_dir=mit_dir, dry_run=dry_run),
        },
        "ptbxl": {
            "train": build_ptbxl_cache(PTBXL_TRAIN_FOLDS, processed_dir=ptb_dir, dry_run=dry_run),
            "val": build_ptbxl_cache([PTBXL_VAL_FOLD], processed_dir=ptb_dir, dry_run=dry_run),
        },
        "local_mitbih_dir": str(mit_dir),
        "local_ptbxl_dir": str(ptb_dir),
    }
    manifest = mit_dir / "cache_manifest.json"
    manifest.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def sync_processed_to_drive(*, dry_run: bool = False) -> dict:
    """Copy local processed caches to Drive for persistence across Colab sessions."""
    if dry_run or not Path("/content").exists():
        return {"status": "skipped", "reason": "dry_run or not colab"}

    copied = {}
    for name, local_sub, drive_sub in (
        ("mitbih", _local_processed_root() / "mitbih", MITBIH_PROCESSED),
        ("ptbxl", _local_processed_root() / "ptbxl", PTBXL_PROCESSED),
    ):
        if not local_sub.exists() or not any(local_sub.glob("*.npz")):
            continue
        drive_sub.mkdir(parents=True, exist_ok=True)
        rsync = shutil.which("rsync")
        if rsync:
            subprocess_run = __import__("subprocess").check_call
            subprocess_run([rsync, "-a", f"{local_sub}/", f"{drive_sub}/"])
        else:
            for path in local_sub.rglob("*"):
                rel = path.relative_to(local_sub)
                out = drive_sub / rel
                if path.is_dir():
                    out.mkdir(parents=True, exist_ok=True)
                else:
                    out.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, out)
        copied[name] = {"src": str(local_sub), "dest": str(drive_sub)}
    return {"status": "copied", "datasets": copied}
