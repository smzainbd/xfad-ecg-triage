"""Seed setting, checkpoint IO, and notebook path helpers."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch


def setup_notebook_paths() -> Path:
    """Add repo root to sys.path when running from notebooks/."""
    import sys

    repo = Path(__file__).resolve().parents[2]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    return repo


def resolve_dry_run(default: bool = True) -> bool:
    """Read XFAD_DRY_RUN once, so every notebook agrees on what the flag means.

    ``XFAD_DRY_RUN=0`` (or false/no/off) means a full experiment run; anything else,
    including an unset variable, keeps the fast smoke-test path.
    """
    raw = os.environ.get("XFAD_DRY_RUN")
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def resolve_student_variant(default: str = "baseline") -> str:
    """Which student is being trained or evaluated: 'baseline' or 'saliency'.

    Notebooks no longer set this manually. Use ``student_variants_for_run()`` instead.
    The env var remains available for smoke tests and scripted overrides.
    """
    from src.config import STUDENT_VARIANTS

    variant = (os.environ.get("XFAD_STUDENT_VARIANT") or default).strip().lower()
    if variant not in STUDENT_VARIANTS:
        raise ValueError(f"Unknown student variant {variant!r}; expected one of {STUDENT_VARIANTS}")
    return variant


def student_variants_for_run(dry_run: bool | None = None) -> tuple[str, ...]:
    """Student arms to process in one notebook run.

  - Smoke test (``DRY_RUN=True``): baseline only — fast path.
  - Full run (``DRY_RUN=False``): baseline + saliency — everything NB06 needs.
  - Optional ``XFAD_STUDENT_VARIANT`` env override (smoke runner / scripts only).
    """
    from src.config import STUDENT_VARIANTS

    override = os.environ.get("XFAD_STUDENT_VARIANT")
    if override and override.strip():
        variant = override.strip().lower()
        if variant not in STUDENT_VARIANTS:
            raise ValueError(f"Unknown student variant {variant!r}; expected one of {STUDENT_VARIANTS}")
        return (variant,)
    if dry_run is None:
        dry_run = resolve_dry_run()
    return ("baseline",) if dry_run else STUDENT_VARIANTS


def print_run_banner(*, notebook: str, dry_run: bool, variants: tuple[str, ...] | None = None) -> None:
    """Print a consistent run-mode summary after the dry-run switch cell."""
    mode = "SMOKE TEST" if dry_run else "FULL PAPER RUN"
    print("=" * 72)
    print(f"  {notebook}  |  {mode}")
    print("=" * 72)
    if variants is not None:
        print(f"Student variants (automatic): {list(variants)}")
    print(f"XFAD_DRY_RUN = {os.environ.get('XFAD_DRY_RUN', '(unset)')}")
    print("=" * 72)


def variant_suffix(variant: str) -> str:
    """Filename suffix for a variant's artifacts ('' for the baseline)."""
    return "" if variant == "baseline" else f"_{variant}"


def student_checkpoint_path(variant: str = "baseline") -> str:
    from src.config import MODEL_DIR

    return os.path.join(MODEL_DIR, f"student{variant_suffix(variant)}.pt")


def variant_result_path(name: str, variant: str = "baseline", *, ext: str = ".json") -> str:
    """Path for a per-variant result file, e.g. ``gate_eval_ds2_saliency.json``."""
    from src.config import RESULTS_DIR

    return os.path.join(RESULTS_DIR, f"{name}{variant_suffix(variant)}{ext}")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_legacy_config() -> None:
    """Backfill constants missing from stale Drive copies of ``src/config.py``."""
    import src.config as cfg

    defaults = {
        "KD_WEIGHT_SALIENCY": 0.3,
        "STUDENT_TRUNK_CHANNELS": 64,
        "STUDENT_EPOCHS": 25,
        "TEACHER_EPOCHS": 25,
        "EARLY_STOPPING_PATIENCE": 5,
        "USE_CLASS_WEIGHTS": True,
    }
    patched = [name for name, value in defaults.items() if not hasattr(cfg, name)]
    for name in patched:
        setattr(cfg, name, defaults[name])
    if patched:
        print("Patched legacy config:", ", ".join(patched))


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def resolve_mitbih_raw_dir() -> Path:
    from src.config import MITBIH_RAW, MITBIH_RAW_LEGACY

    if MITBIH_RAW.exists() and any(MITBIH_RAW.glob("*.dat")):
        return MITBIH_RAW
    if MITBIH_RAW_LEGACY.exists() and any(MITBIH_RAW_LEGACY.glob("*.dat")):
        return MITBIH_RAW_LEGACY
    return MITBIH_RAW
