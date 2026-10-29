"""Single self-contained configuration for notebooks and library code."""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo / artifact paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
# XFAD_ARTIFACTS redirects checkpoints and results, so smoke runs cannot overwrite the
# artifacts of a real experiment.
ARTIFACTS = Path(os.environ.get("XFAD_ARTIFACTS") or REPO_ROOT / "artifacts")
CHECKPOINTS = ARTIFACTS / "checkpoints"
RESULTS = ARTIFACTS / "results"

# ---------------------------------------------------------------------------
# MIT-BIH splits (de Chazal et al.) - do not re-derive
# ---------------------------------------------------------------------------
MITBIH_EXCLUDED_RECORDS = (102, 104, 107, 217)
MITBIH_DS1 = (
    101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124,
    201, 203, 205, 207, 208, 209, 215, 220, 223, 230,
)
MITBIH_DS2_TEST = (
    100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212,
    213, 214, 219, 221, 222, 228, 231, 232, 233, 234,
)
MITBIH_DS1_VAL = (119, 124, 215, 220, 223)
MITBIH_DS1_TRAIN = tuple(r for r in MITBIH_DS1 if r not in MITBIH_DS1_VAL)
MITBIH_EXCLUDED = list(MITBIH_EXCLUDED_RECORDS)
MITBIH_DS2 = list(MITBIH_DS2_TEST)
MITBIH_NORMAL_CLASS = "N"
MITBIH_FS = 360

AAMI_MAP = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    "A": "S", "a": "S", "J": "S", "S": "S",
    "V": "V", "E": "V",
    "F": "F",
    "/": "Q", "f": "Q", "Q": "Q",
}
MITBIH_AAMI_CLASS_LIST = ["N", "S", "V", "F", "Q"]
AAMI_CLASS_TO_IDX = {c: i for i, c in enumerate(MITBIH_AAMI_CLASS_LIST)}

# ---------------------------------------------------------------------------
# PTB-XL rhythm task (12-class, single-label; normal = SR not NORM)
# ---------------------------------------------------------------------------
PTBXL_RHYTHM_CLASSES = (
    "SR", "AFIB", "STACH", "SARRH", "SBRAD", "PACE",
    "SVARR", "BIGU", "AFLT", "SVTAC", "PSVT", "TRIGU",
)
PTBXL_NORMAL_CLASS = "SR"
PTBXL_TRAIN_FOLDS = (1, 2, 3, 4, 5, 6, 7, 8)
PTBXL_VAL_FOLD = 9
PTBXL_TEST_FOLD = 10
PTBXL_FS = 500
PTBXL_CLASS_TO_IDX = {c: i for i, c in enumerate(PTBXL_RHYTHM_CLASSES)}
PTBXL_NOISE_COLS = ["baseline_drift", "static_noise", "burst_noise", "electrodes_problems"]

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
TARGET_FS = 250
COMMON_FS = TARGET_FS
KD_TEMPERATURE_CANDIDATES = (2.0, 4.0, 8.0)
KD_TEMPERATURE_DEFAULT = 4.0
KD_WEIGHT_HARD = 0.3
KD_WEIGHT_RESPONSE = 0.5
KD_WEIGHT_FEATURE = 0.2
KD_WEIGHT_SALIENCY = 0.3
MULTITASK_FALLBACK_AUROC_MARGIN = 0.02
SENSITIVITY_LOSS_BUDGET = 0.02
GATE_WEIGHTS_BEAT = (1 / 3, 1 / 3, 1 / 3)
GATE_WEIGHTS_RECORD = (1 / 6, 5 / 12, 5 / 12)
GATE_MIN_FALSE_ALARMS_FOR_LR = 50
# Share of alarm events allowed into the Stage-B escalation band (bounds connectivity cost).
GATE_ESCALATION_BUDGET = 0.20
BOOTSTRAP_N = 1000
SEED = 42

# Saliency-matching distillation (NB06 ablation). The baseline student ignores this.
STUDENT_VARIANTS = ("baseline", "saliency")
STUDENT_TRUNK_CHANNELS = 64
STUDENT_EPOCHS = 25
TEACHER_EPOCHS = 25
EARLY_STOPPING_PATIENCE = 5
USE_CLASS_WEIGHTS = True

# ---------------------------------------------------------------------------
# Data paths - Colab Drive when mounted, else repo root
# ---------------------------------------------------------------------------
_colab_drive = Path("/content/drive/MyDrive/XFAD")
_env_root = os.environ.get("XFAD_ROOT")
if _env_root and Path(_env_root).exists():
    DATA_ROOT = Path(_env_root)
elif _colab_drive.exists():
    DATA_ROOT = _colab_drive
else:
    DATA_ROOT = REPO_ROOT

DRIVE_ROOT = str(DATA_ROOT)
LOCAL_DATA = os.environ.get(
    "XFAD_LOCAL_DATA",
    "/content/data" if _colab_drive.exists() else str(REPO_ROOT / "data" / "cache"),
)
MITBIH_RAW = DATA_ROOT / "data" / "raw" / "mitbih"
PTBXL_RAW = DATA_ROOT / "data" / "raw" / "ptbxl"
MITBIH_PROCESSED = DATA_ROOT / "data" / "processed" / "mitbih"
PTBXL_PROCESSED = DATA_ROOT / "data" / "processed" / "ptbxl"
MITBIH_RAW_LEGACY = DATA_ROOT / "data" / "raw" / "mitdb"
MODEL_DIR = str(CHECKPOINTS)
RESULTS_DIR = str(RESULTS)
