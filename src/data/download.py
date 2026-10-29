"""Download MIT-BIH and PTB-XL into project raw-data directories."""

from __future__ import annotations

import os
import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Iterable

import pandas as pd

PTBXL_PHYSIONET_BASE = "https://physionet.org/files/ptb-xl/1.0.3/"
PTBXL_S3_PREFIX = "s3://physionet-open/ptb-xl/1.0.3"
PTBXL_METADATA_FILES = ("ptbxl_database.csv", "scp_statements.csv")
PTBXL_DB_NAME = "ptb-xl"
MITBIH_DB_NAME = "mitdb"
DEFAULT_COLAB_STAGE = Path("/content/data/raw/ptbxl")


def _require_wfdb():
    try:
        import wfdb
    except ImportError as exc:
        raise ImportError("wfdb is required for dataset download. Run: pip install wfdb") from exc
    return wfdb


def _download_url(url: str, dest: Path) -> bool:
    """Download a file if missing. Returns True when a new file was written."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return False
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception:
        if dest.exists():
            dest.unlink(missing_ok=True)
        raise
    return True


def _run(cmd: list[str]) -> None:
    print(">", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def _ensure_aws_cli() -> None:
    try:
        subprocess.check_call(
            ["aws", "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("Installing awscli...", flush=True)
        _run([os.sys.executable, "-m", "pip", "install", "-q", "awscli"])


def _count_dat(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(1 for _ in root.rglob("*.dat"))


def copy_tree_to_drive(src: Path, dest: Path) -> dict:
    """Copy staged local data onto Drive (incremental; skips identical sizes)."""
    src = Path(src)
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    # Prefer rsync when available (Colab/Linux); fallback to copy2.
    rsync = shutil.which("rsync")
    if rsync:
        _run([rsync, "-a", "--info=progress2", f"{src}/", f"{dest}/"])
    else:
        for path in src.rglob("*"):
            rel = path.relative_to(src)
            out = dest / rel
            if path.is_dir():
                out.mkdir(parents=True, exist_ok=True)
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            if out.exists() and out.stat().st_size == path.stat().st_size:
                continue
            shutil.copy2(path, out)

    return {
        "status": "copied",
        "src": str(src),
        "dest": str(dest),
        "dat_files": _count_dat(dest),
    }


def download_mitbih(
    raw_dir: str | Path,
    *,
    excluded_records: Iterable[int],
    dry_run: bool = False,
) -> dict:
    """Download MIT-BIH Arrhythmia Database records (incremental, wfdb)."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    excluded = set(excluded_records)

    if dry_run:
        return {"status": "skipped", "reason": "dry_run", "path": str(raw_dir)}

    wfdb = _require_wfdb()
    records = wfdb.get_record_list(MITBIH_DB_NAME)
    records_to_dl = [r for r in records if int(r) not in excluded]
    existing = {f.split(".")[0] for f in os.listdir(raw_dir) if f.endswith(".dat")}
    missing = [r for r in records_to_dl if r not in existing]

    if not missing:
        return {"status": "present", "path": str(raw_dir), "records": len(records_to_dl)}

    wfdb.dl_database(MITBIH_DB_NAME, str(raw_dir), records=missing)
    return {
        "status": "downloaded",
        "path": str(raw_dir),
        "records_downloaded": len(missing),
        "records_total": len(records_to_dl),
    }


def download_ptbxl_metadata(raw_dir: str | Path, *, dry_run: bool = False) -> dict:
    """Download PTB-XL metadata CSV files from PhysioNet."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        return {"status": "skipped", "reason": "dry_run", "path": str(raw_dir)}

    downloaded = []
    for name in PTBXL_METADATA_FILES:
        dest = raw_dir / name
        if _download_url(f"{PTBXL_PHYSIONET_BASE}{name}", dest):
            downloaded.append(name)

    return {
        "status": "downloaded" if downloaded else "present",
        "path": str(raw_dir),
        "files_downloaded": downloaded,
    }


def ptbxl_record_paths_for_folds(raw_dir: str | Path, folds: Iterable[int]) -> list[str]:
    """Return 500 Hz record paths (filename_hr) for the requested strat folds."""
    csv_path = Path(raw_dir) / "ptbxl_database.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing {csv_path}. Run download_ptbxl_metadata() first."
        )
    df = pd.read_csv(csv_path, usecols=["strat_fold", "filename_hr"])
    fold_set = set(folds)
    records = (
        df.loc[df["strat_fold"].isin(fold_set), "filename_hr"].dropna().unique().tolist()
    )
    return sorted(records)


def _ptbxl_record_present(raw_dir: Path, record_path: str) -> bool:
    base = raw_dir / record_path
    return base.with_suffix(".hea").exists() and base.with_suffix(".dat").exists()


def download_ptbxl_records(
    raw_dir: str | Path,
    folds: Iterable[int],
    *,
    dry_run: bool = False,
    batch_size: int = 100,
) -> dict:
    """Download PTB-XL 500 Hz ECG records for selected folds (incremental, batched)."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    fold_list = sorted(set(folds))

    if dry_run:
        return {
            "status": "skipped",
            "reason": "dry_run",
            "path": str(raw_dir),
            "folds": fold_list,
        }

    wfdb = _require_wfdb()
    records = ptbxl_record_paths_for_folds(raw_dir, fold_list)
    missing = [r for r in records if not _ptbxl_record_present(raw_dir, r)]

    if not missing:
        return {
            "status": "present",
            "path": str(raw_dir),
            "folds": fold_list,
            "records": len(records),
        }

    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = None

    batches = [missing[i : i + batch_size] for i in range(0, len(missing), batch_size)]
    iterator = tqdm(batches, desc="PTB-XL records") if tqdm else batches
    for batch in iterator:
        wfdb.dl_database(PTBXL_DB_NAME, str(raw_dir), records=batch, keep_subdirs=True)

    return {
        "status": "downloaded",
        "path": str(raw_dir),
        "folds": fold_list,
        "records_downloaded": len(missing),
        "records_total": len(records),
    }


def download_ptbxl_s3(
    stage_dir: str | Path,
    *,
    dry_run: bool = False,
    include_records100: bool = False,
) -> dict:
    """Bulk-download PTB-XL via PhysioNet's public S3 bucket into a local stage dir."""
    stage_dir = Path(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        return {"status": "skipped", "reason": "dry_run", "path": str(stage_dir)}

    _ensure_aws_cli()
    for name in PTBXL_METADATA_FILES:
        dest = stage_dir / name
        if not dest.exists() or dest.stat().st_size == 0:
            _run(
                [
                    "aws",
                    "s3",
                    "cp",
                    "--no-sign-request",
                    f"{PTBXL_S3_PREFIX}/{name}",
                    str(dest),
                ]
            )

    records_dest = stage_dir / "records500"
    records_dest.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "aws",
            "s3",
            "sync",
            "--no-sign-request",
            f"{PTBXL_S3_PREFIX}/records500",
            str(records_dest),
        ]
    )

    extra = {}
    if include_records100:
        r100 = stage_dir / "records100"
        r100.mkdir(parents=True, exist_ok=True)
        _run(
            [
                "aws",
                "s3",
                "sync",
                "--no-sign-request",
                f"{PTBXL_S3_PREFIX}/records100",
                str(r100),
            ]
        )
        extra["records100_dat"] = _count_dat(r100)

    return {
        "status": "downloaded",
        "method": "s3",
        "path": str(stage_dir),
        "records500_dat": _count_dat(records_dest),
        **extra,
    }


def download_ptbxl(
    raw_dir: str | Path,
    folds: Iterable[int],
    *,
    dry_run: bool = False,
    include_test_fold: bool = False,
    batch_size: int = 100,
    method: str = "auto",
    stage_dir: str | Path | None = None,
    dump_to_drive: bool = True,
) -> dict:
    """Download PTB-XL.

    method:
      - auto: use S3->local stage on Colab, else wfdb fold download
      - s3:   always PhysioNet S3 bulk sync (then optional copy to raw_dir)
      - wfdb: per-record fold download into raw_dir
    """
    raw_dir = Path(raw_dir)
    method = (method or "auto").lower()
    in_colab = Path("/content").exists()

    if method == "auto":
        method = "s3" if in_colab else "wfdb"

    if method == "s3":
        stage = Path(stage_dir) if stage_dir else (
            Path(os.environ["XFAD_PTBXL_STAGE"])
            if os.environ.get("XFAD_PTBXL_STAGE")
            else (DEFAULT_COLAB_STAGE if in_colab else raw_dir)
        )
        staged = download_ptbxl_s3(stage, dry_run=dry_run)
        result = {"stage": staged, "method": "s3"}
        if dry_run:
            return result
        if dump_to_drive and Path(stage).resolve() != raw_dir.resolve():
            result["drive"] = copy_tree_to_drive(stage, raw_dir)
        else:
            result["drive"] = {"status": "skipped", "path": str(raw_dir)}
        return result

    # wfdb path (fold-selective)
    meta = download_ptbxl_metadata(raw_dir, dry_run=dry_run)
    if dry_run:
        return {"metadata": meta, "records": {"status": "skipped", "reason": "dry_run"}}

    fold_list = list(folds)
    if include_test_fold:
        from src.config import PTBXL_TEST_FOLD

        if PTBXL_TEST_FOLD not in fold_list:
            fold_list.append(PTBXL_TEST_FOLD)

    records = download_ptbxl_records(
        raw_dir, fold_list, dry_run=False, batch_size=batch_size
    )
    return {"metadata": meta, "records": records, "method": "wfdb"}


def download_all_datasets(
    *,
    mitbih_dir: str | Path,
    ptbxl_dir: str | Path,
    mitbih_excluded: Iterable[int],
    ptbxl_folds: Iterable[int],
    dry_run: bool = False,
    include_ptbxl_test_fold: bool = False,
    ptbxl_method: str = "auto",
    ptbxl_stage_dir: str | Path | None = None,
    dump_ptbxl_to_drive: bool = True,
) -> dict:
    """Download MIT-BIH and PTB-XL into configured raw directories."""
    mitbih = download_mitbih(
        mitbih_dir, excluded_records=mitbih_excluded, dry_run=dry_run
    )
    ptbxl = download_ptbxl(
        ptbxl_dir,
        ptbxl_folds,
        dry_run=dry_run,
        include_test_fold=include_ptbxl_test_fold,
        method=ptbxl_method,
        stage_dir=ptbxl_stage_dir,
        dump_to_drive=dump_ptbxl_to_drive,
    )
    return {"mitbih": mitbih, "ptbxl": ptbxl}
