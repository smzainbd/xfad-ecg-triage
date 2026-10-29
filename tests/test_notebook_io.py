"""Tests for notebook run-mode helpers."""

import os

import pytest

from src.utils.io import student_variants_for_run


@pytest.fixture(autouse=True)
def _clear_variant_override():
    old = os.environ.pop("XFAD_STUDENT_VARIANT", None)
    yield
    if old is None:
        os.environ.pop("XFAD_STUDENT_VARIANT", None)
    else:
        os.environ["XFAD_STUDENT_VARIANT"] = old


def test_student_variants_smoke_is_baseline_only():
    assert student_variants_for_run(dry_run=True) == ("baseline",)


def test_student_variants_full_run_includes_saliency():
    assert student_variants_for_run(dry_run=False) == ("baseline", "saliency")


def test_student_variants_env_override():
    os.environ["XFAD_STUDENT_VARIANT"] = "saliency"
    assert student_variants_for_run(dry_run=True) == ("saliency",)
