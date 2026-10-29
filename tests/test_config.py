"""Tests for unified config consistency."""

from src.config import (
    COMMON_FS,
    MITBIH_DS1,
    MITBIH_DS1_TRAIN,
    MITBIH_DS1_VAL,
    MITBIH_DS2,
    MITBIH_DS2_TEST,
    MITBIH_EXCLUDED,
    PTBXL_NORMAL_CLASS,
    PTBXL_RHYTHM_CLASSES,
    PTBXL_TEST_FOLD,
    TARGET_FS,
)


def test_common_fs_matches_target():
    assert COMMON_FS == TARGET_FS == 250


def test_mitbih_splits_disjoint():
    assert not set(MITBIH_DS1_TRAIN) & set(MITBIH_DS1_VAL)
    assert not set(MITBIH_DS1) & set(MITBIH_DS2)
    assert set(MITBIH_DS1_TRAIN) | set(MITBIH_DS1_VAL) == set(MITBIH_DS1)


def test_mitbih_ds2_matches_plan():
    assert set(MITBIH_DS2) == set(MITBIH_DS2_TEST)


def test_excluded_not_in_splits():
    all_ids = set(MITBIH_DS1) | set(MITBIH_DS2)
    assert not set(MITBIH_EXCLUDED) & all_ids


def test_ptbxl_normal_is_sr():
    assert PTBXL_NORMAL_CLASS == "SR"
    assert "SR" in PTBXL_RHYTHM_CLASSES
    assert PTBXL_TEST_FOLD == 10
