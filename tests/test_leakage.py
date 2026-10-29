"""Leakage checks for MIT-BIH and PTB-XL splits."""

from __future__ import annotations

from src.config import (
    MITBIH_DS1,
    MITBIH_DS1_TRAIN,
    MITBIH_DS1_VAL,
    MITBIH_DS2,
    MITBIH_EXCLUDED,
    PTBXL_TEST_FOLD,
    PTBXL_TRAIN_FOLDS,
    PTBXL_VAL_FOLD,
)


def test_mitbih_ds1_ds2_disjoint():
    assert not set(MITBIH_DS1) & set(MITBIH_DS2)


def test_mitbih_train_val_disjoint():
    assert not set(MITBIH_DS1_TRAIN) & set(MITBIH_DS1_VAL)


def test_mitbih_excluded_not_in_splits():
    all_ids = set(MITBIH_DS1) | set(MITBIH_DS2)
    assert not set(MITBIH_EXCLUDED) & all_ids


def test_ptbxl_folds_disjoint():
    train = set(PTBXL_TRAIN_FOLDS)
    assert not (train & {PTBXL_VAL_FOLD})
    assert not (train & {PTBXL_TEST_FOLD})


def test_ptbxl_covers_official_folds():
    all_folds = set(PTBXL_TRAIN_FOLDS) | {PTBXL_VAL_FOLD, PTBXL_TEST_FOLD}
    assert all_folds == set(range(1, 11))
