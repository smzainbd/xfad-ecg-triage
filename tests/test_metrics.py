"""Statistical test and cache-metadata unit tests."""

import numpy as np
from sklearn.metrics import roc_auc_score

from src.metrics.significance import delong_roc_test, holm_bonferroni, mcnemar_test


def test_delong_auc_matches_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    a = rng.normal(y * 1.2, 1.0)
    b = rng.normal(y * 0.4, 1.0)
    result = delong_roc_test(y, a, b)
    assert abs(result["auc_a"] - roc_auc_score(y, a)) < 1e-6
    assert abs(result["auc_b"] - roc_auc_score(y, b)) < 1e-6


def test_delong_detects_real_difference():
    rng = np.random.default_rng(1)
    y = np.repeat([0, 1], 500)
    strong = rng.normal(y * 2.0, 1.0)
    weak = rng.normal(y * 0.1, 1.0)
    result = delong_roc_test(y, strong, weak)
    assert result["delta_auc"] > 0
    assert result["p_value"] < 0.01


def test_delong_identical_scores_not_significant():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 200)
    scores = rng.normal(y, 1.0)
    result = delong_roc_test(y, scores, scores)
    assert result["delta_auc"] == 0.0
    assert result["p_value"] == 1.0


def test_mcnemar_and_holm():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    good = y.copy()
    bad = 1 - y
    assert mcnemar_test(y, good, bad) < 0.05
    corrected = holm_bonferroni([("a", 0.001), ("b", 0.4)])
    assert corrected["a"]["significant"]
    assert not corrected["b"]["significant"]


def test_cache_roundtrip_preserves_groups(tmp_path):
    from src.data.cache import cache_has_groups, load_samples_npz, save_samples_npz

    samples = [(np.zeros((2, 8), dtype=np.float32), i % 3) for i in range(6)]
    groups = ["100", "100", "101", "101", "102", "102"]
    path = tmp_path / "cache.npz"
    meta = save_samples_npz(path, samples, groups)
    assert meta["has_groups"]
    assert cache_has_groups(path)

    loaded, loaded_groups = load_samples_npz(path, with_groups=True)
    assert len(loaded) == 6
    assert loaded_groups == groups

    legacy = tmp_path / "legacy.npz"
    save_samples_npz(legacy, samples)
    assert not cache_has_groups(legacy)
    _, empty = load_samples_npz(legacy, with_groups=True)
    assert empty == []
