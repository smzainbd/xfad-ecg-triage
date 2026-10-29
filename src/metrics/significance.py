"""Statistical tests: McNemar, DeLong (AUROC), Holm-Bonferroni."""

from __future__ import annotations

import numpy as np
import scipy.stats as stats


def mcnemar_test(y_true: np.ndarray, y_pred1: np.ndarray, y_pred2: np.ndarray) -> float:
    correct1 = y_true == y_pred1
    correct2 = y_true == y_pred2
    b = int(np.sum(~correct1 & correct2))
    c = int(np.sum(correct1 & ~correct2))
    if b + c == 0:
        return 1.0
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    return float(1 - stats.chi2.cdf(stat, 1))


def _midrank(x: np.ndarray) -> np.ndarray:
    """Mid-ranks with ties averaged (DeLong et al., 1988)."""
    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x)
    ranks = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, dtype=float)
    out[order] = ranks
    return out


def _fast_delong(scores_pos_first: np.ndarray, n_pos: int) -> tuple[np.ndarray, np.ndarray]:
    """Fast DeLong AUROC estimates and covariance (Sun & Xu, 2014).

    ``scores_pos_first`` is (n_models, n_samples) with all positive-label columns first.
    """
    m = n_pos
    n = scores_pos_first.shape[1] - m
    k = scores_pos_first.shape[0]
    positives = scores_pos_first[:, :m]
    negatives = scores_pos_first[:, m:]

    tx = np.empty([k, m])
    ty = np.empty([k, n])
    tz = np.empty([k, m + n])
    for r in range(k):
        tx[r, :] = _midrank(positives[r, :])
        ty[r, :] = _midrank(negatives[r, :])
        tz[r, :] = _midrank(scores_pos_first[r, :])

    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.atleast_2d(np.cov(v01))
    sy = np.atleast_2d(np.cov(v10))
    cov = sx / m + sy / n
    return aucs, cov


def delong_roc_test(y_true: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray) -> dict:
    """Two-sided DeLong test for two correlated AUROCs on the same samples."""
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        raise ValueError("DeLong test needs both positive and negative labels")

    order = np.argsort(-y_true, kind="stable")  # positives first
    n_pos = int((y_true == 1).sum())
    n_neg = len(y_true) - n_pos
    stacked = np.vstack([np.asarray(scores_a, dtype=float), np.asarray(scores_b, dtype=float)])[:, order]
    aucs, cov = _fast_delong(stacked, n_pos)

    # The variance estimate needs at least two samples per class.
    var_delta = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    delta = float(aucs[0] - aucs[1])
    if n_pos < 2 or n_neg < 2 or not np.isfinite(var_delta):
        return {
            "auc_a": float(aucs[0]),
            "auc_b": float(aucs[1]),
            "delta_auc": delta,
            "std_error": None,
            "z": None,
            "p_value": None,
            "note": f"variance undefined with n_pos={n_pos}, n_neg={n_neg}",
        }
    if var_delta <= 0:
        z, p = 0.0, 1.0
    else:
        z = delta / np.sqrt(var_delta)
        p = float(2 * stats.norm.sf(abs(z)))
    return {
        "auc_a": float(aucs[0]),
        "auc_b": float(aucs[1]),
        "delta_auc": delta,
        "std_error": float(np.sqrt(var_delta)) if var_delta > 0 else 0.0,
        "z": float(z),
        "p_value": p,
    }


def delong_test(y_true: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray) -> tuple[float, float]:
    """Backwards-compatible wrapper returning (delta_auroc, p_value)."""
    result = delong_roc_test(y_true, scores_a, scores_b)
    return result["delta_auc"], result["p_value"]


def holm_bonferroni(p_values: list[tuple[str, float]], alpha: float = 0.05) -> dict[str, dict]:
    sorted_p = sorted(p_values, key=lambda x: x[1])
    m = len(sorted_p)
    out = {}
    for k, (name, p) in enumerate(sorted_p):
        adj = alpha / (m - k)
        out[name] = {"p_value": p, "alpha_adj": adj, "significant": p < adj}
    return out
