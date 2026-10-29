"""Parameter- and data-randomisation sanity checks (Adebayo et al.)."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


def _map_on_cpu(tensor: torch.Tensor) -> torch.Tensor:
    """Detach a saliency map and move it to CPU for scalar metric computation."""
    return tensor.detach().reshape(-1).to(dtype=torch.float32, device="cpu")


def cosine_similarity_map(a: torch.Tensor, b: torch.Tensor) -> float:
    a = _map_on_cpu(a)
    b = _map_on_cpu(b)
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def _ranks(values: torch.Tensor) -> torch.Tensor:
    """Ordinal ranks (ties broken by position, adequate for continuous CAMs)."""
    v = _map_on_cpu(values)
    order = torch.argsort(v)
    ranks = torch.empty_like(v)
    ranks[order] = torch.arange(v.numel(), dtype=v.dtype)
    return ranks


def rank_correlation_map(a: torch.Tensor, b: torch.Tensor) -> float:
    """Spearman rank correlation between two saliency maps.

    Grad-CAM maps are non-negative, which inflates cosine similarity even for
    unrelated maps, so rank correlation is the primary sanity-check statistic
    (as in Adebayo et al., 2018).
    """
    a = _ranks(a)
    b = _ranks(b)
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    if denom == 0:
        return 0.0
    return float((a @ b) / denom)


def _randomized_model_copy(model: torch.nn.Module, seed: int | None = None) -> torch.nn.Module:
    """Deep-copy and fully randomize weights; reset BN running stats."""
    if seed is not None:
        torch.manual_seed(seed)
    randomized = copy.deepcopy(model)
    for module in randomized.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.reset_running_stats()
    for param in randomized.parameters():
        torch.nn.init.normal_(param)
    # Use batch stats from the current input, not stale trained running means.
    randomized.train()
    return randomized


def parameter_randomization_report(
    model: torch.nn.Module,
    x: torch.Tensor,
    target_class: int,
    layer_name: str,
    *,
    task: str | None = None,
    grad_cam_fn=None,
    seed: int | None = None,
) -> dict[str, float]:
    """Similarity between the trained and weight-randomized Grad-CAM for one input."""
    from src.explain.gradcam import grad_cam_1d

    grad_cam_fn = grad_cam_fn or grad_cam_1d
    trained = copy.deepcopy(model).eval()
    base = grad_cam_fn(trained, x, target_class, layer_name, task=task)

    randomized = _randomized_model_copy(model, seed=seed)
    rand = grad_cam_fn(randomized, x, target_class, layer_name, task=task)
    return {
        "cosine": cosine_similarity_map(base, rand),
        "spearman": rank_correlation_map(base, rand),
    }


def parameter_randomization_similarity(
    model: torch.nn.Module,
    x: torch.Tensor,
    target_class: int,
    layer_name: str,
    *,
    task: str | None = None,
    grad_cam_fn=None,
    seed: int | None = None,
) -> float:
    """Cosine-similarity-only view of :func:`parameter_randomization_report`."""
    return parameter_randomization_report(
        model, x, target_class, layer_name, task=task, grad_cam_fn=grad_cam_fn, seed=seed,
    )["cosine"]


# (fail_above_mean, fail_above_median, pass_below_mean, pass_below_median, low_cut)
SANITY_THRESHOLDS = {
    "cosine": (0.70, 0.75, 0.30, 0.45, 0.50),
    "spearman": (0.50, 0.55, 0.20, 0.30, 0.30),
}


def evaluate_sanity_checks(similarities: list[float], *, metric: str = "cosine") -> str:
    """Return 'pass', 'fail', or 'inconclusive' for a set of randomization similarities.

    Cosine similarity between two non-negative maps is bounded below by zero and runs
    high by construction, so its thresholds are looser than the rank-correlation ones.
    """
    if not similarities:
        return "inconclusive"
    fail_mean, fail_median, pass_mean, pass_median, low_cut = SANITY_THRESHOLDS[metric]
    mean_sim = sum(similarities) / len(similarities)
    median_sim = sorted(similarities)[len(similarities) // 2]
    low_frac = sum(s < low_cut for s in similarities) / len(similarities)

    if mean_sim > fail_mean or median_sim > fail_median:
        return "fail"
    if mean_sim < pass_mean or (median_sim < pass_median and low_frac >= 0.6):
        return "pass"
    return "inconclusive"
