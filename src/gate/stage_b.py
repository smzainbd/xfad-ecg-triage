"""Stage B: teacher escalation with fail-safe TRIGGER when offline."""

from __future__ import annotations

import numpy as np
import torch


def iou(saliency_a: torch.Tensor | np.ndarray, saliency_b: torch.Tensor | np.ndarray, threshold: float = 0.5) -> float:
    if isinstance(saliency_a, torch.Tensor):
        saliency_a = saliency_a.detach().cpu().numpy()
    if isinstance(saliency_b, torch.Tensor):
        saliency_b = saliency_b.detach().cpu().numpy()
    a = np.asarray(saliency_a).flatten() >= threshold
    b = np.asarray(saliency_b).flatten() >= threshold
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(inter / union)


def stage_b_decision(
    student_pred: int,
    teacher_pred: int,
    student_saliency: torch.Tensor,
    teacher_saliency: torch.Tensor,
    normal_class_idx: int,
    *,
    connectivity_available: bool = True,
    divergence_threshold: float = 0.5,
) -> str:
    if not connectivity_available:
        return "TRIGGER"

    divergence = 1.0 - iou(student_saliency, teacher_saliency, threshold=0.5)

    if teacher_pred == student_pred:
        return "TRIGGER"
    if teacher_pred == normal_class_idx:
        if divergence < divergence_threshold:
            return "SUPPRESS"
        return "SUPPRESS_AND_LOG_FOR_REVIEW"
    return "TRIGGER_AND_LOG_CLASS_MISMATCH"
