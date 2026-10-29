"""Saliency-matching distillation: align the student's Grad-CAM with the teacher's.

Standard response/feature distillation constrains *what* the student predicts. This adds
a term constraining *where* it looks, which is the property the false-alarm gate depends
on (Stage A scores the fraction of saliency mass inside the QRS complex).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.explain.gradcam import grad_cam_1d_batch


def saliency_alignment_loss(student_cam: torch.Tensor, teacher_cam: torch.Tensor) -> torch.Tensor:
    """Mean squared error between two peak-normalised saliency maps."""
    return F.mse_loss(student_cam, teacher_cam.detach())


def saliency_kd_term(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    x: torch.Tensor,
    target_classes: torch.Tensor,
    *,
    student_layer: str,
    teacher_layer: str,
    task: str = "A",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (loss, student_cam, teacher_cam) for one batch.

    The teacher CAM needs gradients internally, so it is computed under ``enable_grad``
    and then detached; only the student CAM carries a graph back to the parameters.
    """
    teacher_cam = grad_cam_1d_batch(
        teacher, x, target_classes, teacher_layer, create_graph=False,
    ).detach()
    student_cam = grad_cam_1d_batch(
        student, x, target_classes, student_layer, task=task, create_graph=True,
    )
    return saliency_alignment_loss(student_cam, teacher_cam), student_cam, teacher_cam
