"""Differentiable Grad-CAM and saliency-matching distillation tests."""

import torch

from src.explain.gradcam import grad_cam_1d, grad_cam_1d_batch
from src.models.saliency_kd import saliency_alignment_loss, saliency_kd_term
from src.models.student import MultiTaskStudent
from src.models.teacher_a import TeacherA
from src.utils.io import student_checkpoint_path, variant_result_path, variant_suffix


def test_batched_cam_matches_per_sample_cam():
    """Batching must not change any individual explanation."""
    torch.manual_seed(0)
    model = MultiTaskStudent().eval()
    x = torch.randn(4, 2, 250)
    targets = torch.tensor([0, 1, 2, 3])

    batched = grad_cam_1d_batch(model, x, targets, MultiTaskStudent.LAST_CONV, task="A")
    for i in range(4):
        single = grad_cam_1d(model, x[i:i + 1], int(targets[i]), MultiTaskStudent.LAST_CONV, task="A")
        assert torch.allclose(batched[i], single[0], atol=1e-5)


def test_cam_shape_and_normalisation():
    torch.manual_seed(0)
    model = MultiTaskStudent().eval()
    x = torch.randn(3, 2, 250)
    cam = grad_cam_1d_batch(model, x, torch.tensor([1, 1, 1]), MultiTaskStudent.LAST_CONV, task="A")
    assert cam.shape == (3, 250)
    assert (cam >= 0).all()
    assert torch.allclose(cam.max(dim=-1).values, torch.ones(3), atol=1e-4)


def test_create_graph_reaches_the_classifier_head():
    """The head only influences the CAM through the gradient weights.

    Without create_graph that path is cut, so head_a never learns where to look.
    """
    torch.manual_seed(0)
    model = MultiTaskStudent()
    x = torch.randn(2, 2, 250)
    targets = torch.tensor([1, 2])
    head_weight = model.head_a.weight

    detached = grad_cam_1d_batch(model, x, targets, MultiTaskStudent.LAST_CONV, task="A")
    grad = torch.autograd.grad(detached.pow(2).mean(), head_weight, allow_unused=True)[0]
    assert grad is None

    differentiable = grad_cam_1d_batch(
        model, x, targets, MultiTaskStudent.LAST_CONV, task="A", create_graph=True,
    )
    grad = torch.autograd.grad(differentiable.pow(2).mean(), head_weight, allow_unused=True)[0]
    assert grad is not None and torch.isfinite(grad).all()


def test_saliency_alignment_loss_is_zero_for_identical_maps():
    cam = torch.rand(2, 250)
    assert saliency_alignment_loss(cam, cam).item() == 0.0
    assert saliency_alignment_loss(cam, torch.zeros_like(cam)).item() > 0.0


def test_saliency_kd_term_trains_the_student():
    """Optimising the alignment term must actually move the student's saliency.

    Eval mode keeps BatchNorm from adding batch-to-batch noise to the CAM, which at
    small batch sizes can mask the signal entirely.
    """
    torch.manual_seed(0)
    student = MultiTaskStudent().eval()
    teacher = TeacherA().eval()
    x = torch.randn(8, 2, 250)
    targets = torch.tensor([1] * 8)
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)

    losses = []
    for _ in range(10):
        loss, s_cam, t_cam = saliency_kd_term(
            student, teacher, x, targets,
            student_layer=MultiTaskStudent.LAST_CONV,
            teacher_layer=TeacherA.LAST_CONV,
        )
        assert s_cam.shape == t_cam.shape
        assert not t_cam.requires_grad
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0] * 0.9


def test_variant_paths_do_not_collide():
    assert variant_suffix("baseline") == ""
    assert variant_suffix("saliency") == "_saliency"
    assert student_checkpoint_path("baseline") != student_checkpoint_path("saliency")
    assert student_checkpoint_path("baseline").endswith("student.pt")
    assert student_checkpoint_path("saliency").endswith("student_saliency.pt")
    assert variant_result_path("gate_eval_ds2", "saliency").endswith("gate_eval_ds2_saliency.json")
