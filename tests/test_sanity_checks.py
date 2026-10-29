"""Sanity-check metric tests."""

import torch

from src.explain.gradcam import grad_cam_1d
from src.explain.sanity_checks import cosine_similarity_map, parameter_randomization_report, rank_correlation_map
from src.models.student import MultiTaskStudent


def test_rank_correlation_on_cuda_when_available():
    if not torch.cuda.is_available():
        return
    a = torch.rand(250, device="cuda")
    b = torch.rand(250, device="cuda")
    rho = rank_correlation_map(a, b)
    assert -1.0 <= rho <= 1.0


def test_cosine_similarity_on_cuda_when_available():
    if not torch.cuda.is_available():
        return
    a = torch.rand(250, device="cuda")
    b = torch.rand(250, device="cuda")
    cos = cosine_similarity_map(a, b)
    assert -1.0 <= cos <= 1.0


def test_parameter_randomization_report_on_cuda_when_available():
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    student = MultiTaskStudent().to(device).eval()
    x = torch.randn(1, 2, 250, device=device)
    report = parameter_randomization_report(
        student, x, 1, MultiTaskStudent.LAST_CONV, task="A", grad_cam_fn=grad_cam_1d, seed=42,
    )
    assert set(report) == {"cosine", "spearman"}
    assert all(-1.0 <= v <= 1.0 for v in report.values())
