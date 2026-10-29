"""KD loss unit test."""

import torch
import torch.nn as nn

from src.models.kd_loss import kd_loss
from src.models.student import MultiTaskStudent
from src.models.teacher_a import TeacherA


def test_kd_loss_finite_positive():
    ta = TeacherA().eval()
    student = MultiTaskStudent()
    adapter = nn.Conv1d(32, 128, 1)
    x = torch.randn(2, 2, 250)
    y = torch.tensor([0, 1])
    with torch.no_grad():
        t_logits, t_feat = ta(x, return_features=True)
    s_logits, s_feat = student(x, "A", return_features=True)
    loss = kd_loss(s_logits, s_feat, t_logits, t_feat, y, adapter)
    assert torch.isfinite(loss)
    assert loss.item() > 0
