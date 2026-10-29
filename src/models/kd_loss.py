"""Knowledge-distillation loss (hard + response + feature)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def kd_loss(
    s_logits: torch.Tensor,
    s_feat: torch.Tensor,
    t_logits: torch.Tensor,
    t_feat: torch.Tensor,
    y_true: torch.Tensor,
    adapter: nn.Module,
    *,
    T: float = 4.0,
    w_hard: float = 0.3,
    w_resp: float = 0.5,
    w_feat: float = 0.2,
    multilabel: bool = False,
    class_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    if multilabel:
        hard = F.binary_cross_entropy_with_logits(s_logits, y_true)
        resp = F.binary_cross_entropy_with_logits(
            s_logits / T, torch.sigmoid(t_logits.detach() / T)
        ) * (T ** 2)
    else:
        hard = F.cross_entropy(s_logits, y_true, weight=class_weight)
        resp = F.kl_div(
            F.log_softmax(s_logits / T, dim=1),
            F.softmax(t_logits.detach() / T, dim=1),
            reduction="batchmean",
        ) * (T ** 2)

    if s_feat.shape[-1] != t_feat.shape[-1]:
        s_feat = F.interpolate(s_feat, size=t_feat.shape[-1], mode="linear", align_corners=False)
    adapted = adapter(s_feat)
    feat = F.mse_loss(adapted, t_feat.detach())

    total = w_hard * hard + w_resp * resp + w_feat * feat
    if not torch.isfinite(total):
        raise ValueError("KD loss is not finite")
    return total
