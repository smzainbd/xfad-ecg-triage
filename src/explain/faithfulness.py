"""Faithfulness metrics: deletion/insertion and pointing-game analogue."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def deletion_auc(model, x: torch.Tensor, target_class: int, cam: torch.Tensor, steps: int = 20, task: str | None = None) -> float:
    seq_len = x.shape[-1]
    order = torch.argsort(cam.flatten(), descending=True)
    confidences = []
    with torch.no_grad():
        for i in range(steps + 1):
            frac = i / steps
            n_mask = int(frac * seq_len)
            x_m = x.clone()
            if n_mask > 0:
                x_m[..., order[:n_mask]] = 0.0
            if task:
                logits = model(x_m, task)
            else:
                logits = model(x_m)
            if isinstance(logits, tuple):
                logits = logits[0]
            conf = F.softmax(logits, dim=1)[0, target_class].item()
            confidences.append(conf)
    return float(sum(confidences) / len(confidences))


def pointing_game_ratios(cam: torch.Tensor, qrs_onset: int, qrs_offset: int, fs: int) -> dict[str, float]:
    cam = cam.flatten()
    p_start = max(0, qrs_onset - int(200 * fs / 1000))
    p_end = max(0, qrs_onset - int(50 * fs / 1000))
    t_start = min(len(cam), qrs_offset + int(50 * fs / 1000))
    t_end = min(len(cam), qrs_offset + int(350 * fs / 1000))
    total = cam.sum().item() + 1e-8
    return {
        "p_ratio": cam[p_start:p_end].sum().item() / total,
        "qrs_ratio": cam[qrs_onset:qrs_offset].sum().item() / total,
        "t_ratio": cam[t_start:t_end].sum().item() / total,
    }
