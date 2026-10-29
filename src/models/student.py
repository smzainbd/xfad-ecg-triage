"""Multi-task student with per-dataset stems and shared trunk."""

from __future__ import annotations

import torch
import torch.nn as nn


class MultiTaskStudent(nn.Module):
    LAST_CONV = "trunk.9"  # last Conv1d in trunk Sequential

    def __init__(self, c_internal: int = 32, num_classes_a: int = 5, num_classes_b: int = 12):
        super().__init__()
        self.stem_a = nn.Conv1d(2, c_internal, kernel_size=1)
        self.stem_b = nn.Conv1d(12, c_internal, kernel_size=1)
        self.trunk = nn.Sequential(
            nn.Conv1d(c_internal, c_internal, 3, padding=1),
            nn.BatchNorm1d(c_internal),
            nn.ReLU(),
            nn.Conv1d(c_internal, c_internal, 3, padding=1),
            nn.BatchNorm1d(c_internal),
            nn.ReLU(),
            nn.Conv1d(c_internal, c_internal, 3, padding=1),
            nn.BatchNorm1d(c_internal),
            nn.ReLU(),
            nn.Conv1d(c_internal, c_internal, 3, padding=1),
            nn.BatchNorm1d(c_internal),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head_a = nn.Linear(c_internal, num_classes_a)
        self.head_b = nn.Linear(c_internal, num_classes_b)

    def forward(self, x: torch.Tensor, task: str, return_features: bool = False):
        if task == "A":
            x = self.stem_a(x)
            head = self.head_a
        elif task == "B":
            x = self.stem_b(x)
            head = self.head_b
        else:
            raise ValueError("task must be 'A' or 'B'")
        feat = self.trunk(x)
        pooled = self.pool(feat).squeeze(-1)
        logits = head(pooled)
        if return_features:
            return logits, feat
        return logits
