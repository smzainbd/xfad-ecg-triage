"""Teacher-A: small 1D-CNN for MIT-BIH beat classification."""

from __future__ import annotations

import torch
import torch.nn as nn


class TeacherA(nn.Module):
    LAST_CONV = "features.12"  # last Conv1d in features Sequential

    def __init__(self, in_channels: int = 2, num_classes: int = 5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        feat = self.features(x)
        pooled = self.pool(feat).squeeze(-1)
        logits = self.fc(pooled)
        if return_features:
            return logits, feat
        return logits
