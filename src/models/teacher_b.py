"""Teacher-B: 1D-ResNet for PTB-XL 12-class rhythm classification."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 5, stride=stride, padding=2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 5, padding=2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class TeacherB(nn.Module):
    LAST_CONV = "layer3.conv2"

    def __init__(self, in_channels: int = 12, num_classes: int = 12):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 32, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=2, padding=1),
        )
        self.layer1 = ResBlock1d(32, 64, stride=2)
        self.layer2 = ResBlock1d(64, 128, stride=2)
        self.layer3 = ResBlock1d(128, 256, stride=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        feat = self.layer3(x)
        pooled = self.pool(feat).squeeze(-1)
        logits = self.fc(pooled)
        if return_features:
            return logits, feat
        return logits
