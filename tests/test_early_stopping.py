"""Tests for early stopping helper."""

import torch
import torch.nn as nn

from src.utils.early_stopping import EarlyStopping


class _Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.tensor(1.0))


def test_early_stopping_saves_best_and_stops():
    model = _Tiny()
    stop = EarlyStopping(patience=2)
    assert not stop.step(1.0, model)
    assert not stop.step(0.9, model)
    assert not stop.step(0.95, model)
    assert stop.step(0.96, model)
    stop.restore(model)
    assert stop.best == 0.9
