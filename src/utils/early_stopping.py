"""Validation-loss early stopping with best-checkpoint restore."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch.nn as nn


class EarlyStopping:
    """Stop when validation loss fails to improve for ``patience`` epochs."""

    def __init__(self, patience: int = 5, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.wait = 0
        self.best_states: list[dict] | None = None
        self.stopped_at: int | None = None

    def step(self, metric: float, *modules: nn.Module) -> bool:
        """Update tracker; return True when training should stop."""
        if metric < self.best - self.min_delta:
            self.best = metric
            self.wait = 0
            self.best_states = [
                {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
                for m in modules
            ]
            return False
        self.wait += 1
        return self.wait >= self.patience

    def restore(self, *modules: nn.Module) -> None:
        if not self.best_states:
            return
        for module, state in zip(modules, self.best_states):
            module.load_state_dict(state)
