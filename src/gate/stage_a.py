"""Stage A: on-device false-alarm risk scoring."""

from __future__ import annotations

import numpy as np
import torch


def fraction_of_saliency_mass_inside(saliency_map: torch.Tensor | np.ndarray, windows: list[tuple[int, int]]) -> float:
    if isinstance(saliency_map, torch.Tensor):
        saliency_map = saliency_map.detach().cpu().numpy()
    saliency_map = np.asarray(saliency_map).flatten()
    mass_inside = sum(saliency_map[start:end].sum() for start, end in windows)
    mass_total = saliency_map.sum()
    if mass_total == 0:
        return 0.0
    return float(mass_inside / mass_total)


def stage_a_features(qrs_overlap: float, sqi: float, p_confidence: float) -> np.ndarray:
    return np.array([[1.0 - qrs_overlap, 1.0 - sqi, 1.0 - p_confidence]])


def stage_a_score(gate_model, qrs_overlap: float, sqi: float, p_confidence: float) -> float:
    X = stage_a_features(qrs_overlap, sqi, p_confidence)
    return float(gate_model.predict_proba(X)[0, 1])


def decide_stage_a(risk: float, low_threshold: float, high_threshold: float) -> str:
    if risk < low_threshold:
        return "TRIGGER"
    if risk > high_threshold:
        return "SUPPRESS"
    return "ESCALATE"
