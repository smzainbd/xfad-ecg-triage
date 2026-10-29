"""Gate calibration on validation alarms only."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.config import (
    GATE_ESCALATION_BUDGET,
    GATE_MIN_FALSE_ALARMS_FOR_LR,
    GATE_WEIGHTS_BEAT,
    GATE_WEIGHTS_RECORD,
    SENSITIVITY_LOSS_BUDGET,
)
from src.gate.stage_a import decide_stage_a, stage_a_features, stage_a_score


@dataclass
class BaselineThreshold:
  """Operating point for a single-feature baseline (SQI or confidence)."""

  feature: str
  threshold: float
  calibration_fa_reduction: float = 0.0
  calibration_sensitivity_loss: float = 0.0


def baseline_feature_values(
    validation_alarms: list[tuple[float, float, float, bool]],
    feature: str,
) -> np.ndarray:
    """Extract SQI or confidence from validation alarm tuples."""
    index = {"qrs_overlap": 0, "sqi": 1, "confidence": 2}[feature]
    return np.array([row[index] for row in validation_alarms], dtype=float)


def calibrate_baseline_threshold(
    validation_alarms: list[tuple[float, float, float, bool]],
    *,
    feature: str,
    budget: float = SENSITIVITY_LOSS_BUDGET,
) -> BaselineThreshold:
    """Pick a deliver threshold for one feature that maximises FA reduction within budget.

    An alarm is *delivered* when ``score >= threshold``. Higher SQI or confidence means
    keep the alarm; lower values are suppressed as likely false alarms.
    """
    scores = baseline_feature_values(validation_alarms, feature)
    is_fa = np.array([bool(fa) for *_, fa in validation_alarms])
    n_fa = int(is_fa.sum())
    n_ta = int((~is_fa).sum())

    grid = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, 201)))
    best_threshold = float(grid[0])
    best_fa_red, best_sens_loss = 0.0, 0.0

    for threshold in grid:
        delivered = scores >= threshold
        sens_loss = float((~delivered & ~is_fa).sum()) / n_ta if n_ta else 0.0
        if sens_loss > budget:
            continue
        fa_red = float((~delivered & is_fa).sum()) / n_fa if n_fa else 0.0
        if fa_red > best_fa_red or (fa_red == best_fa_red and sens_loss < best_sens_loss):
            best_threshold = float(threshold)
            best_fa_red = fa_red
            best_sens_loss = sens_loss

    return BaselineThreshold(
        feature=feature,
        threshold=best_threshold,
        calibration_fa_reduction=best_fa_red,
        calibration_sensitivity_loss=best_sens_loss,
    )


def calibrate_baseline_thresholds(
    validation_alarms: list[tuple[float, float, float, bool]],
    *,
    budget: float = SENSITIVITY_LOSS_BUDGET,
) -> dict[str, BaselineThreshold]:
    """Budget-matched SQI-only and confidence-only thresholds on validation alarms."""
    return {
        "sqi_only": calibrate_baseline_threshold(validation_alarms, feature="sqi", budget=budget),
        "confidence_only": calibrate_baseline_threshold(
            validation_alarms, feature="confidence", budget=budget,
        ),
    }


@dataclass
class GateThresholds:
    low: float
    high: float
    divergence: float = 0.5
    used_fallback: bool = False
    escalation_rate: float = 0.0
    calibration_sensitivity_loss: float = 0.0
    calibration_fa_reduction: float = 0.0


def fit_gate_model(validation_alarms: list[tuple[float, float, float, bool]], *, min_false_alarms: int = GATE_MIN_FALSE_ALARMS_FOR_LR):
    """Fit logistic regression or return None if too few false-alarm examples."""
    y = np.array([int(fa) for _, _, _, fa in validation_alarms])
    if y.sum() < min_false_alarms:
        return None
    X = np.array([stage_a_features(qo, sqi, pc)[0] for qo, sqi, pc, _ in validation_alarms])
    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(X, y)
    return model


def fallback_risk(qrs_overlap: float, sqi: float, p_confidence: float, *, record_level: bool = False) -> float:
    w1, w2, w3 = GATE_WEIGHTS_RECORD if record_level else GATE_WEIGHTS_BEAT
    return w1 * (1 - qrs_overlap) + w2 * (1 - sqi) + w3 * (1 - p_confidence)


def simulate_gate(
    validation_alarms: list[tuple[float, float, float, bool]],
    low: float,
    high: float,
    gate_model,
    *,
    record_level: bool = False,
) -> tuple[float, float]:
    """Return (sensitivity_loss, false_alarm_reduction) on validation alarms."""
    tp = fp = fn = tn = 0
    baseline_fp = sum(1 for *_, fa in validation_alarms if fa)
    gated_fp = 0
    for qo, sqi, pc, is_fa in validation_alarms:
        if gate_model is not None:
            risk = stage_a_score(gate_model, qo, sqi, pc)
        else:
            risk = fallback_risk(qo, sqi, pc, record_level=record_level)
        decision = decide_stage_a(risk, low, high)
        alarm_triggered = decision != "SUPPRESS"
        if is_fa:
            if alarm_triggered:
                gated_fp += 1
            if not alarm_triggered:
                tn += 1
            else:
                fp += 1
        else:
            if alarm_triggered:
                tp += 1
            else:
                fn += 1
    sens = tp / (tp + fn + 1e-8)
    baseline_sens = 1.0  # no gate => all alarms trigger
    sens_loss = max(0.0, baseline_sens - sens)
    fa_reduction = 1.0 - (gated_fp / (baseline_fp + 1e-8))
    return sens_loss, fa_reduction


def risk_scores(
    validation_alarms: list[tuple[float, float, float, bool]],
    gate_model,
    *,
    record_level: bool = False,
) -> np.ndarray:
    """Stage-A risk for every validation alarm, using the LR model when available."""
    if gate_model is not None:
        X = np.vstack([stage_a_features(qo, sqi, pc) for qo, sqi, pc, _ in validation_alarms])
        return gate_model.predict_proba(X)[:, 1]
    return np.array([
        fallback_risk(qo, sqi, pc, record_level=record_level)
        for qo, sqi, pc, _ in validation_alarms
    ])


def _choose_low(risks: np.ndarray, high: float, escalation_budget: float) -> float:
    """Smallest threshold whose escalation band [low, high] stays inside the cost budget."""
    for low in np.unique(np.quantile(risks, np.linspace(0.0, 1.0, 101))):
        if low > high:
            break
        band = float(np.mean((risks >= low) & (risks <= high)))
        if band <= escalation_budget:
            return float(low)
    return float(high)


def calibrate_thresholds(
    validation_alarms: list[tuple[float, float, float, bool]],
    gate_model,
    *,
    record_level: bool = False,
    budget: float = SENSITIVITY_LOSS_BUDGET,
    escalation_budget: float = GATE_ESCALATION_BUDGET,
    divergence: float = 0.5,
) -> GateThresholds:
    """Pick the Stage-A thresholds that maximise FA reduction within the sensitivity budget.

    Candidates are quantiles of the validation risk distribution, so the search adapts to
    both the logistic-regression probability scale and the fallback rule's score scale.
    Only ``high`` moves the sensitivity/FA trade-off (alarms with risk > high are
    suppressed); ``low`` sets how many borderline alarms reach Stage B.
    """
    risks = risk_scores(validation_alarms, gate_model, record_level=record_level)
    is_fa = np.array([bool(fa) for *_, fa in validation_alarms])
    n_fa = int(is_fa.sum())
    n_ta = int((~is_fa).sum())

    grid = np.unique(np.concatenate([
        np.quantile(risks, np.linspace(0.0, 1.0, 201)),
        [float(risks.max()) + 1e-6],
    ]))

    best = (float(grid[-1]), 0.0, 0.0)  # (high, fa_reduction, sensitivity_loss)
    for high in grid:
        suppressed = risks > high
        sens_loss = float((suppressed & ~is_fa).sum()) / n_ta if n_ta else 0.0
        if sens_loss > budget:
            continue
        fa_red = float((suppressed & is_fa).sum()) / n_fa if n_fa else 0.0
        if fa_red > best[1] or (fa_red == best[1] and sens_loss < best[2]):
            best = (float(high), fa_red, sens_loss)

    high, fa_red, sens_loss = best
    low = _choose_low(risks, high, escalation_budget)
    return GateThresholds(
        low=low,
        high=high,
        divergence=divergence,
        used_fallback=fa_red <= 0.0,
        escalation_rate=float(np.mean((risks >= low) & (risks <= high))),
        calibration_sensitivity_loss=sens_loss,
        calibration_fa_reduction=fa_red,
    )
