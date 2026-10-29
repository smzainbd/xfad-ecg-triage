"""Gate module unit tests."""

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from src.gate.calibration import calibrate_thresholds, fit_gate_model
from src.gate.stage_a import decide_stage_a, fraction_of_saliency_mass_inside, stage_a_score
from src.gate.stage_b import iou, stage_b_decision


def test_fraction_of_saliency_mass():
    cam = torch.zeros(100)
    cam[10:20] = 1.0
    assert fraction_of_saliency_mass_inside(cam, [(10, 20)]) == 1.0
    assert fraction_of_saliency_mass_inside(cam, [(0, 5)]) == 0.0


def test_stage_a_decisions():
    assert decide_stage_a(0.1, 0.3, 0.7) == "TRIGGER"
    assert decide_stage_a(0.9, 0.3, 0.7) == "SUPPRESS"
    assert decide_stage_a(0.5, 0.3, 0.7) == "ESCALATE"


def test_stage_b_offline_failsafe():
    cam = torch.ones(50)
    assert stage_b_decision(1, 0, cam, cam, 0, connectivity_available=False) == "TRIGGER"


def test_iou_identical_maps():
    cam = torch.zeros(10)
    cam[2:5] = 1.0
    assert iou(cam, cam) == 1.0


def test_calibration_runs():
    alarms = [(0.2, 0.3, 0.4, True), (0.8, 0.9, 0.95, False)] * 30
    model = fit_gate_model(alarms)
    th = calibrate_thresholds(alarms, model)
    assert th.low <= th.high


def test_calibration_respects_sensitivity_budget():
    """Thresholds must not sacrifice more true alarms than the budget allows."""
    rng = np.random.default_rng(0)
    alarms = []
    for _ in range(200):  # false alarms: noisy, off-QRS, low confidence
        alarms.append((float(rng.uniform(0.0, 0.5)), float(rng.uniform(0.2, 0.6)), float(rng.uniform(0.5, 0.8)), True))
    for _ in range(200):  # true alarms: clean, on-QRS, confident
        alarms.append((float(rng.uniform(0.5, 1.0)), float(rng.uniform(0.6, 1.0)), float(rng.uniform(0.8, 1.0)), False))

    th = calibrate_thresholds(alarms, fit_gate_model(alarms), budget=0.02)
    assert th.calibration_sensitivity_loss <= 0.02
    assert th.calibration_fa_reduction > 0.5
    assert not th.used_fallback


def test_calibration_without_lr_uses_fallback_scale():
    """Quantile search must also work on the fallback rule's score range."""
    rng = np.random.default_rng(1)
    alarms = [(float(rng.uniform(0.0, 0.4)), 0.3, 0.6, True) for _ in range(40)]
    alarms += [(float(rng.uniform(0.7, 1.0)), 0.9, 0.95, False) for _ in range(40)]
    th = calibrate_thresholds(alarms, None, budget=0.02)
    assert 0.0 <= th.low <= th.high <= 1.0
    assert th.calibration_fa_reduction > 0.0


def _synthetic_alarm_records():
    from src.gate.evaluation import AlarmRecord

    false_alarms = [
        AlarmRecord(index=i, y_true=0, student_pred=1, teacher_pred=0, p_conf=0.6,
                    sqi=0.2, qrs_overlap=0.1, group=f"rec{i % 5}")
        for i in range(10)
    ]
    true_alarms = [
        AlarmRecord(index=10 + i, y_true=1, student_pred=1, teacher_pred=1, p_conf=0.95,
                    sqi=0.9, qrs_overlap=0.8, group=f"rec{i % 5}")
        for i in range(10)
    ]
    return false_alarms + true_alarms


def test_gate_metrics_on_alarm_population():
    from src.gate.evaluation import compute_gate_metrics, evaluate_baselines

    records = _synthetic_alarm_records()
    assert sum(r.is_false_alarm for r in records) == 10
    assert sum(r.is_true_alarm for r in records) == 10

    delivered = np.array([r.sqi >= 0.5 for r in records])
    metrics = compute_gate_metrics(records, delivered, name="sqi", budget=0.02)
    assert metrics.false_alarm_reduction == 1.0
    assert metrics.sensitivity == 1.0
    assert metrics.meets_budget is True

    results = evaluate_baselines(records, gate_model=None, low=0.3, high=0.5, budget=0.02, n_boot=50)
    assert results["no_gate"].false_alarm_reduction == 0.0
    assert results["no_gate"].sensitivity == 1.0
    assert results["stage_a_offline"].false_alarm_reduction > 0.0
    assert results["stage_a_offline"].sensitivity_loss <= 0.02
    assert len(results["no_gate"].false_alarm_reduction_ci) == 2
    assert "stage_a_plus_b" not in results  # needs CAMs


def test_stage_b_only_runs_inside_escalation_band():
    """Stage A+B must never deliver an alarm Stage A already suppressed."""
    from src.gate.evaluation import policy_decisions

    records = _synthetic_alarm_records()
    cam_a = torch.zeros(64)
    cam_a[10:20] = 1.0
    cam_b = torch.zeros(64)
    cam_b[40:50] = 1.0
    cams = [(cam_a, cam_b)] * len(records)

    decisions = policy_decisions(records, gate_model=None, low=0.1, high=0.5, cams=cams)
    offline = decisions["stage_a_offline"]
    with_stage_b = decisions["stage_a_plus_b"]
    assert not (with_stage_b & ~offline).any()


def test_baseline_threshold_calibration_respects_budget():
    from src.gate.calibration import calibrate_baseline_thresholds

    rng = np.random.default_rng(2)
    alarms = []
    for _ in range(100):
        alarms.append((0.1, float(rng.uniform(0.0, 0.5)), float(rng.uniform(0.4, 0.7)), True))
    for _ in range(100):
        alarms.append((0.9, float(rng.uniform(0.7, 1.0)), float(rng.uniform(0.85, 0.99)), False))

    baselines = calibrate_baseline_thresholds(alarms, budget=0.02)
    for name in ("sqi_only", "confidence_only"):
        bt = baselines[name]
        assert bt.calibration_sensitivity_loss <= 0.02
        assert 0.0 <= bt.threshold <= 1.0
