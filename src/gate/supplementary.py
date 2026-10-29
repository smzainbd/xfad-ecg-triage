"""Stage-A coefficient report and paired bootstrap policy comparisons."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from src.config import (
    AAMI_CLASS_TO_IDX,
    BOOTSTRAP_N,
    COMMON_FS,
    MITBIH_DS1_VAL,
    MITBIH_DS2_TEST,
    MITBIH_NORMAL_CLASS,
    MODEL_DIR,
    RESULTS_DIR,
    SENSITIVITY_LOSS_BUDGET,
    STUDENT_TRUNK_CHANNELS,
)
from src.data.mitbih import MITBIHDataset
from src.gate.calibration import _choose_low
from src.gate.evaluation import (
    collect_alarm_records,
    evaluate_baselines,
    policy_decisions,
    records_to_tuples,
)
from src.gate.stage_a import stage_a_features
from src.models.student import MultiTaskStudent
from src.models.teacher_a import TeacherA
from src.utils.io import student_checkpoint_path, variant_result_path

FEATURE_NAMES = ("1-QRS overlap", "1-SQI", "1-confidence")

STAGE_A_ABLATION_COLUMNS = {
    "full": (0, 1, 2),
    "no_saliency": (1, 2),
    "sqi_only_lr": (1,),
    "saliency_only_lr": (0,),
}


def fit_lr_on_columns(
    alarms: list[tuple[float, float, float, bool]],
    columns: tuple[int, ...],
) -> LogisticRegression | None:
    y = np.array([int(fa) for _, _, _, fa in alarms])
    if y.sum() < 50:
        return None
    X = np.array([stage_a_features(qo, sqi, pc)[0][list(columns)] for qo, sqi, pc, _ in alarms])
    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(X, y)
    return model


def risk_scores_subset(
    alarms: list[tuple[float, float, float, bool]],
    model: LogisticRegression,
    columns: tuple[int, ...],
) -> np.ndarray:
    X = np.array([stage_a_features(qo, sqi, pc)[0][list(columns)] for qo, sqi, pc, _ in alarms])
    return model.predict_proba(X)[:, 1]


def calibrate_subset_model(
    alarms: list[tuple[float, float, float, bool]],
    model: LogisticRegression,
    columns: tuple[int, ...],
    *,
    budget: float = SENSITIVITY_LOSS_BUDGET,
) -> dict[str, float]:
    risks = risk_scores_subset(alarms, model, columns)
    is_fa = np.array([bool(fa) for *_, fa in alarms])
    n_fa = int(is_fa.sum())
    n_ta = int((~is_fa).sum())
    grid = np.unique(np.concatenate([
        np.quantile(risks, np.linspace(0.0, 1.0, 201)),
        [float(risks.max()) + 1e-6],
    ]))
    best = (float(grid[-1]), 0.0, 0.0)
    for high in grid:
        suppressed = risks > high
        sens_loss = float((suppressed & ~is_fa).sum()) / n_ta if n_ta else 0.0
        if sens_loss > budget:
            continue
        fa_red = float((suppressed & is_fa).sum()) / n_fa if n_fa else 0.0
        if fa_red > best[1] or (fa_red == best[1] and sens_loss < best[2]):
            best = (float(high), fa_red, sens_loss)
    high, fa_red, sens_loss = best
    low = _choose_low(risks, high, 0.20)
    return {
        "low": low,
        "high": high,
        "calibration_fa_reduction": fa_red,
        "calibration_sensitivity_loss": sens_loss,
    }


def coefficient_report(gate_model: LogisticRegression) -> dict[str, Any]:
    coef = gate_model.coef_[0]
    intercept = float(gate_model.intercept_[0])
    return {
        "intercept": intercept,
        "coefficients": {FEATURE_NAMES[i]: float(coef[i]) for i in range(3)},
        "odds_ratios": {FEATURE_NAMES[i]: float(np.exp(coef[i])) for i in range(3)},
    }


def paired_bootstrap_fa_difference(
    records,
    delivered_a: np.ndarray,
    delivered_b: np.ndarray,
    *,
    n_boot: int = BOOTSTRAP_N,
    seed: int = 42,
) -> dict[str, float | int | list[float]]:
    groups = np.array([r.group for r in records])
    unique_groups = np.unique(groups)
    index_by_group = {g: np.where(groups == g)[0] for g in unique_groups}
    is_fa = np.array([r.is_false_alarm for r in records])
    delivered_a = np.asarray(delivered_a, dtype=bool)
    delivered_b = np.asarray(delivered_b, dtype=bool)

    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    for _ in range(n_boot):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([index_by_group[g] for g in sampled])
        fa_mask = is_fa[idx]
        if not fa_mask.any():
            continue
        fa_a = 1.0 - delivered_a[idx][fa_mask].sum() / fa_mask.sum()
        fa_b = 1.0 - delivered_b[idx][fa_mask].sum() / fa_mask.sum()
        deltas.append(fa_a - fa_b)

    arr = np.asarray(deltas, dtype=float)
    lo, hi = np.quantile(arr, [0.025, 0.975])
    p_two_sided = float(2 * min(np.mean(arr <= 0), np.mean(arr >= 0)))
    return {
        "n_bootstrap": int(len(arr)),
        "mean_delta_pp": float(arr.mean() * 100),
        "median_delta_pp": float(np.median(arr) * 100),
        "ci95_pp": [float(lo * 100), float(hi * 100)],
        "p_two_sided": p_two_sided,
    }


def stage_a_ablations(
    val_alarms: list[tuple[float, float, float, bool]],
    *,
    budget: float = SENSITIVITY_LOSS_BUDGET,
) -> dict[str, dict]:
    ablations: dict[str, dict] = {}
    for name, cols in STAGE_A_ABLATION_COLUMNS.items():
        model = fit_lr_on_columns(val_alarms, cols)
        th = calibrate_subset_model(val_alarms, model, cols, budget=budget)
        entry = {
            "columns": [FEATURE_NAMES[i] for i in cols],
            "validation_fa_reduction_pct": round(th["calibration_fa_reduction"] * 100, 1),
            "validation_sensitivity_loss_pct": round(th["calibration_sensitivity_loss"] * 100, 2),
            "low": th["low"],
            "high": th["high"],
        }
        if model is not None:
            c = model.coef_[0]
            entry["coefficients"] = {
                entry["columns"][i]: float(c[i]) for i in range(len(cols))
            }
        ablations[name] = entry
    return ablations


def format_supplementary_summary(payload: dict[str, Any]) -> str:
    coef = payload["coefficients"]["coefficients"]
    intercept = payload["coefficients"]["intercept"]
    abl = payload["stage_a_ablations_validation"]
    paired = payload["paired_bootstrap_test"]
    sqi = paired["stage_a_plus_b_vs_sqi_only"]
    stage_b = paired["stage_a_plus_b_vs_stage_a_offline"]

    lines = [
        "§3.6 Stage-A logistic regression (validation fit)",
        (
            f"  β={coef['1-QRS overlap']:.2f} (1−QRS overlap), "
            f"β={coef['1-SQI']:.2f} (1−SQI), "
            f"β={coef['1-confidence']:.2f} (1−confidence); "
            f"intercept {intercept:.2f}"
        ),
        (
            f"  Without saliency (SQI + confidence only): "
            f"{abl['no_saliency']['validation_fa_reduction_pct']:.1f}% validation FA reduction"
        ),
        (
            f"  SQI-only LR: {abl['sqi_only_lr']['validation_fa_reduction_pct']:.1f}%; "
            f"saliency-only LR: {abl['saliency_only_lr']['validation_fa_reduction_pct']:.1f}%"
        ),
        "",
        "§4.4 Paired bootstrap FA-reduction differences (DS2 test)",
        (
            f"  Stage A+B vs SQI-only: mean Δ={sqi['mean_delta_pp']:.1f} pp "
            f"p={'<0.001' if stage_b['p_two_sided'] == 0 else format(stage_b['p_two_sided'], '.3f')})"
        ),
        (
            f"  Stage A+B vs Stage A alone: mean Δ={stage_b['mean_delta_pp']:.1f} pp "
            f"(95% CI {stage_b['ci95_pp'][0]:.1f} to {stage_b['ci95_pp'][1]:.1f}; "
            f"p={'<0.001' if stage_b['p_two_sided'] == 0 else format(stage_b['p_two_sided'], '.3f')})"
        ),
    ]
    return "\n".join(lines)


def run_gate_supplementary_analysis(
    *,
    device: torch.device,
    student_variant: str = "baseline",
    n_boot: int | None = None,
    max_alarms: int = 0,
    progress: bool = True,
    results_dir: str | Path | None = None,
    save: bool = True,
) -> dict[str, Any]:
    """Fit Stage-A ablations and paired-bootstrap policy comparisons."""
    n_boot = BOOTSTRAP_N if n_boot is None else n_boot
    results_dir = Path(results_dir or RESULTS_DIR)

    gate_model_path = variant_result_path("gate_model", student_variant, ext=".joblib")
    gate_model = joblib.load(gate_model_path)
    with open(variant_result_path("gate_thresholds", student_variant), encoding="utf-8") as f:
        gate_cfg = json.load(f)

    coef_info = coefficient_report(gate_model)

    student = MultiTaskStudent(c_internal=STUDENT_TRUNK_CHANNELS).to(device)
    student.load_state_dict(
        torch.load(student_checkpoint_path(student_variant), map_location=device),
    )
    teacher_a = TeacherA().to(device)
    teacher_a.load_state_dict(torch.load(Path(MODEL_DIR) / "teacher_a.pt", map_location=device))
    student.eval()
    teacher_a.eval()
    normal_idx = AAMI_CLASS_TO_IDX[MITBIH_NORMAL_CLASS]

    mit_val = MITBIHDataset(MITBIH_DS1_VAL, dry_run=False, use_cache=True)
    val_records, _ = collect_alarm_records(
        mit_val,
        student,
        normal_idx=normal_idx,
        fs=COMMON_FS,
        device=device,
        student_layer=MultiTaskStudent.LAST_CONV,
        max_alarms=max_alarms,
        progress=progress,
    )
    val_alarms = records_to_tuples(val_records)
    ablations = stage_a_ablations(val_alarms)

    mit_test = MITBIHDataset(MITBIH_DS2_TEST, dry_run=False, use_cache=True)
    test_records, cams = collect_alarm_records(
        mit_test,
        student,
        normal_idx=normal_idx,
        fs=COMMON_FS,
        device=device,
        student_layer=MultiTaskStudent.LAST_CONV,
        teacher=teacher_a,
        teacher_layer=TeacherA.LAST_CONV,
        with_cams=True,
        max_alarms=max_alarms,
        progress=progress,
    )

    low = gate_cfg["low_threshold"]
    high = gate_cfg["high_threshold"]
    div_thr = gate_cfg.get("divergence_threshold", 0.5)
    baseline_cfg = gate_cfg.get("baseline_calibration", {})
    sqi_thr = baseline_cfg["sqi_only"]["threshold"]
    conf_thr = baseline_cfg["confidence_only"]["threshold"]

    results = evaluate_baselines(
        test_records,
        gate_model=gate_model,
        low=low,
        high=high,
        divergence_threshold=div_thr,
        normal_idx=normal_idx,
        cams=cams,
        budget=SENSITIVITY_LOSS_BUDGET,
        n_boot=n_boot,
        sqi_threshold=sqi_thr,
        confidence_threshold=conf_thr,
    )
    decisions = policy_decisions(
        test_records,
        gate_model=gate_model,
        low=low,
        high=high,
        divergence_threshold=div_thr,
        normal_idx=normal_idx,
        cams=cams,
        sqi_threshold=sqi_thr,
        confidence_threshold=conf_thr,
    )

    comparisons = {
        "stage_a_plus_b_vs_sqi_only": paired_bootstrap_fa_difference(
            test_records, decisions["stage_a_plus_b"], decisions["sqi_only"], n_boot=n_boot,
        ),
        "stage_a_plus_b_vs_stage_a_offline": paired_bootstrap_fa_difference(
            test_records, decisions["stage_a_plus_b"], decisions["stage_a_offline"], n_boot=n_boot,
        ),
        "stage_a_offline_vs_sqi_only": paired_bootstrap_fa_difference(
            test_records, decisions["stage_a_offline"], decisions["sqi_only"], n_boot=n_boot,
        ),
    }

    payload = {
        "student_variant": student_variant,
        "coefficients": coef_info,
        "stage_a_ablations_validation": ablations,
        "paired_bootstrap_test": comparisons,
        "point_estimates_test": {
            name: {
                "fa_reduction_pct": round(m.false_alarm_reduction * 100, 1),
                "sensitivity_loss_pct": round(m.sensitivity_loss * 100, 2),
            }
            for name, m in results.items()
        },
    }

    if save:
        out_path = results_dir / "gate_supplementary.json"
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        payload["output_path"] = str(out_path)

    return payload
