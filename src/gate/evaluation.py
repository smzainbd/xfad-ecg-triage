"""Gate evaluation on alarm events.

The gate only acts on beats where the student raises an alarm (prediction != normal),
so calibration (validation) and evaluation (test) must use that same population.
Sensitivity here means *alarm retention*: the fraction of true alarms still delivered
after gating, so a no-gate policy retains 100% by definition.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from src.gate.calibration import fallback_risk
from src.gate.stage_a import decide_stage_a, fraction_of_saliency_mass_inside, stage_a_score
from src.gate.stage_b import stage_b_decision

SUPPRESSING_STAGE_B = {"SUPPRESS", "SUPPRESS_AND_LOG_FOR_REVIEW"}


@dataclass
class AlarmRecord:
    """One student alarm event (student prediction != normal class)."""

    index: int
    y_true: int
    student_pred: int
    teacher_pred: int
    p_conf: float
    sqi: float
    qrs_overlap: float
    normal_idx: int = 0
    group: str = ""

    @property
    def is_false_alarm(self) -> bool:
        return self.y_true == self.normal_idx

    @property
    def is_true_alarm(self) -> bool:
        return self.y_true != self.normal_idx


@dataclass
class GateMetrics:
    """Operating point of one gate policy on a fixed alarm population."""

    baseline: str
    n_alarm_events: int
    n_false_alarms: int
    n_true_alarms: int
    gated_false_alarms: int
    gated_true_alarms: int
    false_alarm_reduction: float
    sensitivity: float
    sensitivity_loss: float
    meets_budget: bool | None = None
    false_alarm_reduction_ci: list[float] | None = None
    sensitivity_ci: list[float] | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def beat_centered_qrs(length: int, fs: int) -> list[tuple[int, int]]:
    """QRS window centred on the annotation when peak detection finds nothing."""
    center = length // 2
    half = int(0.08 * fs)
    return [(max(0, center - half), min(length, center + half))]


def records_to_tuples(records: Sequence[AlarmRecord]) -> list[tuple[float, float, float, bool]]:
    """Convert to the (qrs_overlap, sqi, p_conf, is_false_alarm) form used by calibration."""
    return [(r.qrs_overlap, r.sqi, r.p_conf, r.is_false_alarm) for r in records]


def collect_alarm_records(
    dataset,
    student,
    *,
    normal_idx: int,
    fs: int,
    device,
    student_layer: str,
    teacher=None,
    teacher_layer: str | None = None,
    indices: Iterable[int] | None = None,
    max_alarms: int = 0,
    with_cams: bool = False,
    batch_size: int = 256,
    progress: bool = True,
) -> tuple[list[AlarmRecord], list[tuple[torch.Tensor, torch.Tensor]]]:
    """Collect student alarm events plus their Stage-A features.

    Pass 1 runs batched forward passes over every beat to find alarms; pass 2 computes
    Grad-CAM and signal-quality features for those alarms only.
    """
    from src.explain.gradcam import grad_cam_1d
    from src.utils.qrs import detect_qrs_windows
    from src.utils.sqi import compute_sqi

    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover - tqdm is available in Colab
        def tqdm(x, **_kwargs):
            return x

    scan_indices = list(range(len(dataset))) if indices is None else list(indices)
    groups = getattr(dataset, "groups", []) or []

    alarms: list[tuple[int, int, int, float]] = []
    with torch.no_grad():
        for start in tqdm(range(0, len(scan_indices), batch_size), desc="scan alarms", disable=not progress):
            chunk = scan_indices[start:start + batch_size]
            xb = torch.stack([dataset[i][0] for i in chunk]).to(device)
            probs = F.softmax(student(xb, "A"), dim=1).cpu()
            preds = probs.argmax(1)
            for pos, i in enumerate(chunk):
                pred = int(preds[pos])
                if pred != normal_idx:
                    alarms.append((i, int(dataset[i][1]), pred, float(probs[pos].max())))

    if max_alarms > 0:
        alarms = alarms[:max_alarms]

    records: list[AlarmRecord] = []
    cams: list[tuple[torch.Tensor, torch.Tensor]] = []
    for i, y, pred, p_conf in tqdm(alarms, desc="gate features", disable=not progress):
        x, _ = dataset[i]
        xb = x.unsqueeze(0).to(device)

        s_cam = grad_cam_1d(student, xb, pred, student_layer, task="A")[0]
        t_pred = pred
        t_cam = s_cam
        if teacher is not None and teacher_layer is not None:
            with torch.no_grad():
                t_pred = int(teacher(xb).argmax(1).item())
            t_cam = grad_cam_1d(teacher, xb, t_pred, teacher_layer)[0]

        signal = x.numpy()
        windows = detect_qrs_windows(signal, fs) or beat_centered_qrs(x.shape[-1], fs)
        records.append(AlarmRecord(
            index=i,
            y_true=y,
            student_pred=pred,
            teacher_pred=t_pred,
            p_conf=p_conf,
            sqi=compute_sqi(signal, fs),
            qrs_overlap=fraction_of_saliency_mass_inside(s_cam, windows),
            normal_idx=normal_idx,
            group=str(groups[i]) if i < len(groups) else f"unknown_{i}",
        ))
        if with_cams:
            cams.append((s_cam.detach(), t_cam.detach()))

    return records, cams


def stage_a_risks(records: Sequence[AlarmRecord], gate_model) -> np.ndarray:
    """Stage-A risk per alarm, from the fitted LR model or the fallback rule."""
    if gate_model is None:
        return np.array([fallback_risk(r.qrs_overlap, r.sqi, r.p_conf) for r in records])
    return np.array([stage_a_score(gate_model, r.qrs_overlap, r.sqi, r.p_conf) for r in records])


def policy_decisions(
    records: Sequence[AlarmRecord],
    *,
    gate_model,
    low: float,
    high: float,
    divergence_threshold: float = 0.5,
    normal_idx: int = 0,
    cams: Sequence[tuple[torch.Tensor, torch.Tensor]] | None = None,
    sqi_threshold: float = 0.5,
    confidence_threshold: float = 0.8,
) -> dict[str, np.ndarray]:
    """Per-policy boolean array: True where the alarm still reaches the clinician."""
    risks = stage_a_risks(records, gate_model)
    stage_a = np.array([decide_stage_a(float(r), low, high) for r in risks])

    decisions: dict[str, np.ndarray] = {
        "no_gate": np.ones(len(records), dtype=bool),
        "sqi_only": np.array([r.sqi >= sqi_threshold for r in records]),
        "confidence_only": np.array([r.p_conf >= confidence_threshold for r in records]),
        # Offline fail-safe: an escalation that cannot reach the teacher still alarms.
        "stage_a_offline": stage_a != "SUPPRESS",
    }

    if cams is not None and len(cams) == len(records):
        delivered = decisions["stage_a_offline"].copy()
        for i, record in enumerate(records):
            if stage_a[i] != "ESCALATE":
                continue
            s_cam, t_cam = cams[i]
            decision_b = stage_b_decision(
                record.student_pred,
                record.teacher_pred,
                s_cam,
                t_cam,
                normal_idx,
                connectivity_available=True,
                divergence_threshold=divergence_threshold,
            )
            delivered[i] = decision_b not in SUPPRESSING_STAGE_B
        decisions["stage_a_plus_b"] = delivered

    return decisions


def compute_gate_metrics(
    records: Sequence[AlarmRecord],
    delivered: np.ndarray,
    *,
    name: str,
    budget: float | None = None,
) -> GateMetrics:
    """Aggregate false-alarm reduction and alarm retention for one decision array."""
    is_fa = np.array([r.is_false_alarm for r in records])
    delivered = np.asarray(delivered, dtype=bool)
    n_fa = int(is_fa.sum())
    n_ta = int((~is_fa).sum())
    gated_fa = int((delivered & is_fa).sum())
    gated_ta = int((delivered & ~is_fa).sum())

    fa_red = 1.0 - gated_fa / n_fa if n_fa else 0.0
    sens = gated_ta / n_ta if n_ta else 1.0
    sens_loss = max(0.0, 1.0 - sens)

    return GateMetrics(
        baseline=name,
        n_alarm_events=len(records),
        n_false_alarms=n_fa,
        n_true_alarms=n_ta,
        gated_false_alarms=gated_fa,
        gated_true_alarms=gated_ta,
        false_alarm_reduction=fa_red,
        sensitivity=sens,
        sensitivity_loss=sens_loss,
        meets_budget=None if budget is None else sens_loss <= budget,
    )


def bootstrap_gate_ci(
    records: Sequence[AlarmRecord],
    delivered: np.ndarray,
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, list[float]]:
    """Patient-level bootstrap CIs, resampling record/patient groups with replacement."""
    groups = np.array([r.group for r in records])
    unique_groups = np.unique(groups)
    index_by_group = {g: np.where(groups == g)[0] for g in unique_groups}
    is_fa = np.array([r.is_false_alarm for r in records])
    delivered = np.asarray(delivered, dtype=bool)

    rng = np.random.default_rng(seed)
    fa_reductions, sensitivities = [], []
    for _ in range(n_boot):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([index_by_group[g] for g in sampled])
        fa_mask = is_fa[idx]
        if fa_mask.any():
            fa_reductions.append(1.0 - delivered[idx][fa_mask].sum() / fa_mask.sum())
        if (~fa_mask).any():
            sensitivities.append(delivered[idx][~fa_mask].sum() / (~fa_mask).sum())

    def percentile_ci(values: list[float]) -> list[float]:
        if not values:
            return [float("nan"), float("nan")]
        arr = np.sort(np.asarray(values, dtype=float))
        return [float(np.quantile(arr, alpha / 2)), float(np.quantile(arr, 1 - alpha / 2))]

    return {
        "n_groups": int(len(unique_groups)),
        "false_alarm_reduction_ci": percentile_ci(fa_reductions),
        "sensitivity_ci": percentile_ci(sensitivities),
    }


def evaluate_baselines(
    records: Sequence[AlarmRecord],
    *,
    gate_model,
    low: float,
    high: float,
    divergence_threshold: float = 0.5,
    normal_idx: int = 0,
    cams: Sequence[tuple[torch.Tensor, torch.Tensor]] | None = None,
    sqi_threshold: float = 0.5,
    confidence_threshold: float = 0.8,
    budget: float | None = None,
    n_boot: int = 0,
) -> dict[str, GateMetrics]:
    """Compare the pre-registered gate policies on one alarm population."""
    if not records:
        raise ValueError("No student alarm events to evaluate")

    decisions = policy_decisions(
        records,
        gate_model=gate_model,
        low=low,
        high=high,
        divergence_threshold=divergence_threshold,
        normal_idx=normal_idx,
        cams=cams,
        sqi_threshold=sqi_threshold,
        confidence_threshold=confidence_threshold,
    )

    results: dict[str, GateMetrics] = {}
    for name, delivered in decisions.items():
        metrics = compute_gate_metrics(records, delivered, name=name, budget=budget)
        if n_boot > 0:
            ci = bootstrap_gate_ci(records, delivered, n_boot=n_boot)
            metrics.false_alarm_reduction_ci = ci["false_alarm_reduction_ci"]
            metrics.sensitivity_ci = ci["sensitivity_ci"]
        results[name] = metrics
    return results


def format_results_table(results: dict[str, GateMetrics], *, budget: float) -> str:
    """Markdown table ready to paste into the paper."""
    lines = [
        "| Policy | FA reduction (95% CI) | Alarm retention (95% CI) | Sens. loss | Within budget | Suppressed FA |",
        "|---|---|---|---|---|---|",
    ]
    labels = {
        "no_gate": "No gate (student alarms)",
        "sqi_only": "SQI only (budget-matched)",
        "confidence_only": "Confidence only (budget-matched)",
        "stage_a_offline": "Stage A only (offline)",
        "stage_a_plus_b": "Stage A + Stage B (online)",
    }
    for name, m in results.items():
        fa = f"{m.false_alarm_reduction * 100:.1f}%"
        if m.false_alarm_reduction_ci:
            fa += f" ({m.false_alarm_reduction_ci[0] * 100:.1f}-{m.false_alarm_reduction_ci[1] * 100:.1f})"
        sens = f"{m.sensitivity * 100:.1f}%"
        if m.sensitivity_ci:
            sens += f" ({m.sensitivity_ci[0] * 100:.1f}-{m.sensitivity_ci[1] * 100:.1f})"
        ok = "yes" if m.sensitivity_loss <= budget else "NO"
        label = labels.get(name, name)
        lines.append(
            f"| {label} | {fa} | {sens} | {m.sensitivity_loss * 100:.1f}% | {ok} | "
            f"{m.n_false_alarms - m.gated_false_alarms}/{m.n_false_alarms} |"
        )
    first = next(iter(results.values()), None)
    lines.append("")
    if first is not None:
        lines.append(
            f"Alarm events: {first.n_alarm_events} "
            f"({first.n_false_alarms} false, {first.n_true_alarms} true). "
            f"Pre-registered sensitivity-loss budget: {budget * 100:.0f}%."
        )
    return "\n".join(lines)
