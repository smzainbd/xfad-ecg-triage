"""Render paper tables (markdown + LaTeX) from the evaluation JSON artifacts.

Table 1 - classification: student vs dataset-specific teacher on both held-out sets.
Table 2 - false-alarm gate: every policy on the DS2 student-alarm population.
"""

from __future__ import annotations

import json
from pathlib import Path

POLICY_LABELS = {
    "no_gate": "No gate (student alarms)",
    "sqi_only": "SQI only (budget-matched)",
    "confidence_only": "Confidence only (budget-matched)",
    "stage_a_offline": "Stage A only (offline)",
    "stage_a_plus_b": "Stage A + Stage B (online)",
}

DATASET_LABELS = {
    "mitbih_ds2": ("MIT-BIH DS2 (beats, 5-class AAMI)", "Teacher A"),
    "ptbxl_fold10": ("PTB-XL fold 10 (rhythms, 12-class)", "Teacher B"),
}


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _pct(value: float | None, digits: int = 1) -> str:
    return "-" if value is None else f"{value * 100:.{digits}f}"


def _ci(bounds: list[float] | None) -> str:
    if not bounds or any(b != b for b in bounds):  # NaN check
        return ""
    return f" ({bounds[0] * 100:.1f}-{bounds[1] * 100:.1f})"


def classification_table(data: dict) -> str:
    """Table 1: accuracy, macro-F1 and abnormal AUROC per dataset and model."""
    rows = [
        "| Dataset | Model | Accuracy % (95% CI) | Macro-F1 % | Abnormal AUROC | DeLong p | McNemar p |",
        "|---|---|---|---|---|---|---|",
    ]
    for key, (dataset_label, teacher_label) in DATASET_LABELS.items():
        block = data.get(key)
        if not block:
            continue
        delong_p = (block.get("delong") or {}).get("p_value")
        mcnemar_p = block.get("mcnemar_accuracy_p")
        for model_key, model_label in (("student", "Student (ours)"), ("teacher", teacher_label)):
            model = block[model_key]
            ci = model.get("accuracy_mean_ci")
            ci_text = f" ({ci[1] * 100:.1f}-{ci[2] * 100:.1f})" if ci else ""
            auroc = model.get("auroc_abnormal")
            rows.append(
                f"| {dataset_label} | {model_label} | {_pct(model['accuracy'])}{ci_text} | "
                f"{_pct(model['macro_f1'])} | {'-' if auroc is None else f'{auroc:.3f}'} | "
                f"{'-' if delong_p is None else f'{delong_p:.3f}'} | "
                f"{'-' if mcnemar_p is None else f'{mcnemar_p:.3f}'} |"
            )
        n = block.get("n_samples")
        groups = block.get("n_groups")
        level = "patient/record-level" if block.get("patient_level_ci") else "sample-level (no group ids)"
        rows.append(f"| | | | | | | *n={n}, groups={groups}, CI: {level}* |")
    return "\n".join(rows)


def gate_table(data: dict) -> str:
    """Table 2: false-alarm reduction vs alarm retention for every gate policy."""
    policies = data.get("policies", {})
    budget = data.get("sensitivity_budget", 0.02)
    rows = [
        "| Policy | FA reduction % (95% CI) | Alarm retention % (95% CI) | Sensitivity loss % | "
        f"Within {budget * 100:.0f}% budget |",
        "|---|---|---|---|---|",
    ]
    for key, label in POLICY_LABELS.items():
        metrics = policies.get(key)
        if not metrics:
            continue
        rows.append(
            f"| {label} | {_pct(metrics['false_alarm_reduction'])}"
            f"{_ci(metrics.get('false_alarm_reduction_ci'))} | "
            f"{_pct(metrics['sensitivity'])}{_ci(metrics.get('sensitivity_ci'))} | "
            f"{_pct(metrics['sensitivity_loss'], 2)} | "
            f"{'yes' if metrics.get('meets_budget') else 'no'} |"
        )
    first = next(iter(policies.values()), None)
    if first:
        rows.append("")
        rows.append(
            f"Population: {data.get('population', 'student alarm events')}; "
            f"{first['n_alarm_events']} alarms ({first['n_false_alarms']} false, "
            f"{first['n_true_alarms']} true) from {data.get('n_groups', '?')} records. "
            f"Stage-A scorer: {'learned logistic regression' if data.get('lr_model_fitted') else 'fallback rule'}."
        )
    return "\n".join(rows)


def explainability_table(data: dict) -> str:
    """Table 3: Grad-CAM sanity checks and faithfulness against a random baseline."""
    sanity = data.get("sanity_checks", {})
    faith = data.get("faithfulness", {})
    rows = ["| Check | Value | Interpretation |", "|---|---|---|"]

    for metric, primary in (("spearman", True), ("cosine", False)):
        block = sanity.get(metric)
        if not block:
            continue
        label = "Rank correlation (primary)" if primary else "Cosine similarity (reference)"
        rows.append(
            f"| Weight-randomisation {label} | mean {block['mean']:.3f}, median {block['median']:.3f}, "
            f"max {block['max']:.3f} | {block['status']} |"
        )

    cam = faith.get("deletion_auc_gradcam")
    rnd = faith.get("deletion_auc_random")
    if cam and rnd:
        rows.append(
            f"| Deletion AUC, Grad-CAM order | {cam['mean']:.3f} +/- {cam['sd']:.3f} | "
            "lower is more faithful |"
        )
        rows.append(
            f"| Deletion AUC, random order | {rnd['mean']:.3f} +/- {rnd['sd']:.3f} | "
            f"advantage {faith.get('deletion_auc_advantage_over_random', 0):.3f} |"
        )

    ratios = faith.get("pointing_ratios", {})
    for key, label in (("p_ratio", "P wave"), ("qrs_ratio", "QRS complex"), ("t_ratio", "T wave")):
        block = ratios.get(key)
        if block:
            rows.append(
                f"| Saliency mass in {label} | {block['mean'] * 100:.1f}% +/- {block['sd'] * 100:.1f} | "
                "physiological plausibility |"
            )

    if faith.get("n_examples"):
        rows.append("")
        rows.append(
            f"Sanity checks on {sanity.get('n_examples', '?')} validation beats; "
            f"faithfulness on {faith['n_examples']} beats with "
            f"{faith.get('deletion_steps', '?')} deletion steps."
        )
    return "\n".join(rows)


def calibration_note(data: dict) -> str:
    """One paragraph documenting how the thresholds were set (methods section)."""
    calibration = data.get("calibration", {})
    scorer = "logistic regression" if data.get("lr_model_fitted") else "fallback risk rule"
    return (
        f"Stage-A thresholds were calibrated on {calibration.get('n_alarm_events', '?')} student alarm "
        f"events from the MIT-BIH DS1 validation records ({calibration.get('n_false_alarms', '?')} false "
        f"alarms), using the {scorer} as risk score. The operating point maximised false-alarm reduction "
        f"subject to a pre-registered sensitivity-loss budget of "
        f"{data.get('sensitivity_budget', 0.02) * 100:.0f}%, giving low={data.get('low_threshold', 0):.3f} "
        f"and high={data.get('high_threshold', 0):.3f} "
        f"(validation false-alarm reduction {calibration.get('fa_reduction', 0) * 100:.1f}% at "
        f"{calibration.get('sensitivity_loss', 0) * 100:.2f}% sensitivity loss; "
        f"Stage-B escalation rate {calibration.get('escalation_rate', 0) * 100:.1f}% of alarms)."
    )


def to_latex(markdown_table: str, *, caption: str, label: str) -> str:
    """Convert one pipe table to a booktabs LaTeX table."""
    lines = [line for line in markdown_table.splitlines() if line.strip().startswith("|")]
    if not lines:
        return ""
    rows = [[c.strip() for c in line.strip().strip("|").split("|")] for line in lines]
    rows = [r for r in rows if not all(set(c) <= {"-", ":", ""} for c in r)]
    header, body = rows[0], rows[1:]

    def escape(cell: str) -> str:
        return cell.replace("%", r"\%").replace("_", r"\_").replace("*", "")

    out = [
        r"\begin{table}[t]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{'l' * len(header)}}}",
        r"\toprule",
        " & ".join(escape(c) for c in header) + r" \\",
        r"\midrule",
    ]
    out += [" & ".join(escape(c) for c in row) + r" \\" for row in body]
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out)


def _ablation_ready(baseline: dict, saliency: dict) -> bool:
    """Both arms need held-out classification + gate eval before Table 4 is meaningful."""
    for arm in (baseline, saliency):
        if arm.get("classification") is None or arm.get("gate") is None:
            return False
    return True


def ablation_table(baseline: dict, saliency: dict) -> str:
    """Table 4: baseline KD vs saliency-matching KD on the metrics the gate depends on."""
    rows = ["| Metric | Baseline KD | + Saliency matching | Delta |", "|---|---|---|---|"]

    def add(label: str, a: float | None, b: float | None, *, scale: float = 100.0, digits: int = 1):
        if a is None or b is None:
            return
        av = round(a * scale, digits)
        bv = round(b * scale, digits)
        delta = bv - av
        rows.append(
            f"| {label} | {av:.{digits}f} | {bv:.{digits}f} | {delta:+.{digits}f} |"
        )

    base_cls, sal_cls = baseline.get("classification") or {}, saliency.get("classification") or {}
    for key, (dataset_label, _) in DATASET_LABELS.items():
        a = (base_cls.get(key) or {}).get("student") or {}
        b = (sal_cls.get(key) or {}).get("student") or {}
        add(f"{dataset_label} accuracy %", a.get("accuracy"), b.get("accuracy"))
        add(f"{dataset_label} macro-F1 %", a.get("macro_f1"), b.get("macro_f1"))

    base_x = (baseline.get("explainability") or {}).get("faithfulness", {})
    sal_x = (saliency.get("explainability") or {}).get("faithfulness", {})
    qrs_a = (base_x.get("pointing_ratios", {}).get("qrs_ratio") or {}).get("mean")
    qrs_b = (sal_x.get("pointing_ratios", {}).get("qrs_ratio") or {}).get("mean")
    add("Saliency mass in QRS %", qrs_a, qrs_b)
    del_a = (base_x.get("deletion_auc_gradcam") or {}).get("mean")
    del_b = (sal_x.get("deletion_auc_gradcam") or {}).get("mean")
    add("Deletion AUC (lower better)", del_a, del_b, scale=1.0, digits=3)

    base_g = ((baseline.get("gate") or {}).get("policies") or {}).get("stage_a_plus_b", {})
    sal_g = ((saliency.get("gate") or {}).get("policies") or {}).get("stage_a_plus_b", {})
    add("FA reduction % (Stage A+B)", base_g.get("false_alarm_reduction"), sal_g.get("false_alarm_reduction"))
    add("Sensitivity loss %", base_g.get("sensitivity_loss"), sal_g.get("sensitivity_loss"), digits=2)

    if len(rows) == 2:
        return "No overlapping metrics between the two variants yet."
    return "\n".join(rows)


def load_variant_results(results_dir: str | Path, variant: str = "baseline") -> dict:
    """Load every result JSON for one student variant, keyed by artifact."""
    results_dir = Path(results_dir)
    suffix = "" if variant == "baseline" else f"_{variant}"
    return {
        "classification": _load(results_dir / f"classification_eval{suffix}.json"),
        "gate": _load(results_dir / f"gate_eval_ds2{suffix}.json"),
        "thresholds": _load(results_dir / f"gate_thresholds{suffix}.json"),
        "explainability": _load(results_dir / f"explainability_eval{suffix}.json"),
    }


def build_all_tables(results_dir: str | Path, variant: str = "baseline") -> dict[str, str]:
    """Build every available table from the JSON files in ``results_dir``."""
    loaded = load_variant_results(results_dir, variant)
    classification = loaded["classification"]
    gate = loaded["gate"]
    thresholds = loaded["thresholds"]
    explainability = loaded["explainability"]

    tables: dict[str, str] = {}
    if classification:
        tables["table1_classification.md"] = classification_table(classification)
    if gate:
        tables["table2_gate.md"] = gate_table(gate)
    if explainability:
        tables["table3_explainability.md"] = explainability_table(explainability)
    if thresholds:
        tables["calibration_note.md"] = calibration_note(thresholds)

    if "table1_classification.md" in tables:
        tables["table1_classification.tex"] = to_latex(
            tables["table1_classification.md"],
            caption="Held-out classification performance of the distilled multitask student "
                    "against dataset-specific teachers.",
            label="tab:classification",
        )
    if "table2_gate.md" in tables:
        tables["table2_gate.tex"] = to_latex(
            tables["table2_gate.md"],
            caption="False-alarm reduction on MIT-BIH DS2 student alarm events under a "
                    "pre-registered 2\\% sensitivity-loss budget.",
            label="tab:gate",
        )
    if "table3_explainability.md" in tables:
        tables["table3_explainability.tex"] = to_latex(
            tables["table3_explainability.md"],
            caption="Grad-CAM sanity checks and faithfulness of the student saliency maps.",
            label="tab:explainability",
        )

    if variant == "baseline":
        saliency = load_variant_results(results_dir, "saliency")
        if _ablation_ready(loaded, saliency):
            tables["table4_ablation.md"] = ablation_table(loaded, saliency)
            tables["table4_ablation.tex"] = to_latex(
                tables["table4_ablation.md"],
                caption="Ablation: adding a saliency-matching term to knowledge distillation.",
                label="tab:ablation",
            )
    return tables


def write_tables(results_dir: str | Path, out_dir: str | Path, variant: str = "baseline") -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, text in build_all_tables(results_dir, variant).items():
        path = out_dir / name
        path.write_text(text + "\n", encoding="utf-8")
        written.append(path)
    return written
