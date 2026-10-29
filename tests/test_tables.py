from src.paper.tables import _ablation_ready, ablation_table


def test_ablation_table_tolerates_missing_classification():
    baseline = {
        "classification": {
            "mitbih_ds2": {"student": {"accuracy": 0.87, "macro_f1": 0.25}},
        },
        "gate": {"policies": {"stage_a_plus_b": {"false_alarm_reduction": 0.3, "sensitivity_loss": 0.01}}},
    }
    saliency = {"classification": None, "gate": None, "explainability": {"sanity_checks": {}}}

    table = ablation_table(baseline, saliency)
    assert "No overlapping metrics" in table


def test_ablation_ready_requires_both_arms():
    baseline = {"classification": {}, "gate": {}}
    saliency_partial = {"classification": None, "gate": None, "thresholds": {}}
    saliency_complete = {"classification": {}, "gate": {}}

    assert _ablation_ready(baseline, saliency_partial) is False
    assert _ablation_ready(baseline, saliency_complete) is True
