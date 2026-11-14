"""Simple rule-based baselines that involve no learning."""
import numpy as np
import pandas as pd

from credit_risk import settings
from credit_risk.costs import cost_per_1000


def rule_always_healthy(frame):
    """Approve everyone."""
    return np.zeros(len(frame), dtype=int)


def rule_always_risky(frame):
    """Reject everyone."""
    return np.ones(len(frame), dtype=int)


def rule_any_90_days_late(frame):
    """Reject anyone with at least one 90+ days late record."""
    return (frame["past_due_90"].fillna(0) > 0).astype(int).to_numpy()


def rule_any_past_due(frame):
    """Reject anyone with any past-due record."""
    columns = ["past_due_30_59", "past_due_60_89", "past_due_90"]
    return (frame[columns].fillna(0).sum(axis=1) > 0).astype(int).to_numpy()


BASELINE_RULES = {
    "always_healthy": rule_always_healthy,
    "always_risky": rule_always_risky,
    "any_90_days_late": rule_any_90_days_late,
    "any_past_due": rule_any_past_due,
}


def evaluate_baselines(frame, cost_fp=settings.COST_FP, cost_fn=settings.COST_FN):
    """Cost per 1000 of every baseline rule, sorted from best to worst."""
    y_true = frame[settings.TARGET].to_numpy()
    rows = []
    for name, rule in BASELINE_RULES.items():
        rows.append({
            "rule": name,
            "cost_per_1000": cost_per_1000(y_true, rule(frame), cost_fp, cost_fn),
        })
    return pd.DataFrame(rows).sort_values("cost_per_1000").reset_index(drop=True)


def reduction_vs_baseline(model_cost, baseline_cost):
    """Relative cost reduction of the model against the baseline."""
    return 1 - model_cost / baseline_cost
