"""Cost matrix, expected cost and threshold selection."""
import numpy as np
import pandas as pd

from credit_risk import settings


def threshold_grid():
    """Candidate thresholds from settings."""
    return np.arange(
        settings.THRESHOLD_GRID_START,
        settings.THRESHOLD_GRID_STOP,
        settings.THRESHOLD_GRID_STEP,
    )


def confusion_counts(y_true, y_pred):
    """Return tn, fp, fn, tp for binary arrays."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    return tn, fp, fn, tp


def total_cost(y_true, y_pred, cost_fp=settings.COST_FP, cost_fn=settings.COST_FN):
    """Total cost of a set of decisions."""
    _, fp, fn, _ = confusion_counts(y_true, y_pred)
    return fp * cost_fp + fn * cost_fn


def cost_per_1000(y_true, y_pred, cost_fp=settings.COST_FP, cost_fn=settings.COST_FN):
    """Expected cost per 1000 applicants."""
    return total_cost(y_true, y_pred, cost_fp, cost_fn) / len(y_true) * 1000


def cost_curve(y_true, proba, thresholds, cost_fp=settings.COST_FP,
               cost_fn=settings.COST_FN):
    """Cost per 1000 applicants for each candidate threshold."""
    rows = []
    for threshold in thresholds:
        decision = (proba >= threshold).astype(int)
        rows.append({
            "threshold": float(threshold),
            "cost_per_1000": cost_per_1000(y_true, decision, cost_fp, cost_fn),
        })
    return pd.DataFrame(rows)


def best_threshold(y_true, proba, thresholds=None, cost_fp=settings.COST_FP,
                   cost_fn=settings.COST_FN):
    """Return (threshold, cost_per_1000) with the lowest cost."""
    grid = threshold_grid() if thresholds is None else thresholds
    curve = cost_curve(y_true, proba, grid, cost_fp, cost_fn)
    best = curve.loc[curve["cost_per_1000"].idxmin()]
    return float(best["threshold"]), float(best["cost_per_1000"])


def theoretical_threshold(cost_fp=settings.COST_FP, cost_fn=settings.COST_FN):
    """Optimal threshold if probabilities are perfectly calibrated."""
    return cost_fp / (cost_fp + cost_fn)


def threshold_sensitivity(y_true, proba, fn_ratios, cost_fp=settings.COST_FP):
    """Best threshold and cost for several FN:FP cost ratios."""
    rows = []
    for ratio in fn_ratios:
        threshold, cost = best_threshold(
            y_true, proba, cost_fp=cost_fp, cost_fn=cost_fp * ratio)
        rows.append({
            "fn_to_fp_ratio": ratio,
            "best_threshold": threshold,
            "cost_per_1000": cost,
            "theoretical_threshold": theoretical_threshold(cost_fp, cost_fp * ratio),
        })
    return pd.DataFrame(rows)
