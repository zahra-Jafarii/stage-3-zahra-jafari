"""Metrics, out-of-fold evaluation, fold stability and subgroup tables."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold

from credit_risk import settings
from credit_risk.costs import confusion_counts, cost_per_1000


@dataclass
class OutOfFoldResult:
    """Out-of-fold probabilities and the fold id of every row."""
    proba: np.ndarray
    fold_ids: np.ndarray


def out_of_fold_proba(pipeline, X, y, n_splits=settings.N_SPLITS, seed=settings.SEED):
    """Predict every row with a model that never saw that row."""
    proba = np.zeros(len(y))
    fold_ids = np.zeros(len(y), dtype=int)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(X, y)):
        model = clone(pipeline).fit(X.iloc[train_idx], y.iloc[train_idx])
        proba[valid_idx] = model.predict_proba(X.iloc[valid_idx])[:, 1]
        fold_ids[valid_idx] = fold
    return OutOfFoldResult(proba=proba, fold_ids=fold_ids)


def classification_metrics(y_true, proba, threshold):
    """Full metric set at one threshold."""
    decision = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_counts(y_true, decision)
    return {
        "threshold": threshold,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "precision": precision_score(y_true, decision, zero_division=0),
        "recall": recall_score(y_true, decision, zero_division=0),
        "f1": f1_score(y_true, decision, zero_division=0),
        "auc": roc_auc_score(y_true, proba),
        "accuracy": accuracy_score(y_true, decision),
        "cost_per_1000": cost_per_1000(y_true, decision),
    }


def per_fold_table(y, result, threshold):
    """Metrics for each fold at a fixed threshold."""
    rows = []
    for fold in np.unique(result.fold_ids):
        mask = result.fold_ids == fold
        row = classification_metrics(y[mask], result.proba[mask], threshold)
        row["fold"] = int(fold)
        rows.append(row)
    return pd.DataFrame(rows).set_index("fold")


def stability_summary(fold_table, columns=("cost_per_1000", "auc", "recall", "precision")):
    """Mean, standard deviation and range across folds."""
    selected = fold_table[list(columns)]
    return pd.DataFrame({
        "mean": selected.mean(), "std": selected.std(),
        "min": selected.min(), "max": selected.max(),
    })


def accuracy_of_constant_healthy(y_true):
    """Accuracy of approving everybody (shows why accuracy misleads)."""
    return float((np.asarray(y_true) == 0).mean())


def subgroup_table(frame, proba, threshold, group_column):
    """Metrics per subgroup at the chosen threshold."""
    rows = []
    for group, part in frame.groupby(group_column, observed=True):
        y_part = part[settings.TARGET].to_numpy()
        p_part = proba[part.index.to_numpy()]
        decision = (p_part >= threshold).astype(int)
        _, fp, fn, tp = confusion_counts(y_part, decision)
        rows.append({
            "group": str(group), "rows": len(part),
            "default_rate": y_part.mean(),
            "reject_rate": decision.mean(),
            "recall": tp / max(tp + fn, 1),
            "false_positive_rate": fp / max((y_part == 0).sum(), 1),
            "cost_per_1000": cost_per_1000(y_part, decision),
        })
    return pd.DataFrame(rows)
