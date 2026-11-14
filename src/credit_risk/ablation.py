"""Feature impact: cost and AUC with and without each engineered feature."""
import pandas as pd
from sklearn.metrics import roc_auc_score

from credit_risk import settings
from credit_risk.costs import best_threshold
from credit_risk.evaluation import out_of_fold_proba
from credit_risk.models import build_pipeline


def score_feature_set(model_name, frame, columns, policy, seed=settings.SEED):
    """Out-of-fold best cost and AUC for one list of feature columns."""
    y = frame[settings.TARGET]
    pipeline = build_pipeline(model_name, columns, policy=policy, seed=seed)
    result = out_of_fold_proba(pipeline, frame[columns], y, seed=seed)
    threshold, cost = best_threshold(y.to_numpy(), result.proba)
    return {"threshold": threshold, "cost_per_1000": cost,
            "auc": roc_auc_score(y, result.proba)}


def drop_one_table(model_name, frame, columns, candidates, policy,
                   seed=settings.SEED):
    """Compare the full set with the set minus each candidate feature."""
    rows = [{"removed": "none", **score_feature_set(
        model_name, frame, columns, policy, seed)}]
    for name in candidates:
        reduced = [col for col in columns if col != name]
        rows.append({"removed": name, **score_feature_set(
            model_name, frame, reduced, policy, seed)})
    table = pd.DataFrame(rows)
    table["cost_change_vs_full"] = table["cost_per_1000"] - table.loc[0, "cost_per_1000"]
    return table
