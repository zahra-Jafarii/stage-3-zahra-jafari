"""Permutation importance on raw input columns."""
import pandas as pd
from sklearn.inspection import permutation_importance

from credit_risk import settings


def top_features(fitted_pipeline, X_valid, y_valid, n_repeats=5, seed=settings.SEED):
    """Permutation importance (AUC drop) sorted from most to least important."""
    result = permutation_importance(
        fitted_pipeline, X_valid, y_valid, scoring="roc_auc",
        n_repeats=n_repeats, random_state=seed, n_jobs=1)
    table = pd.DataFrame({
        "feature": X_valid.columns,
        "auc_drop_mean": result.importances_mean,
        "auc_drop_std": result.importances_std,
    })
    return table.sort_values("auc_drop_mean", ascending=False).reset_index(drop=True)
