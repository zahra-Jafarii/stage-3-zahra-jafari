"""Checks that make the train/evaluation boundary inspectable."""
import numpy as np


def assert_disjoint_ids(train_ids, eval_ids):
    """No applicant may appear in both parts."""
    overlap = set(np.asarray(train_ids)) & set(np.asarray(eval_ids))
    if overlap:
        raise AssertionError(f"{len(overlap)} ids appear in both parts")


def imputer_medians(fitted_pipeline):
    """Medians learned by the numeric imputer of a fitted pipeline."""
    numeric = fitted_pipeline.named_steps["preprocess"].named_transformers_["numeric"]
    return numeric.named_steps["imputer"].statistics_


def assert_fit_uses_train_only(fitted_pipeline, train_frame, columns):
    """Learned medians must equal the medians of the training part alone."""
    expected = train_frame[columns].median().to_numpy()
    learned = imputer_medians(fitted_pipeline)
    if not np.allclose(expected, learned, equal_nan=True):
        raise AssertionError("Imputer statistics differ from training medians")


def audit_table():
    """Leakage audit: is each engineered feature known before the loan decision."""
    rows = [
        ("total_past_due", "sum of past-due counts", "history up to application"),
        ("serious_delinquency_flag", "60+ days late history", "history up to application"),
        ("income_per_person", "income / (dependents + 1)", "applicant data at application"),
        ("monthly_debt_estimate", "debt_ratio * income", "applicant data at application"),
        ("real_estate_share", "real estate loans / open lines", "history up to application"),
        ("over_limit_flag", "utilization above 1", "balance at application"),
    ]
    import pandas as pd
    return pd.DataFrame(rows, columns=["feature", "definition", "known_at_decision"])
