"""Row-wise feature engineering (no statistics, so no leakage)."""
import numpy as np
import pandas as pd

PAST_DUE_COLUMNS = ["past_due_30_59", "past_due_60_89", "past_due_90"]


def add_total_past_due(frame):
    """Sum of all past-due counts; NaN only if every count is missing."""
    return frame[PAST_DUE_COLUMNS].sum(axis=1, min_count=1)


def add_serious_delinquency_flag(frame):
    """1 if the applicant was ever 60+ days late."""
    serious = (frame["past_due_60_89"] > 0) | (frame["past_due_90"] > 0)
    return serious.astype(int)


def add_income_per_person(frame):
    """Monthly income divided by household size (dependents + the applicant)."""
    return frame["monthly_income"] / (frame["dependents"] + 1)


def add_monthly_debt_estimate(frame):
    """Debt ratio times income gives the estimated monthly debt payment."""
    return frame["debt_ratio"] * frame["monthly_income"]


def add_real_estate_share(frame):
    """Share of real estate loans among all open credit lines."""
    return frame["real_estate_loans"] / (frame["open_credit_lines"] + 1)


def add_over_limit_flag(frame):
    """1 if revolving utilization is above 100 percent."""
    return (frame["revolving_utilization"] > 1).astype(int)


FEATURE_BUILDERS = {
    "total_past_due": add_total_past_due,
    "serious_delinquency_flag": add_serious_delinquency_flag,
    "income_per_person": add_income_per_person,
    "monthly_debt_estimate": add_monthly_debt_estimate,
    "real_estate_share": add_real_estate_share,
    "over_limit_flag": add_over_limit_flag,
}


def build_features(frame, names=None):
    """Return a copy of the frame with the requested engineered features."""
    result = frame.copy()
    for name in names or FEATURE_BUILDERS:
        result[name] = FEATURE_BUILDERS[name](result)
    return result.replace([np.inf, -np.inf], np.nan)
