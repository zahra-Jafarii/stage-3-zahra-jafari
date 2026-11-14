"""Synthetic cleaned frame used by the tests."""
import numpy as np
import pandas as pd


def make_clean_frame(rows=600, seed=0):
    """Build a small frame with the same columns as the cleaning view."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        "id": np.arange(rows),
        "revolving_utilization": rng.uniform(0, 1.5, rows),
        "age": rng.integers(21, 80, rows),
        "past_due_30_59": rng.poisson(0.3, rows),
        "debt_ratio": rng.uniform(0, 2, rows),
        "monthly_income": rng.uniform(1000, 9000, rows),
        "open_credit_lines": rng.integers(0, 15, rows),
        "past_due_90": rng.poisson(0.1, rows),
        "real_estate_loans": rng.integers(0, 4, rows),
        "past_due_60_89": rng.poisson(0.1, rows),
        "dependents": rng.integers(0, 5, rows).astype(float),
        "has_delinquency_code": False,
        "util_was_capped": False,
    })
    risk = 0.05 + 0.25 * (frame["past_due_90"] > 0)
    frame["target"] = (rng.uniform(size=rows) < risk).astype(int)
    frame.loc[rng.choice(rows, 60, replace=False), "monthly_income"] = np.nan
    return frame
