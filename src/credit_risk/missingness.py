"""Evidence about whether missing values are random."""
import numpy as np
import pandas as pd

from credit_risk import settings


def missing_summary(frame, columns):
    """Count and share of missing values per column."""
    counts = frame[columns].isna().sum()
    return pd.DataFrame({"missing": counts, "share": counts / len(frame)})


def default_rate_by_missing(frame, column):
    """Default rate and group size for missing vs present values."""
    flag = frame[column].isna().rename(f"{column}_missing")
    grouped = frame.groupby(flag)[settings.TARGET].agg(["mean", "count"])
    return grouped.rename(columns={"mean": "default_rate", "count": "rows"})


def standardized_gap(frame, column, other_columns):
    """Standardized mean gap of other columns between missing and present rows."""
    missing = frame[frame[column].isna()]
    present = frame[frame[column].notna()]
    rows = {}
    for other in other_columns:
        pooled = np.sqrt((missing[other].var() + present[other].var()) / 2)
        gap = (missing[other].mean() - present[other].mean()) / pooled
        rows[other] = float(gap)
    return pd.Series(rows, name="standardized_gap").sort_values()
