"""Preprocessing pipelines; every statistic is fitted on training data only."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MISSING_POLICIES = ("median", "median_indicator", "native")


def build_numeric_branch(policy, scale):
    """Numeric steps for one missing-value policy."""
    steps = []
    if policy in ("median", "median_indicator"):
        steps.append(("imputer", SimpleImputer(
            strategy="median", add_indicator=(policy == "median_indicator"))))
    if scale:
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps) if steps else "passthrough"


def build_categorical_branch():
    """Categorical steps: mode imputation and one-hot encoding."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])


def build_preprocessor(numeric_columns, categorical_columns=(),
                       policy="median", scale=True):
    """ColumnTransformer for numeric and (optional) categorical columns."""
    if policy not in MISSING_POLICIES:
        raise ValueError(f"Unknown missing policy: {policy}")
    parts = [("numeric", build_numeric_branch(policy, scale), list(numeric_columns))]
    if categorical_columns:
        parts.append(("categorical", build_categorical_branch(),
                      list(categorical_columns)))
    return ColumnTransformer(parts, remainder="drop")
