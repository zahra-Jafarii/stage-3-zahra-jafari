"""Model factories: two different model families."""
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from credit_risk import settings
from credit_risk.preprocessing import build_preprocessor

MODEL_NAMES = ("logistic", "boosting")


def make_estimator(name, seed=settings.SEED):
    """Create an unfitted estimator by name."""
    if name == "logistic":
        return LogisticRegression(max_iter=1000, random_state=seed)
    if name == "boosting":
        return HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, random_state=seed)
    raise ValueError(f"Unknown model: {name}")


def needs_scaling(name):
    """Logistic regression is scale-sensitive; tree boosting is not."""
    return name == "logistic"


def build_pipeline(name, numeric_columns, categorical_columns=(),
                   policy="median", seed=settings.SEED):
    """Preprocessor + estimator as one pipeline."""
    preprocessor = build_preprocessor(
        numeric_columns, categorical_columns, policy, scale=needs_scaling(name))
    return Pipeline([
        ("preprocess", preprocessor),
        ("model", make_estimator(name, seed)),
    ])
