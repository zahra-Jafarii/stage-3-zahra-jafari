"""Save and load the trained model together with its settings."""
import joblib
import sklearn

from credit_risk import settings

DEFAULT_PATH = settings.MODELS_DIR / "credit_model.joblib"


def save_artifact(pipeline, threshold, feature_columns, extra_features,
                  path=DEFAULT_PATH):
    """Store pipeline, threshold, columns and cost matrix in one file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "pipeline": pipeline,
        "threshold": float(threshold),
        "feature_columns": list(feature_columns),
        "extra_features": list(extra_features),
        "cost_fp": settings.COST_FP,
        "cost_fn": settings.COST_FN,
        "seed": settings.SEED,
        "sklearn_version": sklearn.__version__,
    }
    joblib.dump(artifact, path)
    return path


def load_artifact(path=DEFAULT_PATH):
    """Load a saved artifact."""
    return joblib.load(path)
