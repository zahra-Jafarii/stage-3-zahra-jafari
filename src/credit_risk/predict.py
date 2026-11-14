"""Score a fresh file with the saved model (same schema as training)."""
import argparse
from pathlib import Path

import pandas as pd

from credit_risk import settings
from credit_risk.artifacts import DEFAULT_PATH, load_artifact
from credit_risk.features import build_features

PREDICT_TABLE = "credit_scoring_input"


def predict_frame(clean_frame, artifact):
    """Probability and decision for an already cleaned frame."""
    features = build_features(clean_frame, artifact["extra_features"])
    columns = artifact["feature_columns"]
    proba = artifact["pipeline"].predict_proba(features[columns])[:, 1]
    result = pd.DataFrame({settings.ID_COLUMN: clean_frame[settings.ID_COLUMN]})
    result["probability"] = proba
    result["decision"] = (proba >= artifact["threshold"]).astype(int)
    return result


def load_new_file_to_database(csv_path):
    """Run the stage 1 factory on the new file into a separate table."""
    from nyc311_factory.config import Config
    from nyc311_factory.pipeline import run_pipeline

    from credit_risk.dataset import DATASET

    config = Config(
        quarantine_threshold=0.05,
        curated_dir=str(settings.CURATED_DIR / "predict"),
        source_type="csv",
        input_path=str(csv_path),
        destinations=["database"],
        database_schema=settings.SCHEMA,
        database_cases_table=PREDICT_TABLE,
        database_quarantine_table=f"{PREDICT_TABLE}_quarantine",
        database_load_mode="replace",
    )
    return run_pipeline(config, dataset=DATASET)


def predict_file(csv_path, model_path=DEFAULT_PATH):
    """New CSV -> factory -> database -> same cleaning query -> predictions."""
    from credit_risk.db import get_engine, load_clean_frame_for_table

    load_new_file_to_database(csv_path)
    engine = get_engine()
    clean = load_clean_frame_for_table(engine, settings.SCHEMA, PREDICT_TABLE)
    return predict_frame(clean, load_artifact(model_path))


def main():
    parser = argparse.ArgumentParser(description="Score a new credit file")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=str(settings.CURATED_DIR / "predictions.csv"))
    args = parser.parse_args()
    result = predict_file(Path(args.input))
    result.to_csv(args.output, index=False)
    print(f"written: {args.output} ({len(result)} rows)")


if __name__ == "__main__":
    main()
