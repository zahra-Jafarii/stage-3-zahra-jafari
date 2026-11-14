"""Run the stage 1 factory on the credit CSV and load it into the database."""
from nyc311_factory.config import load_config
from nyc311_factory.pipeline import run_pipeline

from credit_risk.dataset import DATASET
from credit_risk.settings import PROJECT_ROOT


def load_training_data(config_path=None):
    """Validate, quarantine and write the training file to the database."""
    path = config_path or PROJECT_ROOT / "config.json"
    config = load_config(str(path))
    return run_pipeline(config, dataset=DATASET)


def main():
    manifest = load_training_data()
    print(manifest)


if __name__ == "__main__":
    main()
