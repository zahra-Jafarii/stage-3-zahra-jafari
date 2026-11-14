"""Single place for constants: seed, paths, cost matrix, column names."""
from pathlib import Path

SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CURATED_DIR = PROJECT_ROOT / "data" / "curated"
MODELS_DIR = PROJECT_ROOT / "models"
SQL_DIR = Path(__file__).resolve().parent / "sql"

# Database objects (same database as stage 2, separate schema)
SCHEMA = "stage3"
TRAIN_TABLE = "credit_training"
CLEAN_VIEW = "v_credit_clean"
DATABASE_URL_ENV = "DATABASE_URL"

# Cost matrix (units per decision)
COST_FP = 1.0
COST_FN = 12.0

# Domain rules applied in the SQL layer (fixed constants, not data statistics)
SENTINEL_CODES = (96, 98)
UTILIZATION_CAP = 10.0
DEBT_RATIO_CAP = 10.0

HOLDOUT_SIZE = 0.15
N_SPLITS = 5

ID_COLUMN = "id"
TARGET = "target"

BASE_FEATURES = [
    "revolving_utilization",
    "age",
    "past_due_30_59",
    "debt_ratio",
    "monthly_income",
    "open_credit_lines",
    "past_due_90",
    "real_estate_loans",
    "past_due_60_89",
    "dependents",
    "has_delinquency_code",
    "util_was_capped",
]

NEW_FEATURES = [
    "total_past_due",
    "serious_delinquency_flag",
    "income_per_person",
    "monthly_debt_estimate",
    "real_estate_share",
    "over_limit_flag",
]

CATEGORICAL_FEATURES = []

THRESHOLD_GRID_START = 0.01
THRESHOLD_GRID_STOP = 0.60
THRESHOLD_GRID_STEP = 0.0025
