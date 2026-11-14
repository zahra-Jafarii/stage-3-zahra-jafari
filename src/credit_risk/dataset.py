"""Contract and Dataset for the credit data (stage 1 package)."""
from nyc311_factory.contract import Column, Contract
from nyc311_factory.dataset import Dataset

from credit_risk.settings import TRAIN_TABLE

CONTRACT = Contract(
    name=TRAIN_TABLE,
    source_columns=(
        Column("id", "int", nullable=False),
        Column("target", "int", nullable=True, min_value=0, max_value=1),
        Column("revolving_utilization", "float", nullable=False, min_value=0),
        Column("age", "int", nullable=False, min_value=18, max_value=120),
        Column("past_due_30_59", "int", nullable=False, min_value=0),
        Column("debt_ratio", "float", nullable=False, min_value=0),
        Column("monthly_income", "float", nullable=True, min_value=0),
        Column("open_credit_lines", "int", nullable=False, min_value=0),
        Column("past_due_90", "int", nullable=False, min_value=0),
        Column("real_estate_loans", "int", nullable=False, min_value=0),
        Column("past_due_60_89", "int", nullable=False, min_value=0),
        Column("dependents", "int", nullable=True, min_value=0),
    ),
    derived_columns=(),
    key_column="id",
    content_columns=(
        "target", "revolving_utilization", "age", "past_due_30_59",
        "debt_ratio", "monthly_income", "open_credit_lines", "past_due_90",
        "real_estate_loans", "past_due_60_89", "dependents",
    ),
    required_columns=("id",),
)

DATASET = Dataset(name=TRAIN_TABLE, build_contract=lambda config: CONTRACT)
