"""Copy the Kaggle CSV with snake_case headers (values are not touched)."""
import csv
import shutil
from pathlib import Path

from credit_risk.settings import RAW_DIR

SOURCE_NAME = "cs-training.csv"
TARGET_NAME = "credit_training.csv"

HEADER_MAP = {
    "": "id",
    "SeriousDlqin2yrs": "target",
    "RevolvingUtilizationOfUnsecuredLines": "revolving_utilization",
    "age": "age",
    "NumberOfTime30-59DaysPastDueNotWorse": "past_due_30_59",
    "DebtRatio": "debt_ratio",
    "MonthlyIncome": "monthly_income",
    "NumberOfOpenCreditLinesAndLoans": "open_credit_lines",
    "NumberOfTimes90DaysLate": "past_due_90",
    "NumberRealEstateLoansOrLines": "real_estate_loans",
    "NumberOfTime60-89DaysPastDueNotWorse": "past_due_60_89",
    "NumberOfDependents": "dependents",
}


def read_header(path):
    """Return the first row of the CSV file as a list of names."""
    with open(path, newline="", encoding="utf-8") as handle:
        return next(csv.reader(handle))


def rename_header(header):
    """Map original column names to snake_case; fail on unknown names."""
    unknown = [name for name in header if name not in HEADER_MAP]
    if unknown:
        raise ValueError(f"Unexpected columns in source file: {unknown}")
    return [HEADER_MAP[name] for name in header]


def copy_with_new_header(source, target):
    """Write the new header, then copy the remaining lines byte for byte."""
    new_header = rename_header(read_header(source))
    with open(source, "rb") as src, open(target, "wb") as dst:
        src.readline()
        dst.write((",".join(new_header) + "\n").encode("utf-8"))
        shutil.copyfileobj(src, dst)


def main():
    source = Path(RAW_DIR) / SOURCE_NAME
    target = Path(RAW_DIR) / TARGET_NAME
    copy_with_new_header(source, target)
    print(f"written: {target}")


if __name__ == "__main__":
    main()
