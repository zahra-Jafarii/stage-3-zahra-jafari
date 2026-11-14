"""Database access: engine, cleaning view and reading through SQLAlchemy."""
import os

import pandas as pd

from credit_risk import settings


def get_engine(env_name=settings.DATABASE_URL_ENV):
    """Create an engine from the environment variable (never from a file)."""
    from sqlalchemy import create_engine

    url = os.environ.get(env_name)
    if not url:
        raise RuntimeError(f"Environment variable {env_name} is not set")
    return create_engine(url)


def render_clean_query(schema, table):
    """Fill the SQL template with the domain constants from settings."""
    template = (settings.SQL_DIR / "clean_credit.sql").read_text(encoding="utf-8")
    return template.format(
        schema=schema,
        table=table,
        sentinels=", ".join(str(code) for code in settings.SENTINEL_CODES),
        util_cap=settings.UTILIZATION_CAP,
        debt_ratio_cap=settings.DEBT_RATIO_CAP,
    )


def create_clean_view(engine, schema=settings.SCHEMA, table=settings.TRAIN_TABLE,
                      view=settings.CLEAN_VIEW):
    """Create or replace the cleaning view over the training table."""
    from sqlalchemy import text

    query = render_clean_query(schema, table)
    statement = f"CREATE OR REPLACE VIEW {schema}.{view} AS {query}"
    with engine.begin() as connection:
        connection.execute(text(statement))


def read_sql_frame(engine, sql):
    """Run a SELECT statement and return a DataFrame."""
    from sqlalchemy import text

    with engine.connect() as connection:
        return pd.read_sql(text(sql), connection)


def load_clean_frame(engine, schema=settings.SCHEMA, view=settings.CLEAN_VIEW):
    """Read the cleaned training data from the view."""
    return read_sql_frame(engine, f"SELECT * FROM {schema}.{view}")


def load_clean_frame_for_table(engine, schema, table):
    """Apply the same cleaning query to any table with the same columns."""
    return read_sql_frame(engine, render_clean_query(schema, table))


def count_rows(engine, schema, table):
    """Return the number of rows in a table."""
    frame = read_sql_frame(engine, f"SELECT COUNT(*) AS n FROM {schema}.{table}")
    return int(frame.loc[0, "n"])
