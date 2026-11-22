"""CLI job runner: the --table-name path must use the same quoting as app.py.

Regression test for the gap left by the quick-fixes PR: app.py quoted the
table name, but src/jobs.py still interpolated it raw into SQL, so
`--table-name 'sales; DROP TABLE x'` reached the database unchecked.
"""

import pandas as pd
import pytest
from sqlalchemy import create_engine

from src.jobs import _load_dataset


def _sales_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Order ID": ["A-1", "A-2"],
            "Order Date": ["2025-01-05", "2025-02-10"],
            "Sales": [1000.0, 2000.0],
            "Profit": [100.0, 200.0],
            "Category": ["Furniture", "Furniture"],
            "Sub-Category": ["Chairs", "Tables"],
            "Region": ["North", "South"],
            "Customer Segment": ["Consumer", "Corporate"],
        }
    )


def test_load_dataset_rejects_malicious_table_name(tmp_path):
    db_path = tmp_path / "test.db"
    url = f"sqlite:///{db_path}"
    with pytest.raises(ValueError, match="Invalid table name"):
        _load_dataset("database", url, "sales; DROP TABLE sales", "")


def test_load_dataset_accepts_plain_table_name(tmp_path):
    db_path = tmp_path / "test.db"
    url = f"sqlite:///{db_path}"
    engine = create_engine(url)
    _sales_frame().to_sql("sales", engine, index=False)
    engine.dispose()

    df, label = _load_dataset("database", url, "sales", "")
    assert label == "database_table_sales"
    assert len(df) == 2
    assert set(df["Sales"]) == {1000.0, 2000.0}
