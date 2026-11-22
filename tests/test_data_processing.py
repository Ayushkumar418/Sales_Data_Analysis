import numpy as np
import pandas as pd
import pytest

from src.anomalies import detect_business_anomalies
from src.data_processing import (
    build_synthetic_demo_dataset,
    clean_and_engineer_dataset,
    normalize_sales_dataset,
)
from src.insights import compute_kpis, generate_business_suggestions
from src.quality import assess_data_quality


def _base_df(**overrides):
    data = {
        "Order Date": pd.to_datetime(["2024-01-10", "2024-02-10", "2024-03-10"]),
        "Sales": [1000.0, 2000.0, 1500.0],
        "Category": ["A", "B", "A"],
        "Sub-Category": ["A1", "B1", "A2"],
        "Product Name": ["P1", "P2", "P3"],
        "Order ID": ["O1", "O2", "O3"],
    }
    data.update(overrides)
    return pd.DataFrame(data)


def test_profit_without_column_stays_unknown_not_12_percent():
    df = normalize_sales_dataset(_base_df())
    assert df["Profit"].isna().all()
    # The old fallback filled profit with 12% of sales (120, 240, 180).
    assert not (df["Profit"] == df["Sales"] * 0.12).any()


def test_profit_column_is_kept_when_supplied():
    df = normalize_sales_dataset(_base_df(Profit=[100.0, 200.0, 150.0]))
    assert df["Profit"].tolist() == [100.0, 200.0, 150.0]


def test_region_and_segment_are_unknown_not_hash_bucketed():
    df = normalize_sales_dataset(_base_df())
    assert set(df["Region"]) == {"Unknown"}
    assert set(df["Customer Segment"]) == {"Unknown"}


def test_no_valid_dates_raises_instead_of_inventing_timeline():
    df = _base_df()
    df["Order Date"] = pd.NaT
    with pytest.raises(ValueError, match="No valid order dates"):
        normalize_sales_dataset(df)
    df2 = _base_df()
    df2 = df2.drop(columns=["Order Date"])
    with pytest.raises(ValueError, match="No valid order dates"):
        normalize_sales_dataset(df2)


def test_kpis_show_profit_as_unknown():
    df = normalize_sales_dataset(_base_df())
    kpis = compute_kpis(df)
    assert kpis["total_profit"] is None
    assert kpis["profit_margin"] is None
    assert kpis["profit_known"] is False
    assert kpis["total_sales"] == 4500.0


def test_suggestions_skip_margin_when_profit_unknown():
    df = normalize_sales_dataset(_base_df())
    suggestions = generate_business_suggestions(df, pd.DataFrame())
    assert any("Profit is not available" in s for s in suggestions)
    assert not any("lost ₹" in s for s in suggestions)


def test_anomalies_do_not_false_flag_low_margin_when_profit_unknown():
    df = normalize_sales_dataset(_base_df())
    result = detect_business_anomalies(df)
    assert result["low_margin_categories"].empty
    assert any("margin checks skipped" in a for a in result["alerts"])


def test_quality_flags_unknown_region():
    df = normalize_sales_dataset(_base_df())
    quality = assess_data_quality(df)
    checks = quality["checks"]
    row = checks[checks["Check"] == "Unknown Region Ratio"]
    assert not row.empty
    assert float(row.iloc[0]["Value"]) == 100.0
    assert row.iloc[0]["Status"] == "FAIL"


def test_demo_builder_is_explicitly_synthetic():
    assert build_synthetic_demo_dataset is clean_and_engineer_dataset
    assert "SYNTHETIC" in build_synthetic_demo_dataset.__doc__
    assert "demo" in build_synthetic_demo_dataset.__doc__.lower()
