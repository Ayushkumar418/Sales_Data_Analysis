import pandas as pd

from src.insights import (
    compute_category_profitability,
    compute_discount_insights,
    compute_monthly_growth,
    compute_product_insights,
    generate_business_suggestions,
)


def _sales_df():
    return pd.DataFrame(
        {
            "Order ID": ["O1", "O2", "O3", "O4", "O5", "O6"],
            "Order Date": pd.to_datetime(
                ["2024-01-10", "2024-01-20", "2024-02-10", "2024-02-20", "2024-03-10", "2024-03-20"]
            ),
            "Sales": [1000.0, 500.0, 2000.0, 300.0, 1500.0, 100.0],
            "Profit": [200.0, 100.0, 400.0, -50.0, 300.0, -20.0],
            "Category": ["Electronics", "Electronics", "Furniture", "Furniture", "Electronics", "Furniture"],
            "Sub-Category": ["Phones", "Phones", "Chairs", "Chairs", "Laptops", "Tables"],
            "Region": ["North", "North", "South", "South", "North", "South"],
            "Customer Segment": ["Consumer", "Consumer", "Corporate", "Corporate", "Consumer", "Corporate"],
            "Product Name": ["Phone A", "Phone A", "Chair B", "Chair B", "Laptop C", "Table D"],
            "Quantity": [1, 1, 2, 1, 1, 1],
            "Unit Price": [1000.0, 500.0, 1000.0, 300.0, 1500.0, 100.0],
            "Discount Percentage": [0.0, 5.0, 10.0, 25.0, 0.0, 30.0],
            "Rating": [4.5, 4.0, 4.2, 3.5, 4.8, 3.0],
            "Rating Count": [10, 5, 8, 2, 20, 1],
        }
    )


def test_product_insights_top_and_worst():
    df = _sales_df()
    result = compute_product_insights(df)
    assert result["product_count"] == 4
    assert result["top_products"].iloc[0]["Product Name"] == "Chair B"
    # Worst by total profit is Table D (-20), not Chair B (350 total).
    assert result["worst_profit_products"].iloc[0]["Product Name"] == "Table D"
    assert float(result["worst_profit_products"].iloc[0]["Profit"]) == -20.0
    # Top product share is a real percentage.
    assert 0 < result["top_share_pct"] <= 100


def test_discount_insights_buckets():
    df = _sales_df()
    result = compute_discount_insights(df)
    assert result["buckets"]["Orders"].sum() == 6
    assert result["high_discount_margin_pct"] is not None
    # 25% and 30% discount orders lost money overall in this fixture.
    assert result["high_discount_margin_pct"] < 0


def test_monthly_growth_mom():
    df = _sales_df()
    growth = compute_monthly_growth(df)
    assert growth["months"] == 3
    assert growth["latest_sales"] == 1600.0
    assert growth["previous_sales"] == 2300.0
    assert abs(growth["mom_change_pct"] - ((1600 - 2300) / 2300 * 100)) < 0.01


def test_category_profitability():
    df = _sales_df()
    table = compute_category_profitability(df)
    assert set(table["Category"]) == {"Electronics", "Furniture"}
    assert "Profit Margin (%)" in table.columns


def test_suggestions_are_specific_with_numbers():
    df = _sales_df()
    forecast = pd.DataFrame(
        {
            "Month": pd.to_datetime(["2024-04-01"]),
            "Predicted Sales": [1800.0],
            "Lower Bound (80%)": [1500.0],
            "Upper Bound (80%)": [2100.0],
        }
    )
    suggestions = generate_business_suggestions(df, forecast)
    text = " ".join(suggestions)
    assert "Chair B" in text
    assert "₹" in text
    assert "1500" in text or "1,500" in text
