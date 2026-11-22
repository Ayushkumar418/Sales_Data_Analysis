import pandas as pd

from src.forecasting import forecast_sales
from src.reporting import build_executive_report_pdf


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


def test_pdf_report_is_valid_pdf_with_content():
    df = _sales_df()
    forecast_result = forecast_sales(df, periods=3, holdout=3, rolling_folds=3, rolling_horizon=1)
    pdf = build_executive_report_pdf(
        dataset_label="test dataset",
        filtered_df=df,
        forecast_result=forecast_result,
        anomaly_result={"alerts": ["test alert"]},
        filter_summary={"start_date": "2024-01-01", "end_date": "2024-03-31"},
        quality_report={"score": 90.0, "critical_failures": [], "warning_failures": []},
        drift_report={"flagged_count": 0, "alerts": []},
    )
    assert pdf[:5] == b"%PDF-"
    assert len(pdf) > 1000
    # PDF content streams are compressed, so check the trailer/structure instead.
    assert b"%%EOF" in pdf


def test_pdf_report_handles_unknown_profit():
    df = _sales_df()
    df["Profit"] = pd.NA
    forecast_result = forecast_sales(df, periods=3, holdout=3, rolling_folds=3, rolling_horizon=1)
    pdf = build_executive_report_pdf(
        dataset_label="no-profit dataset",
        filtered_df=df,
        forecast_result=forecast_result,
        anomaly_result={"alerts": []},
        filter_summary={},
    )
    assert pdf[:5] == b"%PDF-"
    assert len(pdf) > 1000
