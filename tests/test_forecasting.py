import numpy as np
import pandas as pd

from src.forecasting import (
    AVAILABLE_MODELS,
    HOLT_WINTERS_MODEL,
    benchmark_models,
    build_monthly_sales_series,
    forecast_sales,
)


def _monthly_df(values, start="2023-01-01"):
    dates = pd.date_range(start, periods=len(values), freq="MS")
    return pd.DataFrame({"Order Date": dates, "Sales": values})


def test_missing_month_is_interpolated_not_zero():
    df = pd.DataFrame(
        {
            "Order Date": pd.to_datetime(["2023-01-15", "2023-02-15", "2023-04-15"]),
            "Sales": [100.0, 200.0, 400.0],
        }
    )
    series = build_monthly_sales_series(df)
    assert len(series) == 4
    # March sits between Feb (200) and Apr (400): interpolated, not 0.
    assert series.iloc[2] == 300.0


def test_holt_winters_on_seasonal_series():
    rng = np.random.default_rng(42)
    months = np.arange(36)
    values = 100 + 2 * months + 20 * np.sin(2 * np.pi * months / 12) + rng.normal(0, 3, 36)
    values = np.clip(values, a_min=10, a_max=None)
    series = pd.Series(values, index=pd.date_range("2023-01-01", periods=36, freq="MS"), name="Sales")
    assert HOLT_WINTERS_MODEL in AVAILABLE_MODELS
    bench = benchmark_models(series, holdout=6)
    assert HOLT_WINTERS_MODEL in bench["Model"].tolist()
    row = bench[bench["Model"] == HOLT_WINTERS_MODEL].iloc[0]
    assert np.isfinite(row["RMSE"]) and np.isfinite(row["MAPE"])


def test_forecast_sales_has_confidence_band():
    rng = np.random.default_rng(7)
    months = np.arange(30)
    values = 200 + 3 * months + 15 * np.sin(2 * np.pi * months / 12) + rng.normal(0, 5, 30)
    df = _monthly_df(np.clip(values, a_min=10, a_max=None))
    result = forecast_sales(df, periods=6, holdout=6, rolling_folds=3, rolling_horizon=1)
    forecast = result["forecast"]
    assert list(forecast.columns) == ["Month", "Predicted Sales", "Lower Bound (80%)", "Upper Bound (80%)"]
    assert len(forecast) == 6
    assert (forecast["Lower Bound (80%)"] <= forecast["Predicted Sales"]).all()
    assert (forecast["Predicted Sales"] <= forecast["Upper Bound (80%)"]).all()
    assert (forecast["Predicted Sales"] >= 0).all()
    assert result["confidence_level"] == 0.80
    assert result["selected_model"] in AVAILABLE_MODELS
    # Band widens with horizon.
    widths = forecast["Upper Bound (80%)"] - forecast["Lower Bound (80%)"]
    assert widths.iloc[-1] >= widths.iloc[0]


def test_forecast_sales_on_short_series_does_not_crash():
    df = _monthly_df([100.0, 110.0, 105.0, 120.0, 130.0, 125.0, 140.0, 150.0])
    result = forecast_sales(df, periods=3, holdout=3, rolling_folds=3, rolling_horizon=1)
    assert len(result["forecast"]) == 3
    assert (result["forecast"]["Predicted Sales"] >= 0).all()
