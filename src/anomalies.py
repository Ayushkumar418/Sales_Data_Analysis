import numpy as np
import pandas as pd


def _compute_monthly_anomalies(df: pd.DataFrame, z_threshold: float = 2.0) -> pd.DataFrame:
    monthly = (
        df.groupby(pd.Grouper(key="Order Date", freq="MS"))[["Sales", "Profit"]]
        .sum()
        .reset_index()
        .sort_values("Order Date")
    )
    if len(monthly) < 6:
        monthly["Sales Z-Score"] = np.nan
        monthly["Is Anomaly"] = False
        return monthly.iloc[0:0]

    mean_val = monthly["Sales"].mean()
    std_val = monthly["Sales"].std(ddof=0)
    if std_val == 0:
        monthly["Sales Z-Score"] = 0.0
        monthly["Is Anomaly"] = False
        return monthly.iloc[0:0]

    monthly["Sales Z-Score"] = (monthly["Sales"] - mean_val) / std_val
    monthly["Is Anomaly"] = monthly["Sales Z-Score"].abs() >= z_threshold
    anomalies = monthly[monthly["Is Anomaly"]].copy()
    return anomalies[["Order Date", "Sales", "Profit", "Sales Z-Score"]]


def detect_business_anomalies(
    df: pd.DataFrame,
    z_threshold: float = 2.0,
    low_margin_threshold: float = 5.0,
    negative_profit_cutoff: float = 0.0,
) -> dict:
    if df.empty:
        return {
            "alerts": ["No data is available for anomaly detection."],
            "monthly_anomalies": pd.DataFrame(
                columns=["Order Date", "Sales", "Profit", "Sales Z-Score"]
            ),
            "negative_profit_orders": pd.DataFrame(columns=df.columns),
            "low_margin_categories": pd.DataFrame(
                columns=["Category", "Sales", "Profit", "Profit Margin (%)"]
            ),
            "region_performance": pd.DataFrame(
                columns=["Region", "Sales", "Profit", "Profit Margin (%)"]
            ),
        }

    alerts: list[str] = []
    monthly_anomalies = _compute_monthly_anomalies(df, z_threshold=z_threshold)

    if not monthly_anomalies.empty:
        alerts.append(
            f"Detected {len(monthly_anomalies)} monthly sales anomalies at z-threshold {z_threshold:.1f}."
        )

    negative_profit_orders = df[df["Profit"] < negative_profit_cutoff].copy()
    if not negative_profit_orders.empty:
        alerts.append(
            f"Found {len(negative_profit_orders)} transactions with profit below {negative_profit_cutoff:.2f}."
        )

    category_perf = (
        df.groupby("Category", as_index=False)[["Sales", "Profit"]]
        .sum()
        .sort_values("Sales", ascending=False)
    )
    category_perf["Profit Margin (%)"] = np.where(
        category_perf["Sales"] == 0,
        0.0,
        (category_perf["Profit"] / category_perf["Sales"]) * 100,
    )
    low_margin_categories = category_perf[
        category_perf["Profit Margin (%)"] < low_margin_threshold
    ].copy()
    if not low_margin_categories.empty:
        alerts.append(
            f"{len(low_margin_categories)} categories are below {low_margin_threshold:.1f}% profit margin."
        )

    region_performance = (
        df.groupby("Region", as_index=False)[["Sales", "Profit"]]
        .sum()
        .sort_values("Profit", ascending=True)
    )
    region_performance["Profit Margin (%)"] = np.where(
        region_performance["Sales"] == 0,
        0.0,
        (region_performance["Profit"] / region_performance["Sales"]) * 100,
    )
    if not region_performance.empty:
        weakest_region = region_performance.iloc[0]
        alerts.append(
            f"Weakest region is '{weakest_region['Region']}' with margin {weakest_region['Profit Margin (%)']:.2f}%."
        )

    if not alerts:
        alerts.append("No major anomaly patterns detected under current filter selection.")

    return {
        "alerts": alerts,
        "monthly_anomalies": monthly_anomalies.sort_values("Order Date", ascending=False),
        "negative_profit_orders": negative_profit_orders.sort_values("Profit").head(25),
        "low_margin_categories": low_margin_categories.sort_values("Profit Margin (%)"),
        "region_performance": region_performance,
    }
