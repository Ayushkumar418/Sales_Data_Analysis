import pandas as pd


def compute_kpis(df: pd.DataFrame) -> dict:
    total_sales = float(df["Sales"].sum())
    total_profit = float(df["Profit"].sum())
    total_orders = int(df["Order ID"].nunique())
    avg_order_value = total_sales / total_orders if total_orders else 0.0
    profit_margin = (total_profit / total_sales) * 100 if total_sales else 0.0

    return {
        "total_sales": total_sales,
        "total_profit": total_profit,
        "total_orders": total_orders,
        "avg_order_value": avg_order_value,
        "profit_margin": profit_margin,
    }


def generate_business_suggestions(df: pd.DataFrame, forecast_df: pd.DataFrame) -> list[str]:
    suggestions: list[str] = []
    if df.empty:
        return ["No records match the current filters. Broaden filters to generate actionable suggestions."]

    category_sales = df.groupby("Category")["Sales"].sum().sort_values(ascending=False)
    best_category = category_sales.index[0]

    region_profit = df.groupby("Region")["Profit"].sum().sort_values()
    weakest_region = region_profit.index[0]

    sub_category_profit = (
        df.groupby("Sub-Category")["Profit"].sum().sort_values(ascending=False).head(3)
    )
    top_sub_categories = ", ".join(sub_category_profit.index.tolist())

    suggestions.append(
        f"Increase inventory depth for '{best_category}' and its best-selling SKUs to avoid stockouts."
    )
    suggestions.append(
        f"Run targeted promotions in '{weakest_region}' where profitability is weakest to improve regional performance."
    )
    suggestions.append(
        f"Prioritize high-margin sub-categories ({top_sub_categories}) in bundle offers and featured placements."
    )

    if not forecast_df.empty:
        first_value = float(forecast_df["Predicted Sales"].iloc[0])
        last_value = float(forecast_df["Predicted Sales"].iloc[-1])
        direction = "upward" if last_value >= first_value else "downward"
        suggestions.append(
            f"Forecast indicates an {direction} sales trend over the next {len(forecast_df)} months; align procurement plans accordingly."
        )

    return suggestions

