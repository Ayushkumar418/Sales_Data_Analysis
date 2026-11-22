import numpy as np
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


def _margin(sales: float, profit: float) -> float:
    return (profit / sales * 100) if sales else 0.0


def compute_monthly_growth(df: pd.DataFrame) -> dict:
    """Month-over-month growth from the monthly sales rollup."""
    if df.empty or "Order Date" not in df.columns:
        return {
            "latest_month": None,
            "latest_sales": 0.0,
            "previous_sales": 0.0,
            "mom_change_pct": None,
            "best_month": None,
            "best_sales": 0.0,
            "avg_monthly_sales": 0.0,
            "months": 0,
        }
    monthly = df.groupby(pd.Grouper(key="Order Date", freq="MS"))["Sales"].sum().sort_index()
    monthly = monthly[monthly.index.notna()]
    if monthly.empty:
        return {
            "latest_month": None,
            "latest_sales": 0.0,
            "previous_sales": 0.0,
            "mom_change_pct": None,
            "best_month": None,
            "best_sales": 0.0,
            "avg_monthly_sales": 0.0,
            "months": 0,
        }
    latest_sales = float(monthly.iloc[-1])
    previous_sales = float(monthly.iloc[-2]) if len(monthly) > 1 else 0.0
    mom = ((latest_sales - previous_sales) / previous_sales * 100) if previous_sales else None
    return {
        "latest_month": monthly.index[-1],
        "latest_sales": latest_sales,
        "previous_sales": previous_sales,
        "mom_change_pct": mom,
        "best_month": monthly.idxmax(),
        "best_sales": float(monthly.max()),
        "avg_monthly_sales": float(monthly.mean()),
        "months": int(len(monthly)),
    }


def compute_product_insights(df: pd.DataFrame, top_n: int = 5) -> dict:
    """Top / worst products with their share of sales and margin."""
    empty = {
        "top_products": pd.DataFrame(columns=["Product Name", "Sales", "Profit", "Profit Margin (%)", "Share of Sales (%)"]),
        "worst_profit_products": pd.DataFrame(columns=["Product Name", "Sales", "Profit", "Profit Margin (%)", "Orders"]),
        "top_share_pct": 0.0,
        "product_count": 0,
    }
    if df.empty or "Product Name" not in df.columns:
        return empty
    total_sales = float(df["Sales"].sum())
    grouped = (
        df.groupby("Product Name", as_index=False)
        .agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"), Orders=("Order ID", "nunique"))
        .sort_values("Sales", ascending=False)
    )
    grouped["Profit Margin (%)"] = [
        _margin(float(s), float(p)) for s, p in zip(grouped["Sales"], grouped["Profit"])
    ]
    grouped["Share of Sales (%)"] = [
        (float(s) / total_sales * 100) if total_sales else 0.0 for s in grouped["Sales"]
    ]
    top = grouped.head(top_n)[
        ["Product Name", "Sales", "Profit", "Profit Margin (%)", "Share of Sales (%)"]
    ].copy()
    worst = grouped.sort_values("Profit").head(top_n)[
        ["Product Name", "Sales", "Profit", "Profit Margin (%)", "Orders"]
    ].copy()
    return {
        "top_products": top,
        "worst_profit_products": worst,
        "top_share_pct": float(top["Share of Sales (%)"].sum()) if not top.empty else 0.0,
        "product_count": int(grouped.shape[0]),
    }


def compute_discount_insights(df: pd.DataFrame) -> dict:
    """Profitability by discount bucket — where deep discounts eat margin."""
    empty = {
        "buckets": pd.DataFrame(columns=["Discount Bucket", "Orders", "Sales", "Profit", "Profit Margin (%)"]),
        "avg_discount_pct": 0.0,
        "high_discount_margin_pct": None,
        "high_discount_sales": 0.0,
    }
    if df.empty or "Discount Percentage" not in df.columns:
        return empty
    work = df.copy()
    work["Discount Percentage"] = pd.to_numeric(work["Discount Percentage"], errors="coerce").fillna(0)
    bins = [-0.01, 0.0, 10.0, 20.0, float("inf")]
    labels = ["No discount", "0–10%", "10–20%", "20%+"]
    work["Discount Bucket"] = pd.cut(work["Discount Percentage"], bins=bins, labels=labels)
    buckets = (
        work.groupby("Discount Bucket", observed=True)
        .agg(Orders=("Order ID", "nunique"), Sales=("Sales", "sum"), Profit=("Profit", "sum"))
        .reset_index()
        .rename(columns={"Discount Bucket": "Discount Bucket"})
    )
    buckets["Profit Margin (%)"] = [
        _margin(float(s), float(p)) for s, p in zip(buckets["Sales"], buckets["Profit"])
    ]
    high = work[work["Discount Percentage"] >= 20]
    high_sales = float(high["Sales"].sum())
    return {
        "buckets": buckets,
        "avg_discount_pct": float(work["Discount Percentage"].mean()),
        "high_discount_margin_pct": _margin(high_sales, float(high["Profit"].sum())) if not high.empty else None,
        "high_discount_sales": high_sales,
    }


def compute_category_profitability(df: pd.DataFrame) -> pd.DataFrame:
    """Category table with sales, profit, margin and share — for the UI."""
    cols = ["Category", "Sales", "Profit", "Profit Margin (%)", "Orders", "Share of Sales (%)"]
    if df.empty or "Category" not in df.columns:
        return pd.DataFrame(columns=cols)
    total_sales = float(df["Sales"].sum())
    grouped = (
        df.groupby("Category", as_index=False)
        .agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"), Orders=("Order ID", "nunique"))
        .sort_values("Sales", ascending=False)
    )
    grouped["Profit Margin (%)"] = [
        _margin(float(s), float(p)) for s, p in zip(grouped["Sales"], grouped["Profit"])
    ]
    grouped["Share of Sales (%)"] = [
        (float(s) / total_sales * 100) if total_sales else 0.0 for s in grouped["Sales"]
    ]
    return grouped[cols]


def generate_business_suggestions(df: pd.DataFrame, forecast_df: pd.DataFrame) -> list[str]:
    """Specific, numbers-backed suggestions — not generic templates."""
    suggestions: list[str] = []
    if df.empty:
        return ["No records match the current filters. Broaden filters to generate actionable suggestions."]

    total_sales = float(df["Sales"].sum())
    growth = compute_monthly_growth(df)
    products = compute_product_insights(df)
    discount = compute_discount_insights(df)

    if not products["top_products"].empty:
        top = products["top_products"].iloc[0]
        suggestions.append(
            f"Protect stock for '{top['Product Name']}': ₹{float(top['Sales']):,.0f} sales "
            f"({float(top['Share of Sales (%)']):.1f}% of total) at {float(top['Profit Margin (%)']):.1f}% margin. "
            f"Top 5 products together are {products['top_share_pct']:.1f}% of sales."
        )
    if not products["worst_profit_products"].empty:
        worst = products["worst_profit_products"].iloc[0]
        if float(worst["Profit"]) < 0:
            suggestions.append(
                f"'{worst['Product Name']}' lost ₹{abs(float(worst['Profit'])):,.0f} "
                f"across {int(worst['Orders'])} orders — review its price or discount, or delist it."
            )

    if discount["high_discount_margin_pct"] is not None:
        suggestions.append(
            f"Orders with 20%+ discount made ₹{discount['high_discount_sales']:,.0f} sales "
            f"but only {discount['high_discount_margin_pct']:.1f}% margin "
            f"(average discount is {discount['avg_discount_pct']:.1f}%). Cap deep discounts on low-margin items."
        )

    if growth["mom_change_pct"] is not None and growth["latest_month"] is not None:
        direction = "up" if growth["mom_change_pct"] >= 0 else "down"
        suggestions.append(
            f"Latest month ({growth['latest_month'].strftime('%Y-%m')}) sales were "
            f"₹{growth['latest_sales']:,.0f}, {direction} {abs(growth['mom_change_pct']):.1f}% "
            f"vs the previous month (average month: ₹{growth['avg_monthly_sales']:,.0f})."
        )

    category_sales = df.groupby("Category")["Sales"].sum().sort_values(ascending=False)
    if not category_sales.empty:
        best_category = category_sales.index[0]
        share = float(category_sales.iloc[0]) / total_sales * 100 if total_sales else 0.0
        suggestions.append(
            f"'{best_category}' is the biggest category at ₹{float(category_sales.iloc[0]):,.0f} "
            f"({share:.1f}% of sales) — keep its best-selling SKUs in stock first."
        )

    region_profit = df.groupby("Region")["Profit"].sum().sort_values()
    if not region_profit.empty:
        weakest_region = region_profit.index[0]
        weakest_margin = _margin(
            float(df[df["Region"] == weakest_region]["Sales"].sum()), float(region_profit.iloc[0])
        )
        suggestions.append(
            f"Weakest region is '{weakest_region}' with {weakest_margin:.1f}% margin — "
            f"check discounting and delivery cost there before spending more on promotion."
        )

    if not forecast_df.empty and "Predicted Sales" in forecast_df.columns:
        first_value = float(forecast_df["Predicted Sales"].iloc[0])
        last_value = float(forecast_df["Predicted Sales"].iloc[-1])
        direction = "upward" if last_value >= first_value else "downward"
        if "Lower Bound (80%)" in forecast_df.columns and "Upper Bound (80%)" in forecast_df.columns:
            low = float(forecast_df["Lower Bound (80%)"].iloc[0])
            high = float(forecast_df["Upper Bound (80%)"].iloc[0])
            suggestions.append(
                f"Forecast is {direction} over the next {len(forecast_df)} months: "
                f"next month ₹{first_value:,.0f} (80% range ₹{low:,.0f}–₹{high:,.0f}). "
                f"Plan procurement for the lower bound, not the point forecast."
            )
        else:
            suggestions.append(
                f"Forecast indicates an {direction} sales trend over the next {len(forecast_df)} months; align procurement plans accordingly."
            )

    return suggestions
