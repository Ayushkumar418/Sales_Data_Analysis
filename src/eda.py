from pathlib import Path

import matplotlib
import pandas as pd
import seaborn as sns

from .config import FIGURES_DIR

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")


def monthly_sales_trend(df: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        df.groupby(pd.Grouper(key="Order Date", freq="MS"))["Sales"]
        .sum()
        .reset_index()
        .rename(columns={"Sales": "Monthly Sales"})
    )
    return monthly


def category_sales(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("Category", as_index=False)["Sales"]
        .sum()
        .sort_values("Sales", ascending=False)
    )


def sub_category_sales(df: pd.DataFrame, top_n: int = 12) -> pd.DataFrame:
    grouped = (
        df.groupby("Sub-Category", as_index=False)["Sales"]
        .sum()
        .sort_values("Sales", ascending=False)
    )
    return grouped.head(top_n)


def region_performance(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("Region", as_index=False)[["Sales", "Profit"]]
        .sum()
        .sort_values("Sales", ascending=False)
    )


def profit_sales_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df[["Sales", "Profit", "Unit Price", "Quantity", "Discount Percentage", "Rating"]]


def plot_monthly_sales_trend(df: pd.DataFrame) -> plt.Figure:
    monthly = monthly_sales_trend(df)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    sns.lineplot(data=monthly, x="Order Date", y="Monthly Sales", marker="o", ax=ax, color="#1f77b4")
    ax.set_title("Monthly Sales Trend")
    ax.set_xlabel("Month")
    ax.set_ylabel("Sales")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    return fig


def plot_category_sales(df: pd.DataFrame) -> plt.Figure:
    grouped = category_sales(df)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    sns.barplot(
        data=grouped,
        x="Category",
        y="Sales",
        hue="Category",
        legend=False,
        ax=ax,
        palette="Blues_r",
    )
    ax.set_title("Category-wise Sales")
    ax.set_xlabel("Category")
    ax.set_ylabel("Sales")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    return fig


def plot_sub_category_sales(df: pd.DataFrame) -> plt.Figure:
    grouped = sub_category_sales(df)
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(
        data=grouped,
        x="Sub-Category",
        y="Sales",
        hue="Sub-Category",
        legend=False,
        ax=ax,
        palette="viridis",
    )
    ax.set_title("Top Sub-Category Sales")
    ax.set_xlabel("Sub-Category")
    ax.set_ylabel("Sales")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig


def plot_region_performance(df: pd.DataFrame) -> plt.Figure:
    grouped = region_performance(df)
    melted = grouped.melt(id_vars="Region", value_vars=["Sales", "Profit"], var_name="Metric")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=melted, x="Region", y="value", hue="Metric", ax=ax, palette="Set2")
    ax.set_title("Region-wise Performance")
    ax.set_xlabel("Region")
    ax.set_ylabel("Amount")
    fig.tight_layout()
    return fig


def plot_profit_vs_sales(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=df,
        x="Sales",
        y="Profit",
        hue="Category",
        size="Quantity",
        alpha=0.7,
        ax=ax,
    )
    ax.set_title("Profit vs Sales Analysis")
    ax.set_xlabel("Sales")
    ax.set_ylabel("Profit")
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    matrix = profit_sales_matrix(df).corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    fig.tight_layout()
    return fig


def save_example_visualizations(df: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    charts = {
        "monthly_sales_trend.png": plot_monthly_sales_trend(df),
        "category_sales.png": plot_category_sales(df),
        "sub_category_sales.png": plot_sub_category_sales(df),
        "region_performance.png": plot_region_performance(df),
        "profit_vs_sales.png": plot_profit_vs_sales(df),
        "correlation_heatmap.png": plot_correlation_heatmap(df),
    }

    for filename, fig in charts.items():
        fig.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
        plt.close(fig)
