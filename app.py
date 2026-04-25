import io
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.anomalies import detect_business_anomalies
from src.config import DEFAULT_FORECAST_PERIODS
from src.data_processing import ensure_processed_dataset, prepare_sales_dataset
from src.forecasting import forecast_sales
from src.insights import compute_kpis, generate_business_suggestions
from src.quality import DriftThresholds, QualityThresholds, assess_data_drift, assess_data_quality
from src.reporting import (
    build_executive_report_dataframe,
    build_executive_report_markdown,
    save_executive_report_bundle,
)


st.set_page_config(
    page_title="Sales Data Analysis Dashboard",
    page_icon=":bar_chart:",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    return ensure_processed_dataset(force_rebuild=False)


@st.cache_data(show_spinner=False)
def load_uploaded_raw_data(file_name: str, file_bytes: bytes) -> pd.DataFrame:
    extension = Path(file_name).suffix.lower()
    buffer = io.BytesIO(file_bytes)

    if extension == ".csv":
        for encoding in ("utf-8", "latin-1"):
            try:
                raw_df = pd.read_csv(buffer, encoding=encoding)
                break
            except UnicodeDecodeError:
                buffer.seek(0)
        else:
            buffer.seek(0)
            raw_df = pd.read_csv(buffer, encoding_errors="ignore")
    elif extension in {".xlsx", ".xls"}:
        raw_df = pd.read_excel(buffer)
    else:
        raise ValueError("Unsupported file type. Please upload CSV or Excel files.")

    return raw_df


@st.cache_data(show_spinner=False)
def load_uploaded_data(file_name: str, file_bytes: bytes) -> pd.DataFrame:
    raw_df = load_uploaded_raw_data(file_name, file_bytes)
    return prepare_sales_dataset(raw_df)


@st.cache_data(show_spinner=False)
def load_database_data(
    connection_url: str, table_name: str = "", sql_query: str = ""
) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        from sqlalchemy import create_engine
    except ImportError as exc:
        raise ValueError("sqlalchemy is required for database loading. Install dependencies again.") from exc

    if not connection_url.strip():
        raise ValueError("Connection URL is required.")
    if not table_name.strip() and not sql_query.strip():
        raise ValueError("Provide either a table name or a SQL query.")

    engine = create_engine(connection_url)
    with engine.connect() as conn:
        if sql_query.strip():
            raw_df = pd.read_sql_query(sql_query, conn)
        else:
            raw_df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)

    if raw_df.empty:
        raise ValueError("Database query returned no rows.")

    processed_df = prepare_sales_dataset(raw_df)
    return raw_df, processed_df


def apply_filters(
    df: pd.DataFrame,
    regions: list[str],
    categories: list[str],
    segments: list[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    filtered_df = df.copy()
    filtered_df = filtered_df[filtered_df["Region"].isin(regions)]
    filtered_df = filtered_df[filtered_df["Category"].isin(categories)]
    filtered_df = filtered_df[filtered_df["Customer Segment"].isin(segments)]
    filtered_df = filtered_df[
        (filtered_df["Order Date"] >= start_date) & (filtered_df["Order Date"] <= end_date)
    ]
    return filtered_df


def format_currency(value: float) -> str:
    return f"${value:,.2f}"


def create_monthly_trend_figure(df: pd.DataFrame) -> go.Figure:
    monthly = (
        df.groupby(pd.Grouper(key="Order Date", freq="MS"))["Sales"]
        .sum()
        .reset_index()
        .rename(columns={"Sales": "Monthly Sales"})
    )
    fig = px.line(
        monthly,
        x="Order Date",
        y="Monthly Sales",
        markers=True,
        title="Monthly Sales Trend",
        template="plotly_white",
    )
    fig.update_layout(hovermode="x unified", xaxis_title="Month", yaxis_title="Sales")
    return fig


def create_profit_vs_sales_figure(df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        df,
        x="Sales",
        y="Profit",
        color="Category",
        size="Quantity",
        hover_data=["Product Name", "Region", "Sub-Category", "Customer Segment"],
        title="Profit vs Sales Analysis",
        template="plotly_white",
    )
    fig.update_layout(legend_title_text="Category")
    return fig


def create_correlation_heatmap_figure(df: pd.DataFrame) -> go.Figure:
    numeric_cols = ["Sales", "Profit", "Unit Price", "Quantity", "Discount Percentage", "Rating"]
    corr = df[numeric_cols].corr(numeric_only=True)
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        origin="lower",
        title="Feature Correlation Heatmap",
        template="plotly_white",
    )
    fig.update_layout(coloraxis_colorbar_title="Correlation")
    return fig


def create_category_sales_figure(df: pd.DataFrame) -> go.Figure:
    grouped = (
        df.groupby("Category", as_index=False)["Sales"]
        .sum()
        .sort_values("Sales", ascending=False)
    )
    fig = px.bar(
        grouped,
        x="Category",
        y="Sales",
        color="Category",
        title="Category-wise Sales",
        template="plotly_white",
    )
    fig.update_layout(showlegend=False, xaxis_title="Category", yaxis_title="Sales")
    return fig


def create_sub_category_sales_figure(df: pd.DataFrame, top_n: int = 12) -> go.Figure:
    grouped = (
        df.groupby("Sub-Category", as_index=False)["Sales"]
        .sum()
        .sort_values("Sales", ascending=False)
        .head(top_n)
    )
    fig = px.bar(
        grouped,
        x="Sub-Category",
        y="Sales",
        color="Sub-Category",
        title="Top Sub-Category Sales",
        template="plotly_white",
    )
    fig.update_layout(showlegend=False, xaxis_title="Sub-Category", yaxis_title="Sales")
    return fig


def create_region_performance_figure(df: pd.DataFrame) -> go.Figure:
    grouped = (
        df.groupby("Region", as_index=False)[["Sales", "Profit"]]
        .sum()
        .sort_values("Sales", ascending=False)
    )
    melted = grouped.melt(id_vars="Region", value_vars=["Sales", "Profit"], var_name="Metric")
    fig = px.bar(
        melted,
        x="Region",
        y="value",
        color="Metric",
        barmode="group",
        title="Region-wise Performance",
        template="plotly_white",
    )
    fig.update_layout(xaxis_title="Region", yaxis_title="Amount")
    return fig


def render_forecast_chart(historical_df: pd.DataFrame, forecast_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fitted_column = "Fitted Sales" if "Fitted Sales" in historical_df.columns else "Trend Sales"

    if not historical_df.empty:
        fig.add_trace(
            go.Scatter(
                x=historical_df["Month"],
                y=historical_df["Actual Sales"],
                mode="lines+markers",
                name="Actual Sales",
                line=dict(color="#1f77b4", width=3),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=historical_df["Month"],
                y=historical_df[fitted_column],
                mode="lines",
                name="Model Fit",
                line=dict(color="#ff7f0e", dash="dash", width=3),
            )
        )

    if not forecast_df.empty:
        fig.add_trace(
            go.Scatter(
                x=forecast_df["Month"],
                y=forecast_df["Predicted Sales"],
                mode="lines+markers",
                name="Forecast Sales",
                line=dict(color="#2ca02c", width=3),
            )
        )

    fig.update_layout(
        title="Sales Forecast (Best-Selected Model)",
        xaxis_title="Month",
        yaxis_title="Sales",
        hovermode="x unified",
        template="plotly_white",
    )
    return fig


def render_dataset_info(
    full_df: pd.DataFrame,
    filtered_df: pd.DataFrame,
    dataset_label: str,
    raw_df: pd.DataFrame | None = None,
) -> None:
    st.subheader("Dataset Information")
    st.caption(f"Current data source: **{dataset_label}**")

    if raw_df is not None:
        st.markdown("### Uploaded Raw Dataset")
        col1, col2, col3 = st.columns(3)
        col1.metric("Raw Rows", f"{len(raw_df):,}")
        col2.metric("Raw Columns", f"{raw_df.shape[1]:,}")
        col3.metric("Raw Missing Values", f"{int(raw_df.isna().sum().sum()):,}")

        with st.expander("Raw Columns and Data Types", expanded=False):
            raw_dtypes = (
                raw_df.dtypes.astype(str)
                .reset_index()
                .rename(columns={"index": "Column", 0: "Data Type"})
            )
            st.dataframe(raw_dtypes, use_container_width=True)

        with st.expander("Raw Sample Records", expanded=False):
            st.dataframe(raw_df.head(20), use_container_width=True)

    st.markdown("### Processed Dataset Used for Dashboard")
    min_date = full_df["Order Date"].min().date()
    max_date = full_df["Order Date"].max().date()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Processed Rows", f"{len(full_df):,}")
    col2.metric("Processed Columns", f"{full_df.shape[1]:,}")
    col3.metric("Date Range Start", str(min_date))
    col4.metric("Date Range End", str(max_date))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Filtered Rows", f"{len(filtered_df):,}")
    col2.metric("Unique Categories", f"{full_df['Category'].nunique():,}")
    col3.metric("Unique Regions", f"{full_df['Region'].nunique():,}")
    col4.metric("Unique Segments", f"{full_df['Customer Segment'].nunique():,}")

    st.markdown("#### Processed Missing Values Summary")
    missing_df = (
        full_df.isna()
        .sum()
        .reset_index()
        .rename(columns={"index": "Column", 0: "Missing Values"})
        .sort_values("Missing Values", ascending=False)
    )
    st.dataframe(missing_df, use_container_width=True)

    st.markdown("#### Processed Column Data Types")
    dtypes_df = (
        full_df.dtypes.astype(str)
        .reset_index()
        .rename(columns={"index": "Column", 0: "Data Type"})
    )
    st.dataframe(dtypes_df, use_container_width=True)

    st.markdown("#### Processed Sample Records")
    st.dataframe(full_df.head(20), use_container_width=True)


def render_alerts_and_actions(
    anomaly_result: dict,
) -> None:
    st.subheader("Anomaly Alerts and Actions")

    for alert in anomaly_result["alerts"]:
        st.warning(alert)

    st.markdown("### Monthly Anomalies")
    monthly_anomalies = anomaly_result["monthly_anomalies"]
    if monthly_anomalies.empty:
        st.info("No monthly anomalies found for current filters.")
    else:
        st.dataframe(
            monthly_anomalies.assign(
                **{
                    "Order Date": lambda d: d["Order Date"].dt.strftime("%Y-%m"),
                    "Sales": lambda d: d["Sales"].round(2),
                    "Profit": lambda d: d["Profit"].round(2),
                    "Sales Z-Score": lambda d: d["Sales Z-Score"].round(2),
                }
            ),
            use_container_width=True,
        )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Negative-Profit Transactions")
        neg_orders = anomaly_result["negative_profit_orders"]
        if neg_orders.empty:
            st.info("No negative-profit transactions found.")
        else:
            st.dataframe(
                neg_orders[
                    [
                        "Order ID",
                        "Order Date",
                        "Category",
                        "Sub-Category",
                        "Region",
                        "Sales",
                        "Profit",
                    ]
                ].assign(
                    **{
                        "Order Date": lambda d: d["Order Date"].dt.strftime("%Y-%m-%d"),
                        "Sales": lambda d: d["Sales"].round(2),
                        "Profit": lambda d: d["Profit"].round(2),
                    }
                ),
                use_container_width=True,
            )

    with col2:
        st.markdown("### Low-Margin Categories")
        low_margin = anomaly_result["low_margin_categories"]
        if low_margin.empty:
            st.info("No categories below 5% margin under current filters.")
        else:
            st.dataframe(
                low_margin.assign(
                    **{
                        "Sales": lambda d: d["Sales"].round(2),
                        "Profit": lambda d: d["Profit"].round(2),
                        "Profit Margin (%)": lambda d: d["Profit Margin (%)"].round(2),
                    }
                ),
                use_container_width=True,
            )

    st.markdown("### Region Risk Table")
    region_perf = anomaly_result["region_performance"]
    st.dataframe(
        region_perf.assign(
            **{
                "Sales": lambda d: d["Sales"].round(2),
                "Profit": lambda d: d["Profit"].round(2),
                "Profit Margin (%)": lambda d: d["Profit Margin (%)"].round(2),
            }
        ),
        use_container_width=True,
    )


def render_data_quality(quality_report: dict, drift_report: dict) -> None:
    st.subheader("Data Quality and Drift Monitoring")
    score = quality_report.get("score", 0.0)
    critical_count = len(quality_report.get("critical_failures", []))
    warning_count = len(quality_report.get("warning_failures", []))
    drift_count = int(drift_report.get("flagged_count", 0))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Quality Score", f"{score:.2f}")
    col2.metric("Critical Failures", f"{critical_count}")
    col3.metric("Warning Failures", f"{warning_count}")
    col4.metric("Drift-Flagged Features", f"{drift_count}")

    if critical_count > 0:
        st.error(
            "Critical quality failures detected: "
            + ", ".join(quality_report.get("critical_failures", []))
        )
    elif warning_count > 0:
        st.warning(
            "Quality warnings detected: " + ", ".join(quality_report.get("warning_failures", []))
        )
    else:
        st.success("All configured quality checks passed.")

    st.markdown("### Quality Checks")
    checks_df = quality_report.get("checks", pd.DataFrame())
    if checks_df.empty:
        st.info("No quality checks available.")
    else:
        st.dataframe(checks_df, use_container_width=True)

    st.markdown("### Drift Table (Current Filter vs Baseline Dataset)")
    drift_df = drift_report.get("drift_table", pd.DataFrame())
    if drift_df.empty:
        st.info("No drift metrics available for current data.")
    else:
        st.dataframe(
            drift_df.assign(
                **{
                    "Current Mean": lambda d: d["Current Mean"].round(2),
                    "Baseline Mean": lambda d: d["Baseline Mean"].round(2),
                    "Mean Change (%)": lambda d: d["Mean Change (%)"].round(2),
                    "KS Statistic": lambda d: d["KS Statistic"].round(3),
                }
            ),
            use_container_width=True,
        )

    drift_alerts = drift_report.get("alerts", [])
    if drift_alerts:
        st.markdown("### Drift Alerts")
        for alert in drift_alerts:
            st.warning(alert)


def main() -> None:
    st.title("Sales Data Analysis Dashboard")
    st.caption(
        "A complete portfolio-style project covering data cleaning, EDA, visualization, "
        "forecasting, and data-driven business suggestions."
    )

    st.sidebar.header("Data Source")
    source_option = st.sidebar.radio(
        "Choose Source",
        options=["Default Dataset", "Upload File", "Database (SQL)"],
        index=0,
    )

    raw_uploaded_df = None
    if source_option == "Upload File":
        uploaded_file = st.sidebar.file_uploader(
            "Upload another sales dataset",
            type=["csv", "xlsx", "xls"],
            help=(
                "Supported formats: CSV, XLSX, XLS. The app auto-maps common sales columns "
                "like date, sales, profit, category, region, and segment."
            ),
        )
        if uploaded_file is None:
            st.info("Upload a file to continue with dashboard analysis.")
            st.stop()

        try:
            with st.spinner("Processing uploaded dataset..."):
                file_bytes = uploaded_file.getvalue()
                raw_uploaded_df = load_uploaded_raw_data(uploaded_file.name, file_bytes)
                df = load_uploaded_data(uploaded_file.name, file_bytes)
            dataset_label = uploaded_file.name
            st.sidebar.success(f"Using uploaded dataset: {uploaded_file.name}")
            st.sidebar.caption(
                f"Pipeline complete: raw ({len(raw_uploaded_df):,}) -> processed ({len(df):,}) rows."
            )
        except Exception as exc:
            st.sidebar.error(f"Could not process uploaded dataset: {exc}")
            st.stop()
    elif source_option == "Database (SQL)":
        st.sidebar.caption("Load data from a SQL database using a connection URL.")
        with st.sidebar.form("db_source_form", clear_on_submit=False):
            connection_url = st.text_input(
                "Connection URL",
                placeholder="sqlite:///data/sales.db",
                help="Use SQLAlchemy format. Example: postgresql+psycopg2://user:password@host:5432/dbname",
            )
            use_sql_query = st.checkbox("Use custom SQL query", value=True)
            sql_query = ""
            table_name = ""
            if use_sql_query:
                sql_query = st.text_area("SQL Query", value="SELECT * FROM sales")
            else:
                table_name = st.text_input("Table Name", value="sales")
            db_submitted = st.form_submit_button("Load Database Data")

        if db_submitted:
            try:
                with st.spinner("Fetching and processing database records..."):
                    raw_db_df, processed_db_df = load_database_data(
                        connection_url=connection_url,
                        table_name=table_name,
                        sql_query=sql_query,
                    )
                st.session_state["db_raw_df"] = raw_db_df
                st.session_state["db_processed_df"] = processed_db_df
                st.session_state["db_dataset_label"] = (
                    "Database Query Result" if use_sql_query else f"Database Table: {table_name}"
                )
                st.sidebar.success("Database dataset loaded successfully.")
            except Exception as exc:
                st.sidebar.error(f"Database load failed: {exc}")

        if "db_processed_df" not in st.session_state:
            st.info("Configure database details in the sidebar and click 'Load Database Data'.")
            st.stop()

        raw_uploaded_df = st.session_state["db_raw_df"]
        df = st.session_state["db_processed_df"]
        dataset_label = st.session_state.get("db_dataset_label", "Database Source")
        st.sidebar.caption(
            f"Pipeline complete: raw ({len(raw_uploaded_df):,}) -> processed ({len(df):,}) rows."
        )
    else:
        df = load_data()
        dataset_label = "data/amazon.csv (processed)"
        st.sidebar.caption("Using default dataset from `data/amazon.csv`.")

    if df.empty:
        st.error("No data available. Check the dataset in data/amazon.csv.")
        st.stop()

    st.sidebar.header("Forecast Settings")
    forecast_periods = st.sidebar.slider("Forecast Months", min_value=1, max_value=24, value=DEFAULT_FORECAST_PERIODS)
    holdout_months = st.sidebar.slider("Single Holdout Months", min_value=3, max_value=12, value=6)
    rolling_folds = st.sidebar.slider("Rolling Backtest Folds", min_value=3, max_value=12, value=6)
    rolling_horizon = st.sidebar.slider("Rolling Horizon (Months)", min_value=1, max_value=3, value=1)

    st.sidebar.header("Alert Thresholds")
    z_threshold = st.sidebar.slider("Sales Anomaly Z-Threshold", min_value=1.0, max_value=4.0, value=2.0, step=0.1)
    low_margin_threshold = st.sidebar.slider(
        "Low Margin Threshold (%)", min_value=0.0, max_value=20.0, value=5.0, step=0.5
    )
    negative_profit_cutoff = st.sidebar.number_input(
        "Negative Profit Cutoff",
        value=0.0,
        step=10.0,
        help="Transactions with profit below this value are flagged.",
    )

    st.sidebar.header("Quality Gates")
    min_rows_threshold = st.sidebar.slider("Minimum Rows", min_value=10, max_value=500, value=50, step=10)
    max_missing_pct_threshold = st.sidebar.slider(
        "Max Missing Ratio (%)", min_value=0.0, max_value=50.0, value=20.0, step=1.0
    )
    max_duplicate_order_id_pct_threshold = st.sidebar.slider(
        "Max Duplicate Order ID Ratio (%)", min_value=0.0, max_value=30.0, value=5.0, step=0.5
    )
    drift_mean_change_threshold = st.sidebar.slider(
        "Drift Mean Change Threshold (%)", min_value=5.0, max_value=100.0, value=25.0, step=1.0
    )
    drift_ks_threshold = st.sidebar.slider(
        "Drift KS Threshold", min_value=0.05, max_value=0.80, value=0.30, step=0.01
    )
    strict_quality_gate = st.sidebar.checkbox("Strict Quality Gate (for scheduled command)", value=False)
    fail_on_drift_gate = st.sidebar.checkbox("Fail on Drift (for scheduled command)", value=False)

    st.sidebar.header("Filters")

    min_date = df["Order Date"].min().date()
    max_date = df["Order Date"].max().date()

    date_range = st.sidebar.date_input(
        "Order Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date = pd.to_datetime(date_range[0])
        end_date = pd.to_datetime(date_range[1])
    else:
        start_date = pd.to_datetime(min_date)
        end_date = pd.to_datetime(max_date)

    regions = st.sidebar.multiselect(
        "Region",
        options=sorted(df["Region"].unique().tolist()),
        default=sorted(df["Region"].unique().tolist()),
    )
    categories = st.sidebar.multiselect(
        "Category",
        options=sorted(df["Category"].unique().tolist()),
        default=sorted(df["Category"].unique().tolist()),
    )
    segments = st.sidebar.multiselect(
        "Customer Segment",
        options=sorted(df["Customer Segment"].unique().tolist()),
        default=sorted(df["Customer Segment"].unique().tolist()),
    )

    filtered_df = apply_filters(df, regions, categories, segments, start_date, end_date)
    st.sidebar.write(f"Rows selected: **{len(filtered_df):,}**")

    if filtered_df.empty:
        st.warning("No records match the selected filters. Please broaden your selection.")
        st.stop()

    quality_report = assess_data_quality(
        filtered_df,
        thresholds=QualityThresholds(
            min_rows=min_rows_threshold,
            max_missing_pct=max_missing_pct_threshold / 100,
            max_duplicate_order_id_pct=max_duplicate_order_id_pct_threshold / 100,
        ),
    )
    drift_report = assess_data_drift(
        current_df=filtered_df,
        baseline_df=df,
        thresholds=DriftThresholds(
            mean_change_pct=drift_mean_change_threshold,
            ks_stat=drift_ks_threshold,
        ),
    )
    st.sidebar.caption(
        f"Quality score: {quality_report.get('score', 0):.2f} | Drift flags: {drift_report.get('flagged_count', 0)}"
    )

    anomaly_result = detect_business_anomalies(
        filtered_df,
        z_threshold=z_threshold,
        low_margin_threshold=low_margin_threshold,
        negative_profit_cutoff=negative_profit_cutoff,
    )
    forecast_result = forecast_sales(
        filtered_df,
        periods=forecast_periods,
        holdout=holdout_months,
        rolling_folds=rolling_folds,
        rolling_horizon=rolling_horizon,
    )

    kpis = compute_kpis(filtered_df)
    kpi_cols = st.columns(5)
    kpi_cols[0].metric("Total Sales", format_currency(kpis["total_sales"]))
    kpi_cols[1].metric("Total Profit", format_currency(kpis["total_profit"]))
    kpi_cols[2].metric("Total Orders", f"{kpis['total_orders']:,}")
    kpi_cols[3].metric("Avg Order Value", format_currency(kpis["avg_order_value"]))
    kpi_cols[4].metric("Profit Margin", f"{kpis['profit_margin']:.2f}%")

    tab_overview, tab_breakdown, tab_forecast, tab_alerts, tab_quality, tab_dataset, tab_report = st.tabs(
        [
            "Overview",
            "Category & Region Analysis",
            "Forecast & Suggestions",
            "Alerts & Actions",
            "Data Quality",
            "Dataset Info",
            "Executive Report",
        ]
    )

    with tab_overview:
        st.subheader("Monthly Trend and Correlations")
        trend_fig = create_monthly_trend_figure(filtered_df)
        st.plotly_chart(trend_fig, use_container_width=True, config={"displaylogo": False})

        col1, col2 = st.columns(2)
        with col1:
            scatter_fig = create_profit_vs_sales_figure(filtered_df)
            st.plotly_chart(scatter_fig, use_container_width=True, config={"displaylogo": False})
        with col2:
            heatmap_fig = create_correlation_heatmap_figure(filtered_df)
            st.plotly_chart(heatmap_fig, use_container_width=True, config={"displaylogo": False})

    with tab_breakdown:
        st.subheader("Sales Breakdown")
        col1, col2 = st.columns(2)
        with col1:
            category_fig = create_category_sales_figure(filtered_df)
            st.plotly_chart(category_fig, use_container_width=True, config={"displaylogo": False})
        with col2:
            sub_category_fig = create_sub_category_sales_figure(filtered_df)
            st.plotly_chart(
                sub_category_fig, use_container_width=True, config={"displaylogo": False}
            )

        region_fig = create_region_performance_figure(filtered_df)
        st.plotly_chart(region_fig, use_container_width=True, config={"displaylogo": False})

    with tab_forecast:
        st.subheader("Sales Prediction")
        historical_df = forecast_result["historical"]
        forecast_df = forecast_result["forecast"]
        metrics = forecast_result["model_metrics"]
        benchmark_df = forecast_result["model_benchmark"]
        rolling_benchmark_df = forecast_result.get("rolling_benchmark_summary", pd.DataFrame())
        rolling_details_df = forecast_result.get("rolling_backtest_details", pd.DataFrame())
        selected_model = forecast_result.get("selected_model", "Linear Trend")

        forecast_fig = render_forecast_chart(historical_df, forecast_df)
        st.plotly_chart(forecast_fig, use_container_width=True, config={"displaylogo": False})

        st.markdown("**Predicted Sales (Next Months)**")
        st.dataframe(
            forecast_df.assign(
                Month=lambda d: d["Month"].dt.strftime("%Y-%m"),
                **{"Predicted Sales": lambda d: d["Predicted Sales"].round(2)},
            ),
            use_container_width=True,
        )

        if metrics["rmse"] is not None and metrics["mape"] is not None:
            col1, col2 = st.columns(2)
            col1.metric("Validation RMSE", f"{metrics['rmse']:.2f}")
            col2.metric("Validation MAPE", f"{metrics['mape']:.2f}%")
            st.caption(f"Metric source: `{metrics.get('metric_source', 'unknown')}`")
        else:
            st.info("Not enough monthly history for holdout validation metrics.")

        st.markdown("### Single-Holdout Benchmark")
        st.caption(f"Selected model: **{selected_model}**")
        st.dataframe(
            benchmark_df.assign(
                **{
                    "RMSE": lambda d: d["RMSE"].round(2),
                    "MAPE": lambda d: d["MAPE"].round(2),
                }
            ),
            use_container_width=True,
        )

        st.markdown("### Rolling-Origin Benchmark")
        st.dataframe(
            rolling_benchmark_df.assign(
                **{
                    "RMSE": lambda d: d["RMSE"].round(2),
                    "MAPE": lambda d: d["MAPE"].round(2),
                }
            ),
            use_container_width=True,
        )

        with st.expander("Rolling Backtest Fold Details", expanded=False):
            if rolling_details_df.empty:
                st.info("No rolling fold details available for current settings.")
            else:
                st.dataframe(
                    rolling_details_df.assign(
                        **{
                            "Train End": lambda d: pd.to_datetime(d["Train End"]).dt.strftime(
                                "%Y-%m"
                            ),
                            "Test Start": lambda d: pd.to_datetime(d["Test Start"]).dt.strftime(
                                "%Y-%m"
                            ),
                            "Test End": lambda d: pd.to_datetime(d["Test End"]).dt.strftime(
                                "%Y-%m"
                            ),
                            "RMSE": lambda d: d["RMSE"].round(2),
                            "MAPE": lambda d: d["MAPE"].round(2),
                        }
                    ),
                    use_container_width=True,
                )

        st.markdown("### Business Suggestions")
        for suggestion in generate_business_suggestions(filtered_df, forecast_df):
            st.write(f"- {suggestion}")

    with tab_alerts:
        render_alerts_and_actions(anomaly_result)

    with tab_quality:
        render_data_quality(quality_report, drift_report)

    with tab_dataset:
        render_dataset_info(df, filtered_df, dataset_label, raw_uploaded_df)

    with tab_report:
        st.subheader("Executive Report Download")
        filter_summary = {
            "start_date": pd.to_datetime(start_date).date(),
            "end_date": pd.to_datetime(end_date).date(),
        }
        report_df = build_executive_report_dataframe(
            dataset_label=dataset_label,
            filtered_df=filtered_df,
            forecast_result=forecast_result,
            anomaly_result=anomaly_result,
            filter_summary=filter_summary,
            quality_report=quality_report,
            drift_report=drift_report,
        )
        report_md = build_executive_report_markdown(
            dataset_label=dataset_label,
            filtered_df=filtered_df,
            forecast_result=forecast_result,
            anomaly_result=anomaly_result,
            filter_summary=filter_summary,
            quality_report=quality_report,
            drift_report=drift_report,
        )

        st.dataframe(report_df.head(30), use_container_width=True)
        st.download_button(
            label="Download Executive Report (CSV)",
            data=report_df.to_csv(index=False).encode("utf-8"),
            file_name="executive_sales_report.csv",
            mime="text/csv",
        )
        st.download_button(
            label="Download Executive Report (Markdown)",
            data=report_md.encode("utf-8"),
            file_name="executive_sales_report.md",
            mime="text/markdown",
        )

        st.markdown("### Save Snapshot to Workspace")
        snapshot_dir = st.text_input(
            "Snapshot Output Directory",
            value="outputs/reports",
            help="Reports are saved with timestamped filenames plus latest copies.",
        )
        if st.button("Save Snapshot Report Files"):
            saved_paths = save_executive_report_bundle(
                report_df=report_df,
                report_md=report_md,
                output_dir=snapshot_dir,
            )
            st.success("Snapshot report files saved.")
            st.code(
                "\n".join(
                    [
                        f"csv: {saved_paths['csv']}",
                        f"markdown: {saved_paths['markdown']}",
                        f"latest_csv: {saved_paths['latest_csv']}",
                        f"latest_markdown: {saved_paths['latest_markdown']}",
                    ]
                )
            )

        st.markdown("### Scheduled Job Command")
        gate_flags = []
        if strict_quality_gate:
            gate_flags.append("--strict-quality")
        if fail_on_drift_gate:
            gate_flags.append("--fail-on-drift")
        gate_flag_str = " ".join(gate_flags)
        if source_option == "Database (SQL)":
            schedule_cmd = (
                "python -m src.jobs --source database --connection-url \"<your-connection-url>\" "
                "--sql-query \"SELECT * FROM sales\" "
                f"--output-dir \"{snapshot_dir}\" --forecast-periods {forecast_periods} "
                f"--holdout-months {holdout_months} --rolling-folds {rolling_folds} "
                f"--rolling-horizon {rolling_horizon} --z-threshold {z_threshold} "
                f"--low-margin-threshold {low_margin_threshold} "
                f"--negative-profit-cutoff {negative_profit_cutoff} "
                f"--min-rows {min_rows_threshold} --max-missing-pct {max_missing_pct_threshold / 100:.4f} "
                f"--max-duplicate-order-id-pct {max_duplicate_order_id_pct_threshold / 100:.4f} "
                f"--drift-mean-change-threshold {drift_mean_change_threshold} "
                f"--drift-ks-threshold {drift_ks_threshold:.4f} {gate_flag_str}".strip()
            )
        else:
            schedule_cmd = (
                "python -m src.jobs --source default "
                f"--output-dir \"{snapshot_dir}\" --forecast-periods {forecast_periods} "
                f"--holdout-months {holdout_months} --rolling-folds {rolling_folds} "
                f"--rolling-horizon {rolling_horizon} --z-threshold {z_threshold} "
                f"--low-margin-threshold {low_margin_threshold} "
                f"--negative-profit-cutoff {negative_profit_cutoff} "
                f"--min-rows {min_rows_threshold} --max-missing-pct {max_missing_pct_threshold / 100:.4f} "
                f"--max-duplicate-order-id-pct {max_duplicate_order_id_pct_threshold / 100:.4f} "
                f"--drift-mean-change-threshold {drift_mean_change_threshold} "
                f"--drift-ks-threshold {drift_ks_threshold:.4f} "
                f"--start-date {pd.to_datetime(start_date).date()} --end-date {pd.to_datetime(end_date).date()} "
                f"--regions \"{','.join(regions)}\" --categories \"{','.join(categories)}\" "
                f"--segments \"{','.join(segments)}\" {gate_flag_str}".strip()
            )
        st.code(schedule_cmd)


if __name__ == "__main__":
    main()
