from datetime import datetime
from pathlib import Path

import pandas as pd

from .insights import compute_kpis


def build_executive_report_dataframe(
    dataset_label: str,
    filtered_df: pd.DataFrame,
    forecast_result: dict,
    anomaly_result: dict,
    filter_summary: dict,
    quality_report: dict | None = None,
    drift_report: dict | None = None,
) -> pd.DataFrame:
    kpis = compute_kpis(filtered_df)
    model_metrics = forecast_result.get("model_metrics", {})
    selected_model = forecast_result.get("selected_model", "N/A")
    forecast_df = forecast_result.get("forecast", pd.DataFrame(columns=["Month", "Predicted Sales"]))

    rows: list[dict] = [
        {"Section": "Meta", "Metric": "Dataset", "Value": dataset_label},
        {"Section": "Meta", "Metric": "Rows (Filtered)", "Value": f"{len(filtered_df):,}"},
        {"Section": "Meta", "Metric": "Date Start", "Value": str(filter_summary.get("start_date", ""))},
        {"Section": "Meta", "Metric": "Date End", "Value": str(filter_summary.get("end_date", ""))},
        {"Section": "KPI", "Metric": "Total Sales", "Value": f"{kpis['total_sales']:.2f}"},
        {"Section": "KPI", "Metric": "Total Profit", "Value": f"{kpis['total_profit']:.2f}"},
        {"Section": "KPI", "Metric": "Total Orders", "Value": f"{kpis['total_orders']}"},
        {"Section": "KPI", "Metric": "Average Order Value", "Value": f"{kpis['avg_order_value']:.2f}"},
        {"Section": "KPI", "Metric": "Profit Margin (%)", "Value": f"{kpis['profit_margin']:.2f}"},
        {"Section": "Forecast", "Metric": "Selected Model", "Value": selected_model},
        {
            "Section": "Forecast",
            "Metric": "Validation RMSE",
            "Value": f"{model_metrics.get('rmse')}" if model_metrics.get("rmse") is not None else "N/A",
        },
        {
            "Section": "Forecast",
            "Metric": "Validation MAPE (%)",
            "Value": f"{model_metrics.get('mape')}" if model_metrics.get("mape") is not None else "N/A",
        },
        {"Section": "Forecast", "Metric": "Metric Source", "Value": model_metrics.get("metric_source", "N/A")},
        {"Section": "Alerts", "Metric": "Alert Count", "Value": f"{len(anomaly_result.get('alerts', []))}"},
    ]

    if quality_report is not None:
        rows.extend(
            [
                {"Section": "Quality", "Metric": "Quality Score", "Value": f"{quality_report.get('score', 'N/A')}"},
                {
                    "Section": "Quality",
                    "Metric": "Critical Failures",
                    "Value": f"{len(quality_report.get('critical_failures', []))}",
                },
                {
                    "Section": "Quality",
                    "Metric": "Warning Failures",
                    "Value": f"{len(quality_report.get('warning_failures', []))}",
                },
            ]
        )

    if drift_report is not None:
        rows.append(
            {
                "Section": "Drift",
                "Metric": "Flagged Features",
                "Value": f"{drift_report.get('flagged_count', 0)}",
            }
        )

    for idx, alert in enumerate(anomaly_result.get("alerts", []), start=1):
        rows.append({"Section": "Alerts", "Metric": f"Alert {idx}", "Value": alert})

    if not forecast_df.empty:
        for _, row in forecast_df.iterrows():
            rows.append(
                {
                    "Section": "Forecast Next Months",
                    "Metric": pd.to_datetime(row["Month"]).strftime("%Y-%m"),
                    "Value": f"{float(row['Predicted Sales']):.2f}",
                }
            )

    return pd.DataFrame(rows)


def build_executive_report_markdown(
    dataset_label: str,
    filtered_df: pd.DataFrame,
    forecast_result: dict,
    anomaly_result: dict,
    filter_summary: dict,
    quality_report: dict | None = None,
    drift_report: dict | None = None,
) -> str:
    kpis = compute_kpis(filtered_df)
    selected_model = forecast_result.get("selected_model", "N/A")
    model_metrics = forecast_result.get("model_metrics", {})
    forecast_df = forecast_result.get("forecast", pd.DataFrame(columns=["Month", "Predicted Sales"]))

    lines = [
        "# Executive Sales Report",
        "",
        "## Scope",
        f"- Dataset: {dataset_label}",
        f"- Filtered rows: {len(filtered_df):,}",
        f"- Date range: {filter_summary.get('start_date', '')} to {filter_summary.get('end_date', '')}",
        "",
        "## KPI Snapshot",
        f"- Total Sales: {kpis['total_sales']:.2f}",
        f"- Total Profit: {kpis['total_profit']:.2f}",
        f"- Total Orders: {kpis['total_orders']}",
        f"- Average Order Value: {kpis['avg_order_value']:.2f}",
        f"- Profit Margin: {kpis['profit_margin']:.2f}%",
        "",
        "## Forecast",
        f"- Selected Model: {selected_model}",
        f"- Validation RMSE: {model_metrics.get('rmse', 'N/A')}",
        f"- Validation MAPE (%): {model_metrics.get('mape', 'N/A')}",
        f"- Metric Source: {model_metrics.get('metric_source', 'N/A')}",
        "",
        "## Forecast Next Months",
    ]

    if forecast_df.empty:
        lines.append("- No forecast rows available.")
    else:
        for _, row in forecast_df.iterrows():
            lines.append(
                f"- {pd.to_datetime(row['Month']).strftime('%Y-%m')}: {float(row['Predicted Sales']):.2f}"
            )

    lines.extend(["", "## Alerts"])
    alerts = anomaly_result.get("alerts", [])
    if not alerts:
        lines.append("- No alerts generated.")
    else:
        for alert in alerts:
            lines.append(f"- {alert}")

    if quality_report is not None:
        lines.extend(
            [
                "",
                "## Data Quality",
                f"- Quality Score: {quality_report.get('score', 'N/A')}",
                f"- Critical Failures: {len(quality_report.get('critical_failures', []))}",
                f"- Warning Failures: {len(quality_report.get('warning_failures', []))}",
            ]
        )

    if drift_report is not None:
        lines.extend(
            [
                "",
                "## Data Drift",
                f"- Drift-Flagged Features: {drift_report.get('flagged_count', 0)}",
            ]
        )
        for alert in drift_report.get("alerts", []):
            lines.append(f"- {alert}")

    return "\n".join(lines)


def save_executive_report_bundle(
    report_df: pd.DataFrame,
    report_md: str,
    output_dir: str | Path,
    base_name: str = "executive_sales_report",
    timestamp: str | None = None,
) -> dict:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_stamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path = out_dir / f"{base_name}_{run_stamp}.csv"
    md_path = out_dir / f"{base_name}_{run_stamp}.md"
    latest_csv_path = out_dir / f"{base_name}_latest.csv"
    latest_md_path = out_dir / f"{base_name}_latest.md"

    report_df.to_csv(csv_path, index=False)
    report_df.to_csv(latest_csv_path, index=False)
    md_path.write_text(report_md, encoding="utf-8")
    latest_md_path.write_text(report_md, encoding="utf-8")

    return {
        "csv": str(csv_path),
        "markdown": str(md_path),
        "latest_csv": str(latest_csv_path),
        "latest_markdown": str(latest_md_path),
    }
