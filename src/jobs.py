import argparse
from pathlib import Path

import pandas as pd

from .anomalies import detect_business_anomalies
from .data_processing import ensure_processed_dataset, prepare_sales_dataset
from .forecasting import forecast_sales
from .quality import DriftThresholds, QualityThresholds, assess_data_drift, assess_data_quality
from .reporting import (
    build_executive_report_dataframe,
    build_executive_report_markdown,
    save_executive_report_bundle,
)


def _parse_list(raw_value: str) -> list[str]:
    if not raw_value.strip():
        return []
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def _load_dataset(source: str, connection_url: str, table_name: str, sql_query: str) -> tuple[pd.DataFrame, str]:
    if source == "default":
        return ensure_processed_dataset(force_rebuild=False), "default_processed_dataset"

    if source == "database":
        try:
            from sqlalchemy import create_engine
        except ImportError as exc:
            raise ValueError("sqlalchemy is required for database source.") from exc

        if not connection_url.strip():
            raise ValueError("--connection-url is required for source=database")
        if not table_name.strip() and not sql_query.strip():
            raise ValueError("Provide --table-name or --sql-query for source=database")

        engine = create_engine(connection_url)
        with engine.connect() as conn:
            if sql_query.strip():
                raw_df = pd.read_sql_query(sql_query, conn)
                label = "database_query_result"
            else:
                raw_df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
                label = f"database_table_{table_name}"

        if raw_df.empty:
            raise ValueError("Database query returned no rows.")
        return prepare_sales_dataset(raw_df), label

    raise ValueError(f"Unsupported source: {source}")


def _apply_filters(
    df: pd.DataFrame,
    start_date: str,
    end_date: str,
    regions: list[str],
    categories: list[str],
    segments: list[str],
) -> pd.DataFrame:
    out = df.copy()
    if start_date:
        out = out[out["Order Date"] >= pd.to_datetime(start_date)]
    if end_date:
        out = out[out["Order Date"] <= pd.to_datetime(end_date)]
    if regions:
        out = out[out["Region"].isin(regions)]
    if categories:
        out = out[out["Category"].isin(categories)]
    if segments:
        out = out[out["Customer Segment"].isin(segments)]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run scheduled refresh and generate executive sales reports."
    )
    parser.add_argument("--source", choices=["default", "database"], default="default")
    parser.add_argument("--connection-url", default="")
    parser.add_argument("--table-name", default="")
    parser.add_argument("--sql-query", default="")
    parser.add_argument("--output-dir", default="outputs/reports")
    parser.add_argument("--forecast-periods", type=int, default=6)
    parser.add_argument("--holdout-months", type=int, default=6)
    parser.add_argument("--rolling-folds", type=int, default=6)
    parser.add_argument("--rolling-horizon", type=int, default=1)
    parser.add_argument("--z-threshold", type=float, default=2.0)
    parser.add_argument("--low-margin-threshold", type=float, default=5.0)
    parser.add_argument("--negative-profit-cutoff", type=float, default=0.0)
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--regions", default="")
    parser.add_argument("--categories", default="")
    parser.add_argument("--segments", default="")
    parser.add_argument("--base-name", default="executive_sales_report")
    parser.add_argument("--strict-quality", action="store_true")
    parser.add_argument("--fail-on-drift", action="store_true")
    parser.add_argument("--min-rows", type=int, default=50)
    parser.add_argument("--max-missing-pct", type=float, default=0.20)
    parser.add_argument("--max-duplicate-order-id-pct", type=float, default=0.05)
    parser.add_argument("--drift-mean-change-threshold", type=float, default=25.0)
    parser.add_argument("--drift-ks-threshold", type=float, default=0.30)
    args = parser.parse_args()

    df, dataset_label = _load_dataset(
        source=args.source,
        connection_url=args.connection_url,
        table_name=args.table_name,
        sql_query=args.sql_query,
    )

    filtered_df = _apply_filters(
        df,
        start_date=args.start_date,
        end_date=args.end_date,
        regions=_parse_list(args.regions),
        categories=_parse_list(args.categories),
        segments=_parse_list(args.segments),
    )
    if filtered_df.empty:
        raise ValueError("Filters removed all rows. No report generated.")

    quality_report = assess_data_quality(
        filtered_df,
        thresholds=QualityThresholds(
            min_rows=args.min_rows,
            max_missing_pct=args.max_missing_pct,
            max_duplicate_order_id_pct=args.max_duplicate_order_id_pct,
        ),
    )
    drift_report = assess_data_drift(
        filtered_df,
        baseline_df=df,
        thresholds=DriftThresholds(
            mean_change_pct=args.drift_mean_change_threshold,
            ks_stat=args.drift_ks_threshold,
        ),
    )

    print(f"Quality Score: {quality_report['score']}")
    print(f"Quality Critical Failures: {len(quality_report['critical_failures'])}")
    print(f"Drift Flagged Features: {drift_report['flagged_count']}")

    if args.strict_quality and quality_report["critical_failures"]:
        raise ValueError(
            "Strict quality gate failed with critical checks: "
            + ", ".join(quality_report["critical_failures"])
        )
    if args.fail_on_drift and drift_report["flagged_count"] > 0:
        raise ValueError(
            "Drift gate failed; flagged features: " + ", ".join(drift_report.get("alerts", []))
        )

    forecast_result = forecast_sales(
        filtered_df,
        periods=args.forecast_periods,
        holdout=args.holdout_months,
        rolling_folds=args.rolling_folds,
        rolling_horizon=args.rolling_horizon,
    )
    anomaly_result = detect_business_anomalies(
        filtered_df,
        z_threshold=args.z_threshold,
        low_margin_threshold=args.low_margin_threshold,
        negative_profit_cutoff=args.negative_profit_cutoff,
    )
    filter_summary = {
        "start_date": str(filtered_df["Order Date"].min().date()),
        "end_date": str(filtered_df["Order Date"].max().date()),
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
    paths = save_executive_report_bundle(
        report_df=report_df,
        report_md=report_md,
        output_dir=Path(args.output_dir),
        base_name=args.base_name,
    )

    print("Scheduled report generation completed.")
    for key, value in paths.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
