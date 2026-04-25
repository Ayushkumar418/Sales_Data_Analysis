from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class QualityThresholds:
    min_rows: int = 50
    max_missing_pct: float = 0.2
    max_duplicate_order_id_pct: float = 0.05


@dataclass
class DriftThresholds:
    mean_change_pct: float = 25.0
    ks_stat: float = 0.30


def _safe_pct(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def _ks_statistic(sample_a: pd.Series, sample_b: pd.Series) -> float:
    a = sample_a.dropna().to_numpy(dtype=float)
    b = sample_b.dropna().to_numpy(dtype=float)
    if len(a) == 0 or len(b) == 0:
        return 0.0

    a_sorted = np.sort(a)
    b_sorted = np.sort(b)
    x = np.sort(np.unique(np.concatenate([a_sorted, b_sorted])))
    cdf_a = np.searchsorted(a_sorted, x, side="right") / len(a_sorted)
    cdf_b = np.searchsorted(b_sorted, x, side="right") / len(b_sorted)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def assess_data_quality(
    df: pd.DataFrame,
    thresholds: QualityThresholds | None = None,
    required_columns: Iterable[str] | None = None,
) -> dict:
    thresholds = thresholds or QualityThresholds()
    required_columns = list(required_columns or ["Order Date", "Sales", "Profit", "Category", "Region"])
    checks: list[dict] = []

    total_rows = len(df)
    total_columns = df.shape[1] if not df.empty else 0

    missing_required_columns = [col for col in required_columns if col not in df.columns]
    checks.append(
        {
            "Check": "Required Columns Present",
            "Status": "FAIL" if missing_required_columns else "PASS",
            "Severity": "critical",
            "Value": ", ".join(missing_required_columns) if missing_required_columns else "All present",
            "Threshold": "No missing required columns",
            "Message": "Required schema check for downstream analytics.",
        }
    )

    checks.append(
        {
            "Check": "Minimum Row Count",
            "Status": "PASS" if total_rows >= thresholds.min_rows else "FAIL",
            "Severity": "critical",
            "Value": total_rows,
            "Threshold": f">= {thresholds.min_rows}",
            "Message": "Ensures enough volume for stable KPI and forecasting outputs.",
        }
    )

    if total_rows > 0 and "Order ID" in df.columns:
        duplicate_order_id_pct = _safe_pct(df["Order ID"].duplicated().sum(), total_rows)
    else:
        duplicate_order_id_pct = 0.0
    checks.append(
        {
            "Check": "Duplicate Order ID Ratio",
            "Status": "PASS"
            if duplicate_order_id_pct <= thresholds.max_duplicate_order_id_pct
            else "FAIL",
            "Severity": "warning",
            "Value": round(duplicate_order_id_pct * 100, 2),
            "Threshold": f"<= {round(thresholds.max_duplicate_order_id_pct * 100, 2)}%",
            "Message": "High duplicates can inflate aggregate results.",
        }
    )

    for col in required_columns:
        if col in df.columns and total_rows > 0:
            missing_pct = _safe_pct(df[col].isna().sum(), total_rows)
            checks.append(
                {
                    "Check": f"Missing Ratio ({col})",
                    "Status": "PASS" if missing_pct <= thresholds.max_missing_pct else "FAIL",
                    "Severity": "critical" if col in {"Order Date", "Sales", "Profit"} else "warning",
                    "Value": round(missing_pct * 100, 2),
                    "Threshold": f"<= {round(thresholds.max_missing_pct * 100, 2)}%",
                    "Message": f"Missing values in {col} can degrade analysis quality.",
                }
            )

    if "Sales" in df.columns and total_rows > 0:
        negative_sales_ratio = _safe_pct((df["Sales"] < 0).sum(), total_rows)
        checks.append(
            {
                "Check": "Negative Sales Ratio",
                "Status": "PASS" if negative_sales_ratio == 0 else "FAIL",
                "Severity": "critical",
                "Value": round(negative_sales_ratio * 100, 2),
                "Threshold": "0%",
                "Message": "Negative sales generally indicate data corruption for this dashboard.",
            }
        )

    if "Order Date" in df.columns and total_rows > 0:
        invalid_date_ratio = _safe_pct(pd.to_datetime(df["Order Date"], errors="coerce").isna().sum(), total_rows)
        checks.append(
            {
                "Check": "Invalid Date Ratio",
                "Status": "PASS" if invalid_date_ratio == 0 else "FAIL",
                "Severity": "critical",
                "Value": round(invalid_date_ratio * 100, 2),
                "Threshold": "0%",
                "Message": "Date integrity is required for time-series and trends.",
            }
        )

    checks_df = pd.DataFrame(checks)
    critical_failures = checks_df[(checks_df["Status"] == "FAIL") & (checks_df["Severity"] == "critical")]
    warning_failures = checks_df[(checks_df["Status"] == "FAIL") & (checks_df["Severity"] == "warning")]

    score = 100.0
    score -= 20 * len(critical_failures)
    score -= 8 * len(warning_failures)
    score = max(score, 0.0)

    return {
        "score": round(score, 2),
        "checks": checks_df,
        "critical_failures": critical_failures["Check"].tolist(),
        "warning_failures": warning_failures["Check"].tolist(),
        "summary": {"rows": total_rows, "columns": total_columns},
    }


def assess_data_drift(
    current_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    thresholds: DriftThresholds | None = None,
    numeric_columns: Iterable[str] | None = None,
) -> dict:
    thresholds = thresholds or DriftThresholds()
    numeric_columns = list(
        numeric_columns or ["Sales", "Profit", "Unit Price", "Quantity", "Discount Percentage", "Rating"]
    )

    rows: list[dict] = []
    for col in numeric_columns:
        if col not in current_df.columns or col not in baseline_df.columns:
            continue
        cur = pd.to_numeric(current_df[col], errors="coerce")
        base = pd.to_numeric(baseline_df[col], errors="coerce")
        if cur.dropna().empty or base.dropna().empty:
            continue

        cur_mean = float(cur.mean())
        base_mean = float(base.mean())
        mean_change_pct = (
            0.0 if base_mean == 0 else float(((cur_mean - base_mean) / abs(base_mean)) * 100)
        )
        ks_value = _ks_statistic(cur, base)
        drift_flag = abs(mean_change_pct) >= thresholds.mean_change_pct or ks_value >= thresholds.ks_stat

        rows.append(
            {
                "Feature": col,
                "Current Mean": cur_mean,
                "Baseline Mean": base_mean,
                "Mean Change (%)": mean_change_pct,
                "KS Statistic": ks_value,
                "Drift Flag": drift_flag,
            }
        )

    drift_df = pd.DataFrame(rows)
    if drift_df.empty:
        return {"drift_table": drift_df, "alerts": [], "flagged_count": 0}

    drift_df = drift_df.sort_values(["Drift Flag", "KS Statistic"], ascending=[False, False]).reset_index(drop=True)
    flagged = drift_df[drift_df["Drift Flag"]]
    alerts: list[str] = []
    for _, row in flagged.iterrows():
        alerts.append(
            f"{row['Feature']} drift flagged: mean change {row['Mean Change (%)']:.2f}% and KS {row['KS Statistic']:.3f}."
        )

    return {"drift_table": drift_df, "alerts": alerts, "flagged_count": int(len(flagged))}

