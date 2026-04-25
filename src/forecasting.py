import numpy as np
import pandas as pd


LINEAR_MODEL = "Linear Trend"
SEASONAL_NAIVE_MODEL = "Seasonal Naive"
SES_MODEL = "Exponential Smoothing"
AVAILABLE_MODELS = [LINEAR_MODEL, SEASONAL_NAIVE_MODEL, SES_MODEL]


def build_monthly_sales_series(df: pd.DataFrame) -> pd.Series:
    monthly = df.groupby(pd.Grouper(key="Order Date", freq="MS"))["Sales"].sum().sort_index()
    if monthly.empty:
        return pd.Series(dtype=float)

    # Fill missing months so model training sees a continuous timeline.
    full_index = pd.date_range(monthly.index.min(), monthly.index.max(), freq="MS")
    monthly = monthly.reindex(full_index, fill_value=0.0)
    monthly.name = "Sales"
    return monthly


def _safe_rmse_mape(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    denominator = np.where(y_true == 0, 1, y_true)
    mape = float(np.mean(np.abs((y_true - y_pred) / denominator)) * 100)
    return rmse, mape


def _linear_coefficients(series: pd.Series) -> tuple[float, float]:
    if len(series) < 2:
        value = float(series.iloc[0]) if len(series) == 1 else 0.0
        return 0.0, value
    x = np.arange(len(series))
    y = series.to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def _predict_linear(series: pd.Series, periods: int) -> np.ndarray:
    slope, intercept = _linear_coefficients(series)
    x_future = np.arange(len(series), len(series) + periods)
    return np.clip(intercept + slope * x_future, a_min=0, a_max=None)


def _fit_linear(series: pd.Series) -> np.ndarray:
    slope, intercept = _linear_coefficients(series)
    x_hist = np.arange(len(series))
    return np.clip(intercept + slope * x_hist, a_min=0, a_max=None)


def _predict_seasonal_naive(series: pd.Series, periods: int, season_length: int = 12) -> np.ndarray:
    if len(series) == 0:
        return np.zeros(periods)

    if len(series) >= season_length:
        recent = series.iloc[-season_length:].to_numpy(dtype=float)
        repeated = np.array([recent[idx % season_length] for idx in range(periods)], dtype=float)
        return np.clip(repeated, a_min=0, a_max=None)

    last_value = float(series.iloc[-1])
    return np.repeat(max(last_value, 0.0), periods)


def _fit_seasonal_naive(series: pd.Series, season_length: int = 12) -> np.ndarray:
    if len(series) == 0:
        return np.array([], dtype=float)
    fitted = series.shift(season_length)
    if fitted.isna().all():
        fitted = series.shift(1)
    fitted = fitted.fillna(series.expanding().mean())
    return np.clip(fitted.to_numpy(dtype=float), a_min=0, a_max=None)


def _ses_fit_with_alpha(series: pd.Series, alpha: float) -> tuple[np.ndarray, float]:
    values = series.to_numpy(dtype=float)
    if len(values) == 0:
        return np.array([], dtype=float), 0.0

    level = float(values[0])
    fitted = [level]
    for idx in range(1, len(values)):
        fitted.append(level)
        level = alpha * float(values[idx]) + (1 - alpha) * level
    return np.array(fitted, dtype=float), level


def _fit_ses(series: pd.Series) -> tuple[np.ndarray, float, float]:
    if len(series) == 0:
        return np.array([], dtype=float), 0.0, 0.3
    if len(series) == 1:
        value = float(series.iloc[0])
        return np.array([value], dtype=float), value, 0.3

    best_alpha = 0.3
    best_rmse = float("inf")
    best_fitted = np.array([], dtype=float)
    best_level = float(series.iloc[-1])
    for alpha in np.arange(0.1, 1.0, 0.1):
        fitted, level = _ses_fit_with_alpha(series, alpha=float(alpha))
        rmse, _ = _safe_rmse_mape(series.to_numpy(dtype=float)[1:], fitted[1:])
        if rmse < best_rmse:
            best_rmse = rmse
            best_alpha = float(alpha)
            best_fitted = fitted
            best_level = level

    return np.clip(best_fitted, a_min=0, a_max=None), max(best_level, 0.0), best_alpha


def _predict_ses(series: pd.Series, periods: int) -> np.ndarray:
    _, level, _ = _fit_ses(series)
    return np.repeat(max(level, 0.0), periods)


def _fit_model(series: pd.Series, model_name: str) -> np.ndarray:
    if model_name == LINEAR_MODEL:
        return _fit_linear(series)
    if model_name == SEASONAL_NAIVE_MODEL:
        return _fit_seasonal_naive(series)
    if model_name == SES_MODEL:
        fitted, _, _ = _fit_ses(series)
        return fitted
    raise ValueError(f"Unsupported model: {model_name}")


def _predict_model(series: pd.Series, model_name: str, periods: int) -> np.ndarray:
    if model_name == LINEAR_MODEL:
        return _predict_linear(series, periods)
    if model_name == SEASONAL_NAIVE_MODEL:
        return _predict_seasonal_naive(series, periods)
    if model_name == SES_MODEL:
        return _predict_ses(series, periods)
    raise ValueError(f"Unsupported model: {model_name}")


def benchmark_models(monthly_sales: pd.Series, holdout: int = 6) -> pd.DataFrame:
    rows: list[dict] = []
    for model_name in AVAILABLE_MODELS:
        if len(monthly_sales) < holdout + 3:
            rows.append(
                {"Model": model_name, "RMSE": None, "MAPE": None, "Holdout Months": 0}
            )
            continue

        train = monthly_sales.iloc[:-holdout]
        test = monthly_sales.iloc[-holdout:]
        preds = _predict_model(train, model_name=model_name, periods=len(test))
        rmse, mape = _safe_rmse_mape(test.to_numpy(dtype=float), preds)
        rows.append(
            {
                "Model": model_name,
                "RMSE": rmse,
                "MAPE": mape,
                "Holdout Months": holdout,
            }
        )

    return pd.DataFrame(rows)


def rolling_origin_backtest(
    monthly_sales: pd.Series,
    model_name: str,
    folds: int = 6,
    horizon: int = 1,
    min_train_points: int = 12,
) -> pd.DataFrame:
    series = monthly_sales.sort_index()
    n_points = len(series)

    if n_points < min_train_points + horizon + 1:
        return pd.DataFrame(
            columns=[
                "Model",
                "Fold",
                "Train End",
                "Test Start",
                "Test End",
                "Horizon",
                "RMSE",
                "MAPE",
            ]
        )

    max_folds_possible = max(0, (n_points - min_train_points) // horizon)
    folds_to_run = min(folds, max_folds_possible)
    if folds_to_run <= 0:
        return pd.DataFrame(
            columns=[
                "Model",
                "Fold",
                "Train End",
                "Test Start",
                "Test End",
                "Horizon",
                "RMSE",
                "MAPE",
            ]
        )

    first_test_start = n_points - folds_to_run * horizon
    rows: list[dict] = []
    fold_no = 1

    for test_start in range(first_test_start, n_points, horizon):
        test_end = min(test_start + horizon, n_points)
        if test_start < min_train_points:
            continue

        train = series.iloc[:test_start]
        test = series.iloc[test_start:test_end]
        preds = _predict_model(train, model_name=model_name, periods=len(test))
        rmse, mape = _safe_rmse_mape(test.to_numpy(dtype=float), preds)

        rows.append(
            {
                "Model": model_name,
                "Fold": fold_no,
                "Train End": train.index.max(),
                "Test Start": test.index.min(),
                "Test End": test.index.max(),
                "Horizon": len(test),
                "RMSE": rmse,
                "MAPE": mape,
            }
        )
        fold_no += 1

    return pd.DataFrame(rows)


def rolling_backtest_models(
    monthly_sales: pd.Series,
    folds: int = 6,
    horizon: int = 1,
    min_train_points: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict] = []
    detail_frames: list[pd.DataFrame] = []

    for model_name in AVAILABLE_MODELS:
        details = rolling_origin_backtest(
            monthly_sales,
            model_name=model_name,
            folds=folds,
            horizon=horizon,
            min_train_points=min_train_points,
        )
        detail_frames.append(details)

        if details.empty:
            summary_rows.append(
                {
                    "Model": model_name,
                    "RMSE": None,
                    "MAPE": None,
                    "Folds": 0,
                    "Horizon": horizon,
                }
            )
        else:
            summary_rows.append(
                {
                    "Model": model_name,
                    "RMSE": float(details["RMSE"].mean()),
                    "MAPE": float(details["MAPE"].mean()),
                    "Folds": int(details["Fold"].nunique()),
                    "Horizon": horizon,
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    details_df = (
        pd.concat(detail_frames, ignore_index=True)
        if detail_frames
        else pd.DataFrame(
            columns=[
                "Model",
                "Fold",
                "Train End",
                "Test Start",
                "Test End",
                "Horizon",
                "RMSE",
                "MAPE",
            ]
        )
    )

    return summary_df, details_df


def _select_best_model(primary_df: pd.DataFrame, fallback_df: pd.DataFrame) -> str:
    primary_valid = primary_df.dropna(subset=["MAPE", "RMSE"]).copy()
    if not primary_valid.empty:
        primary_valid = primary_valid.sort_values(["MAPE", "RMSE"], ascending=[True, True])
        return str(primary_valid.iloc[0]["Model"])

    fallback_valid = fallback_df.dropna(subset=["MAPE", "RMSE"]).copy()
    if not fallback_valid.empty:
        fallback_valid = fallback_valid.sort_values(["MAPE", "RMSE"], ascending=[True, True])
        return str(fallback_valid.iloc[0]["Model"])

    return LINEAR_MODEL


def forecast_sales(
    df: pd.DataFrame,
    periods: int = 6,
    holdout: int = 6,
    rolling_folds: int = 6,
    rolling_horizon: int = 1,
) -> dict:
    monthly_sales = build_monthly_sales_series(df)
    if monthly_sales.empty:
        empty = pd.DataFrame(columns=["Month", "Sales"])
        return {
            "historical": empty,
            "forecast": pd.DataFrame(columns=["Month", "Predicted Sales"]),
            "model_metrics": {"rmse": None, "mape": None, "holdout_months": 0, "metric_source": "none"},
            "model_benchmark": pd.DataFrame(columns=["Model", "RMSE", "MAPE", "Holdout Months"]),
            "rolling_benchmark_summary": pd.DataFrame(columns=["Model", "RMSE", "MAPE", "Folds", "Horizon"]),
            "rolling_backtest_details": pd.DataFrame(
                columns=["Model", "Fold", "Train End", "Test Start", "Test End", "Horizon", "RMSE", "MAPE"]
            ),
            "selected_model": LINEAR_MODEL,
        }

    benchmark_df = benchmark_models(monthly_sales, holdout=holdout)
    rolling_summary_df, rolling_details_df = rolling_backtest_models(
        monthly_sales,
        folds=rolling_folds,
        horizon=rolling_horizon,
        min_train_points=max(8, holdout),
    )

    selected_model = _select_best_model(rolling_summary_df, benchmark_df)
    fitted = _fit_model(monthly_sales, selected_model)

    future_index = pd.date_range(
        monthly_sales.index.max() + pd.offsets.MonthBegin(1), periods=periods, freq="MS"
    )
    predicted = _predict_model(monthly_sales, selected_model, periods=periods)

    rolling_selected = rolling_summary_df[rolling_summary_df["Model"] == selected_model]
    holdout_selected = benchmark_df[benchmark_df["Model"] == selected_model]

    if not rolling_selected.empty and rolling_selected["MAPE"].notna().any():
        selected_metrics = rolling_selected.iloc[0]
        model_metrics = {
            "rmse": float(selected_metrics["RMSE"]),
            "mape": float(selected_metrics["MAPE"]),
            "holdout_months": int(selected_metrics["Folds"]),
            "metric_source": "rolling_backtest",
        }
    elif not holdout_selected.empty and holdout_selected["MAPE"].notna().any():
        selected_metrics = holdout_selected.iloc[0]
        model_metrics = {
            "rmse": float(selected_metrics["RMSE"]),
            "mape": float(selected_metrics["MAPE"]),
            "holdout_months": int(selected_metrics["Holdout Months"]),
            "metric_source": "single_holdout",
        }
    else:
        model_metrics = {
            "rmse": None,
            "mape": None,
            "holdout_months": 0,
            "metric_source": "none",
        }

    historical_df = pd.DataFrame(
        {
            "Month": monthly_sales.index,
            "Actual Sales": monthly_sales.values,
            "Fitted Sales": fitted,
            "Trend Sales": fitted,  # Backward compatibility for existing dashboard labels.
        }
    )
    forecast_df = pd.DataFrame({"Month": future_index, "Predicted Sales": predicted})

    return {
        "historical": historical_df,
        "forecast": forecast_df,
        "model_metrics": model_metrics,
        "model_benchmark": benchmark_df,
        "rolling_benchmark_summary": rolling_summary_df,
        "rolling_backtest_details": rolling_details_df,
        "selected_model": selected_model,
    }
