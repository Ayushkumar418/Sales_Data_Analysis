# Sales Data Analysis Dashboard

A production-ready analytics project built with Python and Streamlit.
It covers data cleaning, EDA, interactive BI dashboards, forecasting, anomaly alerts, and executive reporting.

Amounts are shown in ₹ (INR) — the default dataset is Amazon India sales.

**Honest data handling:** the dashboard never invents data for real uploads.
If a CSV has no Profit column, profit is shown as **N/A**, not filled with 12% of sales.
If it has no Region/Segment, those rows are labelled **Unknown**, not assigned a made-up region.
If it has no valid order dates, the upload is rejected with a clear error instead of an invented timeline.
The default `data/amazon.csv` dataset is different: the Amazon catalogue has no dates, profit or regions at all,
so it is built by an explicitly-labelled **synthetic demo generator** (`build_synthetic_demo_dataset`) and the app
shows a banner saying its dates, regions and profit are simulated for illustration only.

## Project Overview

The project can analyze:
- Default dataset (`data/amazon.csv` -> processed to retail format)
- Uploaded files (`.csv`, `.xlsx`, `.xls`)
- SQL database sources (connection URL + table/query)

All inputs are normalized into a common analytics schema before visualization and forecasting.

## Features

- Data pipeline
  - Data cleaning, deduplication, type conversion
  - Schema normalization for common sales datasets
  - Unified fields: `Order Date`, `Sales`, `Profit`, `Category`, `Sub-Category`, `Region`, `Customer Segment`

- Interactive dashboard (Streamlit + Plotly)
  - KPI cards and dynamic filters
  - Interactive trend, category, region, and correlation views
  - Dataset Info tab (raw + processed transparency)

- Forecasting (real-world evaluation)
  - Multi-model comparison: Linear Trend, Seasonal Naive, Exponential Smoothing, Holt-Winters (trend + seasonality, no new dependency)
  - 80% confidence band on forecasts — the band widens with horizon, so plan for the lower bound
  - Missing months are interpolated, not filled with 0
  - Single holdout benchmark and rolling-origin backtest
  - Auto-selection of best model by error metrics
  - Download forecast as CSV from the Forecast tab

- Product, discount & growth insights (Product Insights tab)
  - Month-over-month growth, best month and average month
  - Top products by sales with share of sales and margin, and lowest-profit products
  - Discount buckets (no discount / 0–10% / 10–20% / 20%+) with margin per bucket
  - Category profitability table
  - Business suggestions with real numbers — product names, ₹ amounts and forecast ranges — instead of generic templates

- Alerts and actions
  - Monthly anomaly detection (z-score)
  - Low-margin category detection
  - Negative-profit transaction detection
  - Region risk summary
  - User-adjustable thresholds from sidebar

- Data quality and drift monitoring
  - Rule-based quality checks (required columns, minimum rows, missing ratios, duplicates)
  - Quality score and critical/warning failure tracking
  - Drift table (current slice vs baseline) using mean-change and KS statistic
  - Configurable fail-fast quality gates for scheduled jobs

- Executive reporting
  - Download report as CSV and Markdown from UI
  - Save timestamped snapshots to workspace
  - CLI job runner for scheduled report generation with strict quality/drift options

## Tech Stack

- Python
- Pandas, NumPy
- Plotly, Streamlit
- Matplotlib, Seaborn (static artifact generation)
- SQLAlchemy (database source)
- Jupyter Notebook

## Project Structure

```text
Sales_Data_Analysis/
|-- app.py
|-- analysis.ipynb
|-- requirements.txt
|-- requirements-dev.txt
|-- LICENSE
|-- README.md
|-- TECH.md
|-- tests/
|   |-- test_config.py
|   |-- test_forecasting.py
|   |-- test_insights.py
|   `-- test_data_processing.py
|-- data/
|   |-- amazon.csv
|   `-- retail_sales.csv
|-- outputs/  # generated, git-ignored; rebuilt by src.pipeline / src.jobs
|   |-- figures/
|   `-- reports/
`-- src/
    |-- anomalies.py
    |-- config.py
    |-- data_processing.py
    |-- eda.py
    |-- forecasting.py
    |-- insights.py
    |-- jobs.py
    |-- pipeline.py
    |-- quality.py
    `-- reporting.py
```

The default dataset lives only in `data/amazon.csv`. Generated files under
`outputs/` are not committed.

## Documentation

For detailed technical documentation, architecture diagrams, design patterns, and API reference, see **[TECH.md](TECH.md)**.

### Key Documentation Topics:
- **System Architecture** - High-level design and data flow
- **Core Modules** - Detailed documentation of each component:
  - Data Processing Pipeline
  - Quality Assessment & Drift Monitoring
  - Anomaly Detection Engine
  - Multi-Model Forecasting
  - Executive Reporting
  - CLI Job Runner
- **Data Schema** - Unified analytics schema definition
- **Technology Stack** - Complete dependency list
- **Deployment Guide** - Production deployment options
- **Extension Points** - How to add new features
- **Troubleshooting** - Common issues and solutions

## Run Locally

1. Clone the repository
```bash
git clone https://github.com/Ayushkumar418/Sales_Data_Analysis.git
cd Sales_Data_Analysis
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

For notebooks and tests, install the dev dependencies instead:

```bash
pip install -r requirements-dev.txt
```

Run tests:

```bash
python -m pytest -q
```

3. Build processed dataset + static figure artifacts:

```bash
python -m src.pipeline
```

4. Run dashboard:

```bash
streamlit run app.py
```
**Note**: If `streamlit run app.py` doesn't work on your system, try:
```bash
python -m streamlit run app.py
```

## Scheduled Reporting

Generate a report snapshot from terminal:

```bash
python -m src.jobs --source default --output-dir outputs/reports
```

Strict quality gate example:

```bash
python -m src.jobs --source default --output-dir outputs/reports --strict-quality --fail-on-drift --min-rows 100 --max-missing-pct 0.15 --drift-mean-change-threshold 20 --drift-ks-threshold 0.25
```

Database-based snapshot:

```bash
python -m src.jobs --source database --connection-url "sqlite:///data/sales.db" --sql-query "SELECT * FROM sales" --output-dir outputs/reports
```

Use these commands with Windows Task Scheduler or cron for recurring automatic report generation.
