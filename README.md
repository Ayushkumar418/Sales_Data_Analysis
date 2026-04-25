# Sales Data Analysis Dashboard

A production-ready analytics project built with Python and Streamlit.
It covers data cleaning, EDA, interactive BI dashboards, forecasting, anomaly alerts, and executive reporting.

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
  - Multi-model comparison: Linear Trend, Seasonal Naive, Exponential Smoothing
  - Single holdout benchmark and rolling-origin backtest
  - Auto-selection of best model by error metrics

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
|-- README.md
|-- TECH.md
|-- data/
|   |-- amazon.csv
|   `-- retail_sales.csv
|-- outputs/
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
