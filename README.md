# Sales Data Analysis Dashboard

A sales analytics dashboard built with **Python + Streamlit**. Upload your sales data — or use the built-in demo dataset — and get KPIs, trends, forecasts, product insights, anomaly alerts, data-quality checks and a shareable executive report (CSV, Markdown or PDF).

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.36-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Tests](https://img.shields.io/badge/Tests-36%20passing-brightgreen)

> **Live demo:** _(add your Streamlit Cloud link here after deploying — see [Deploy to Streamlit Cloud](#deploy-to-streamlit-cloud-free))_

---

## Contents

- [What it does](#what-it-does)
- [Quick start](#quick-start)
- [What you will see](#what-you-will-see)
- [Use your own data](#use-your-own-data)
- [Forecasting](#forecasting)
- [Features](#features)
- [Project structure](#project-structure)
- [Development](#development)
- [Scheduled reports](#scheduled-reports)
- [Troubleshooting](#troubleshooting)
- [Tech stack](#tech-stack)
- [License](#license)

---

## What it does

Point it at a sales file and it answers the questions a shop or small business actually asks:

- How much did we sell, and how much profit did we make?
- Which products and categories earn the most — and which lose money?
- Are sales going up or down, and what might the next 6 months look like?
- Which discounts are eating our margin?
- Is there anything unusual in the numbers — a sudden drop, a loss-making category, a bad month?
- Is the data itself trustworthy — missing values, duplicates, drift?

Everything is shown in **₹ (INR)**. No sign-up, no database, no paid service is needed to run it.

---

## Quick start

Pick the way that suits you. All three end at the same dashboard on **http://localhost:8501**.

### Option A — Docker (fastest, recommended)

```bash
git clone https://github.com/Ayushkumar418/Sales_Data_Analysis.git
cd Sales_Data_Analysis
docker compose up --build
```

Then open **http://localhost:8501** in your browser.

> `docker compose up -d --build` runs it in the background, so you can close the terminal and the app keeps running. Stop it with `docker compose down`.

### Option B — Run locally with Python

Needs **Python 3.12**.

```bash
git clone https://github.com/Ayushkumar418/Sales_Data_Analysis.git
cd Sales_Data_Analysis
pip install -r requirements.txt
streamlit run app.py
```

Then open **http://localhost:8501**.

> If `streamlit run app.py` does not work on your system, try `python -m streamlit run app.py`.

### Deploy to Streamlit Cloud (free)

Option C — deploy the app free on Streamlit Cloud:

1. Fork or push this repo to your GitHub account.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, and click **New app**.
3. Pick this repo, set the main file to `app.py`, and deploy. No secrets are needed for the default dataset.
4. Paste the live URL at the top of this README so the project has a public demo link.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, and click **New app**.
3. Pick this repo, set the main file to `app.py`, and deploy. No secrets are needed for the default dataset.
4. Paste the live URL at the top of this README so the project has a public demo link.

---

## What you will see

The dashboard opens with the built-in demo dataset (see [the demo data note](#a-note-on-the-demo-data)). Use the sidebar to switch the data source, tune the forecast, and set alert thresholds. The dashboard has 8 tabs:

| Tab | What it shows |
| --- | --- |
| **Overview** | Total sales, profit, margin, orders, average order value, monthly trend |
| **Category & Region Analysis** | Sales and profit by category, sub-category and region |
| **Forecast & Suggestions** | Next-months forecast with an 80% confidence band, model comparison, and business suggestions with real numbers |
| **Product Insights** | Month-over-month growth, top and lowest-profit products, discount buckets and their margins, category profitability |
| **Alerts & Actions** | Monthly sales anomalies (z-score), low-margin categories, negative-profit transactions, region risk |
| **Data Quality** | Quality score, rule checks (missing values, duplicates, minimum rows), and drift vs the baseline |
| **Dataset Info** | The raw and processed data, side by side, so you can see exactly what was cleaned |
| **Executive Report** | A summary report you can download as **CSV, Markdown or PDF**, or save as a timestamped snapshot |

### Sidebar controls

- **Data Source** — Default Dataset, Upload File, or Database (SQL).
- **Forecast Settings** — Forecast months (1–24, default 6), holdout months and rolling-backtest folds for model comparison.
- **Alert Thresholds** — Sales anomaly z-threshold, low-margin threshold, negative-profit cutoff.
- **Quality Gates** — Minimum rows, maximum missing %, maximum duplicate Order IDs, drift thresholds, and strict/fail-on-drift switches (used by scheduled jobs).

---

## Use your own data

### Supported formats

- **Upload a file:** `.csv`, `.xlsx`, `.xls` — pick **Upload File** in the sidebar.
- **SQL database:** pick **Database (SQL)** in the sidebar, give a connection URL and either a table name or a SQL query.
- **Default dataset:** the built-in Amazon India catalogue (`data/amazon.csv`), used when you choose **Default Dataset**.

### Columns your file should have

| Column | What it is | Required? |
| --- | --- | --- |
| `Order ID` | Unique ID for each order | Yes |
| `Order Date` | Date of the order | Yes |
| `Sales` | Sale amount for the order | Yes |
| `Profit` | Profit for the order | Recommended — see below |
| `Category` | Product category | Recommended |
| `Sub-Category` | Product sub-category | Recommended |
| `Region` | Sales region | Recommended |
| `Customer Segment` | e.g. Consumer, Corporate | Recommended |

Extra columns are fine — `Unit Price`, `Quantity`, `Discount Percentage`, `Rating`, `Rating Count` and `Product Name` are used when present.

### Honest data handling — nothing is invented

This dashboard **never makes up data** for your uploads:

- **No Profit column?** Profit is shown as **N/A**, not filled in as 12% of sales.
- **No Region / Segment?** Those rows are labelled **Unknown**, not given a made-up region.
- **No valid order dates?** The upload is rejected with a clear error instead of inventing a timeline.

### A note on the demo data

The default dataset is the Amazon India product catalogue, which has **no order dates, no profit and no regions at all**. So the app builds it with an explicitly labelled synthetic demo generator (`build_synthetic_demo_dataset` in `src/data_processing.py`) and shows a banner in the app saying its dates, regions and profit are simulated for illustration only.

That means the demo forecast is **not** a real prediction — the "months" behind it are simulated, so the forecast is flat and its error is high. Upload your own data with real dates and profit to get a meaningful forecast.

---

## Forecasting

The forecast compares four models and picks the best one by error (RMSE/MAPE) on a holdout period, then re-checks it with a rolling-origin backtest:

- **Linear Trend** — a straight-line trend.
- **Seasonal Naive** — repeats the same month from last year.
- **Exponential Smoothing** — weights recent months more heavily.
- **Holt-Winters** — trend + seasonality (falls back to Holt's linear trend when there are fewer than 2 seasons of data).

Every forecast comes with an **80% confidence band** — a lower and upper bound that widen the further out you forecast, so plan for the lower bound. Missing months are **interpolated**, not filled with 0. You can download the forecast as a CSV from the Forecast tab.

---

## Features

- **Data pipeline** — cleaning, deduplication, type conversion and schema normalisation into one common format, whatever the source.
- **Interactive dashboard** — Streamlit + Plotly, with KPI cards and dynamic filters.
- **Product, discount & growth insights** — month-over-month growth, best month, top products by sales with share and margin, lowest-profit products, discount buckets (no discount / 0–10% / 10–20% / 20%+) with margin per bucket, and category profitability.
- **Business suggestions with real numbers** — product names, ₹ amounts and forecast ranges, instead of generic templates.
- **Alerts** — monthly anomaly detection (z-score), low-margin categories, negative-profit transactions, region risk, all with adjustable thresholds.
- **Data quality & drift** — rule-based quality checks, a quality score, critical/warning failure tracking, and a drift table (current slice vs baseline) using mean-change and the KS statistic.
- **Executive reporting** — download the report as CSV, Markdown or PDF; save timestamped snapshots; run the same report from the terminal for scheduled jobs (see below).
- **Safety** — table names in SQL queries are validated and quoted, so a malicious table name cannot inject SQL.
- **Tests** — 36 automated tests (`pytest`) covering config, data processing, forecasting, insights, jobs and PDF reporting.

---

## Project structure

```text
Sales_Data_Analysis/
|-- app.py                  # Streamlit dashboard (8 tabs)
|-- analysis.ipynb          # Executed notebook: cleaning, EDA, forecasting, suggestions
|-- Dockerfile              # One-command Docker run
|-- docker-compose.yml      # Docker run with data/outputs persisted on the host
|-- .dockerignore
|-- requirements.txt        # Runtime dependencies (pinned)
|-- requirements-dev.txt    # Dev-only: notebooks + tests
|-- LICENSE                 # MIT
|-- README.md
|-- TECH.md                 # Detailed technical docs: architecture, modules, API
|-- tests/
|   |-- test_config.py
|   |-- test_data_processing.py
|   |-- test_forecasting.py
|   |-- test_insights.py
|   |-- test_jobs.py
|   `-- test_reporting.py
|-- data/
|   |-- amazon.csv          # Default demo dataset (Amazon India catalogue)
|   `-- retail_sales.csv    # Processed demo dataset (built from amazon.csv)
|-- outputs/                # Generated, git-ignored; rebuilt by src/pipeline and src/jobs
|   |-- figures/
|   `-- reports/
`-- src/
    |-- anomalies.py        # Anomaly + low-margin + negative-profit detection
    |-- config.py           # Paths, currency (₹), table-name validation, constants
    |-- data_processing.py  # Cleaning, normalisation, synthetic demo generator
    |-- eda.py              # Plots for the notebook and dashboard
    |-- forecasting.py      # Linear / Seasonal Naive / SES / Holt-Winters + backtests
    |-- insights.py         # KPIs, product/discount/category insights, suggestions
    |-- jobs.py             # CLI job runner for scheduled reports
    |-- pipeline.py         # Build processed dataset + static figures
    |-- quality.py          # Quality checks, score, drift
    `-- reporting.py        # CSV / Markdown / PDF executive reports
```

Generated files under `outputs/` are not committed to git.

For detailed technical documentation — architecture, design patterns, module-by-module reference and extension points — see **[TECH.md](TECH.md)**.

---

## Development

Install the dev dependencies (notebooks + tests) instead of the runtime ones:

```bash
pip install -r requirements-dev.txt
```

Run the tests:

```bash
python -m pytest -q
```

Build the processed dataset and static figure artifacts:

```bash
python -m src.pipeline
```

Run the executed notebook again after changing the code:

```bash
jupyter nbconvert --to notebook --execute --inplace analysis.ipynb
```

> The notebook needs `ipython==8.26.0` (already pinned in `requirements-dev.txt`) — IPython 9 breaks matplotlib 3.8.4 when the notebook runs.

---

## Scheduled reports

Generate a report snapshot from the terminal:

```bash
python -m src.jobs --source default --output-dir outputs/reports
```

Strict quality gate example (fails if quality or drift checks fail):

```bash
python -m src.jobs --source default --output-dir outputs/reports --strict-quality --fail-on-drift --min-rows 100 --max-missing-pct 0.15 --drift-mean-change-threshold 20 --drift-ks-threshold 0.25
```

Database-based snapshot:

```bash
python -m src.jobs --source database --connection-url "sqlite:///data/sales.db" --sql-query "SELECT * FROM sales" --output-dir outputs/reports
```

Use these commands with Windows Task Scheduler or cron for recurring automatic report generation.

---

## Troubleshooting

| Problem | Fix |
| --- | --- |
| Browser shows nothing at `http://0.0.0.0:8501` | `0.0.0.0` is a listen address, not a web address. Open **`http://localhost:8501`** instead. |
| `streamlit run app.py` does not work | Try `python -m streamlit run app.py`. |
| `ImportError` / crash on the first table after a fresh install | Re-install from `requirements.txt` — `pyarrow` is pinned to `14.0.2` because newer versions need NumPy 2 while this project pins NumPy 1.26.4. |
| Port 8501 already in use | Stop the other Streamlit app, or run with `streamlit run app.py --server.port 8502` (Docker: change the port mapping in `docker-compose.yml`). |
| Uploaded file is rejected: "No valid order dates found" | Your file needs a valid `Order Date` column. See [Use your own data](#use-your-own-data). |
| Forecast looks flat / error is very high | You are probably on the demo dataset, whose dates are simulated. Upload real sales data with real dates. |
| Docker build is slow the first time | Normal — it downloads the Python image and installs all dependencies once; later builds reuse the cache. |

---

## Tech stack

- **Python 3.12** — Pandas, NumPy
- **Streamlit 1.36** + **Plotly** — interactive dashboard
- **Matplotlib, Seaborn** — static charts (notebook and pipeline)
- **SQLAlchemy** — database source
- **ReportLab** — PDF executive reports
- **Jupyter Notebook** — `analysis.ipynb`
- **Docker** — one-command run
- **pytest** — 36 automated tests

---

## License

MIT — see [LICENSE](LICENSE). Free to use, modify and share.
