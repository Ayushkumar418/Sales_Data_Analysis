# Technical Documentation - Sales Data Analysis Dashboard

## Overview

Sales Data Analysis Dashboard is a production-ready analytics platform built with Python and Streamlit. It provides a comprehensive solution for sales data processing, visualization, forecasting, and intelligent anomaly detection with enterprise-grade quality monitoring.

## Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Sources Layer                       │
│  ┌──────────────┬──────────────┬────────────────────────┐   │
│  │  CSV Files   │ Excel Files  │ SQL Databases          │   │
│  │ (Local/URL)  │              │ (SQLAlchemy Support)   │   │
│  └──────────────┴──────────────┴────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                  Data Processing Pipeline                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ • Schema Normalization                              │    │
│  │ • Type Conversion & Parsing                         │    │
│  │ • Deduplication & Cleaning                          │    │
│  │ • Dynamic Column Mapping                            │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│              Core Analytics & Monitoring                    │
│  ┌──────────────┬──────────────┬──────────────────────┐     │
│  │ Quality      │ Anomaly      │ KPI & Business       │     │
│  │ Assessment   │ Detection    │ Insights             │     │
│  └──────────────┴──────────────┴──────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│           Forecasting & Predictive Analytics                │
│  ┌────────────────┬────────────────┬─────────────────┐      │
│  │ Linear Trend   │ Seasonal Naive │ Exponential     │      │
│  │ Model          │ Model          │ Smoothing       │      │
│  └────────────────┴────────────────┴─────────────────┘      │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│              Presentation & Reporting Layer                 │
│  ┌──────────────────┬──────────────────┬────────────────┐   │
│  │ Streamlit UI     │ Executive        │ CLI Job        │   │
│  │ (Interactive)    │ Reports          │ Runner         │   │
│  │ • Dashboard      │ (CSV/Markdown)   │ (Scheduled)    │   │
│  │ • Filters        │                  │                │   │
│  │ • Visualizations │                  │                │   │
│  └──────────────────┴──────────────────┴────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Core Modules

### 1. Data Processing (`src/data_processing.py`)

**Responsibilities:**
- Load raw datasets from multiple formats (CSV, Excel, SQL)
- Normalize inconsistent data schemas to unified analytics format
- Handle encoding issues and data type conversions
- Implement intelligent column mapping with aliases

**Key Functions:**
- `load_raw_dataset()` - Load data with fallback encoding handling
- `normalize_sales_dataset()` - Convert raw data to standard schema
- `prepare_sales_dataset()` - End-to-end data preparation pipeline
- `ensure_processed_dataset()` - Caching layer for processed data

**Unified Schema:**
```python
{
    "Order ID": str,
    "Order Date": datetime64,
    "Sales": float64,
    "Profit": float64,
    "Discount Percentage": float64,
    "Quantity": int64,
    "Category": str,
    "Sub-Category": str,
    "Region": str,
    "Customer Segment": str,
    "Unit Price": float64,
    "Rating": float64
}
```

**Column Mapping Strategy:**
- Uses normalized key lookup (lowercase, alphanumeric only)
- Supports multiple aliases (e.g., "Order Date", "OrderDate", "Sale Date")
- Stable bucketing for consistent sampling across runs

---

### 2. Configuration (`src/config.py`)

**Centralized Configuration:**
- Data paths and directory structure
- Standard region and customer segment lists
- Required columns definition
- Forecast periods (default: 6 months)
- Quality and drift thresholds

**Purpose:** Single source of truth for constants, enabling easy modification without code changes.

---

### 3. Quality Assessment (`src/quality.py`)

**Data Quality Framework:**

#### Quality Checks
1. **Required Columns** - Schema validation
2. **Minimum Row Count** - Ensures sufficient volume for stable analytics
3. **Missing Data Ratio** - Tracks data completeness
4. **Duplicate Orders** - Detects data integrity issues
5. **Data Type Validation** - Ensures numeric/categorical correctness

**Quality Score Calculation:**
```
score = (1 - failing_checks / total_checks) * 100
```

**Severity Levels:**
- `critical` - Blocks downstream analytics
- `warning` - May reduce accuracy but allows processing

#### Data Drift Monitoring

Detects changes between current and baseline (previous slice) using:

1. **Mean Change Percentage** - For numeric columns
   ```
   drift = |current_mean - baseline_mean| / baseline_mean * 100
   ```

2. **Kolmogorov-Smirnov (KS) Test** - Distribution comparison
   ```
   KS_stat = max(|CDF_current(x) - CDF_baseline(x)|) for all x
   ```

**Configurable Thresholds:**
- `mean_change_pct` - Default: 25% change triggers flag
- `ks_stat` - Default: 0.30 KS statistic threshold

**Use Cases:**
- Detect data pipeline issues early
- Track seasonal and structural changes
- Fail-fast gates for scheduled jobs

---

### 4. Anomaly Detection (`src/anomalies.py`)

**Business Anomaly Detection:**

#### 1. Monthly Sales Anomalies
- **Method:** Z-score statistical analysis
- **Formula:** `Z = (X - μ) / σ`
- **Detection Rule:** `|Z-score| ≥ threshold` (default: 2.0)
- **Interpretation:**
  - Z ≥ 2.0: Unusually high sales (+2 std deviations)
  - Z ≤ -2.0: Unusually low sales (-2 std deviations)

#### 2. Negative Profit Detection
- Identifies transactions with profit below threshold
- Useful for margin management and pricing reviews

#### 3. Low-Margin Category Detection
- Calculates profit margin per category: `(Profit / Sales) × 100`
- Flags categories below threshold (default: 5%)
- Helps identify loss-making product lines

#### 4. Region Performance Analysis
- Regional profitability breakdown
- Identifies underperforming regions

**Alert Generation:**
- Structured alert messages with counts and metrics
- Severity context (z-scores, margin percentages)
- Ready for executive summarization

---

### 5. Forecasting (`src/forecasting.py`)

**Multi-Model Forecasting Approach:**

Three complementary models for robust predictions:

#### 1. Linear Trend Model
**Use Case:** Steady growth or decline trends
```
ŷ = slope × t + intercept

Training: Ordinary Least Squares (OLS) regression
Predictions: Clipped to [0, ∞) for realistic sales values
```

#### 2. Seasonal Naive Model
**Use Case:** Strong seasonal patterns
```
ŷ(t) = y(t - season_length)

Default: 12-month seasonality
Logic: "Same month last year" assumption
Fallback: Uses 1-month lag if insufficient history
```

#### 3. Exponential Smoothing Model
**Use Case:** Adaptive trend + level changes
```
level(t) = α × y(t) + (1 - α) × level(t-1)
ŷ(t+h) = level(t)

Auto-tuning: Optimizes α via grid search [0.01, 0.99]
Metric: RMSE minimization
```

**Model Selection Strategy:**

1. **Validation Setup:**
   - **Single Holdout:** Reserve last 12 months for validation
   - **Rolling Origin Backtest:** Multiple train-validation splits

2. **Performance Metrics:**
   - **RMSE:** Penalizes large errors (scale: same as sales)
   - **MAPE:** Percentage error (interpretable, robust to scale)
   - **Formula:** 
     ```
     RMSE = √(mean((y_true - y_pred)²))
     MAPE = mean(|y_true - y_pred| / |y_true|) × 100
     ```

3. **Best Model Selection:**
   - Automatic selection based on lowest RMSE
   - Metadata tracking for explainability

**Forecast Output:**
```python
{
    "selected_model": "Linear Trend",
    "forecast": pd.DataFrame({
        "Month": [...],
        "Predicted Sales": [...]
    }),
    "model_metrics": {
        "rmse": 15000.50,
        "mape": 8.75,
        "metric_source": "single_holdout_validation"
    }
}
```

---

### 6. Business Insights & KPIs (`src/insights.py`)

**Key Performance Indicators:**

1. **Total Sales** - Sum of all sales transactions
2. **Total Profit** - Sum of profits (Sales - Costs)
3. **Total Orders** - Count of unique transactions
4. **Average Order Value (AOV)** - `Total Sales / Total Orders`
5. **Profit Margin (%)** - `(Total Profit / Total Sales) × 100`
6. **Category/Region/Segment Analysis** - Breakdown performance
7. **Dynamic Suggestions** - Business recommendations based on data patterns

**Smart Recommendations Engine:**
- Low-performing segment detection
- High-profit category emphasis
- Geographic opportunity identification
- Trend-based growth suggestions

---

### 7. Reporting (`src/reporting.py`)

**Executive Report Generation:**

#### Report Structure

**CSV Format** (`executive_sales_report_*.csv`):
- Tabular format for spreadsheet import
- Sections: Meta, KPI, Forecast, Quality, Drift, Alerts
- Timestamped for audit trail

**Markdown Format** (`executive_sales_report_*.md`):
- Human-readable narrative format
- Highlighted alerts and anomalies
- Formatted tables for readability
- Executive summary at top

#### Report Content

| Section | Metrics |
|---------|---------|
| **Meta** | Dataset label, row count, date range |
| **KPI** | Sales, profit, orders, margin, AOV |
| **Forecast** | Selected model, RMSE, MAPE, accuracy |
| **Quality** | Quality score, critical/warning failures |
| **Drift** | Flagged features, distribution changes |
| **Alerts** | Business anomalies detected |

#### Report Versioning

```
outputs/reports/
├── executive_sales_report_latest.csv       (current)
├── executive_sales_report_latest.md        (current)
├── executive_sales_report_20260425_000211.csv  (timestamped)
└── executive_sales_report_20260425_000211.md  (timestamped)
```

**Features:**
- Timestamped snapshots for version history
- `_latest` symlink/copy for quick access
- Reproducible report generation

---

### 8. CLI Job Runner (`src/jobs.py`)

**Scheduled Report Generation:**

**Purpose:** Automate report generation for batch processing or cron jobs

**Command Format:**
```bash
python -m src.jobs \
  --source default \
  --start-date "2024-01-01" \
  --end-date "2024-12-31" \
  --regions "North,South" \
  --categories "Electronics,Furniture" \
  --segments "Consumer" \
  --quality-gate strict \
  --drift-gate flag
```

**Options:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--source` | Data source: `default` or `database` | `default` |
| `--connection-url` | Database connection string | - |
| `--table-name` | Database table name | - |
| `--sql-query` | Custom SQL query | - |
| `--start-date` | Filter start date (YYYY-MM-DD) | None |
| `--end-date` | Filter end date (YYYY-MM-DD) | None |
| `--regions` | Comma-separated regions | All |
| `--categories` | Comma-separated categories | All |
| `--segments` | Comma-separated segments | All |
| `--quality-gate` | `lenient`, `standard`, `strict` | `standard` |
| `--drift-gate` | `none`, `flag`, `fail` | `flag` |

**Quality Gates:**
- `lenient`: Only critical checks enforced
- `standard`: All checks, warnings permitted
- `strict`: All checks must pass, fail on warnings

**Drift Gates:**
- `none`: No drift checking
- `flag`: Log warnings but continue
- `fail`: Stop job if drift detected

---

### 9. EDA & Visualization (`src/eda.py`)

**Static Visualization Generation:**

Purpose: Generate publication-quality figures for reports and presentations

**Plots Generated:**
1. **Monthly Sales Trend** - Line plot with markers
2. **Category Sales Distribution** - Bar chart (top N)
3. **Sub-Category Performance** - Top 12 performers
4. **Region Performance** - Sales vs Profit by region
5. **Correlation Matrix** - Heatmap of numeric relationships
6. **Segment Analysis** - Customer segment breakdown

**Output Format:**
- PNG files (300 DPI for printing)
- Saved to `outputs/figures/`
- Consistent styling (Seaborn + Matplotlib)

---

### 10. Pipeline Orchestration (`src/pipeline.py`)

**Entry Point for Batch Processing:**

```python
def main():
    # 1. Load & process dataset
    df = ensure_processed_dataset(force_rebuild=True)
    
    # 2. Generate static visualizations
    save_example_visualizations(df)
    
    # 3. Output paths logged
```

**Usage:**
```bash
python -m src.pipeline
```

---

## Streamlit Application (`app.py`)

### Page Structure

#### 1. **Data Input**
- Default dataset (Amazon CSV → retail schema)
- File upload (CSV, Excel)
- Database connection (SQLAlchemy)
- Format validation & error handling

#### 2. **Dashboard**
- KPI metric cards
- Interactive filters (regions, categories, segments, date range)
- Trend visualization (Plotly line chart)
- Category drill-down (bar chart)
- Region performance (scatter/bar)
- Category correlation heatmap

#### 3. **Forecasting**
- Model selection sidebar
- 6-month forecast visualization
- Backtest metrics display
- Model performance comparison

#### 4. **Anomalies & Alerts**
- Monthly sales anomalies table
- Negative profit transactions
- Low-margin category warnings
- Regional risk summary
- Adjustable threshold controls

#### 5. **Data Quality**
- Quality check results with status
- Severity indicators (critical/warning)
- Quality score display
- Drift analysis table
- Configurable thresholds

#### 6. **Insights**
- Auto-generated business recommendations
- Key metrics summary
- Segment performance analysis
- Actionable suggestions engine

#### 7. **Executive Report**
- Generate comprehensive report
- Download as CSV or Markdown
- Save timestamped snapshot
- View latest report

### Caching Strategy

**Streamlit Cache Decorators:**
```python
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    return ensure_processed_dataset(force_rebuild=False)
```

**Benefits:**
- Eliminates redundant computations
- Fast interactive filtering
- Responsive dashboard experience

---

## Data Flow

### Interactive Dashboard Flow
```
User Input (Filters)
        ↓
Load Cached Data
        ↓
Apply Filters
        ↓
Compute KPIs
        ↓
Run Forecasting
        ↓
Detect Anomalies
        ↓
Assess Quality/Drift
        ↓
Generate Visualizations
        ↓
Render Dashboard
```

### Batch Report Generation Flow
```
CLI Arguments
        ↓
Load Dataset (CSV/Excel/DB)
        ↓
Normalize Schema
        ↓
Apply Filters
        ↓
Quality Gate Check (fail-fast)
        ↓
Drift Detection
        ↓
Compute Analytics (KPIs, Forecast, Anomalies)
        ↓
Build Report (CSV + Markdown)
        ↓
Save Timestamped Snapshot
        ↓
Update _latest Reference
```

---

## Technology Stack

### Core Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| **pandas** | ≥2.2.0 | Data manipulation & analysis |
| **numpy** | ≥1.26.0 | Numerical computations |
| **streamlit** | ≥1.36.0 | Web UI framework |
| **plotly** | ≥5.22.0 | Interactive visualizations |
| **matplotlib** | ≥3.8.0 | Static plotting |
| **seaborn** | ≥0.13.0 | Statistical visualizations |
| **sqlalchemy** | ≥2.0.30 | Database abstraction |
| **openpyxl** | ≥3.1.2 | Excel file support |
| **xlrd** | ≥2.0.1 | Legacy Excel support |
| **jupyter** | ≥1.0.0 | Notebook environment |

### Python Version
- **3.8+** (recommend 3.10+)

### External Dependencies
- SQLAlchemy (optional): For database connectivity
- Database drivers (PostgreSQL, MySQL, SQLite, etc.): As needed

---

## Design Patterns

### 1. **Schema Normalization Pattern**
- Flexible input → Standardized schema
- Handles variations in column names and types
- Enables pluggable data sources

### 2. **Quality Gate Pattern**
- Early validation before expensive operations
- Configurable fail-fast behavior
- Comprehensive audit trail

### 3. **Multi-Model Selection Pattern**
- Ensemble approach for robustness
- Automatic model selection via metrics
- Transparent metric reporting

### 4. **Caching Layer Pattern**
- Streamlit cache for performance
- Reduced data load frequency
- Fast interactive experience

### 5. **Configuration-Driven Design**
- Centralized config (`config.py`)
- Easy threshold/parameter tuning
- No code changes for configuration

---

## Performance Considerations

### Data Processing
- Vectorized Pandas operations (no Python loops)
- Efficient column mapping using dictionaries
- In-memory processing for typical datasets (< 1GB)

### Forecasting
- Lightweight statistical models (linear, exponential smoothing)
- No ML library overhead
- Fast inference time (< 1 second for 6-month forecast)

### Visualization
- Plotly caching at browser level
- Lazy rendering for large datasets
- Matplotlib backend optimized for headless rendering

### Scaling Considerations
- **For 10M+ rows:** Implement data sampling or batching
- **For real-time updates:** Add event-streaming integration
- **For multi-user:** Deploy on Streamlit Cloud or Heroku

---

## Error Handling & Validation

### Data Validation
```python
# Encoding fallback (UTF-8 → Latin-1)
# Missing column handling (skip or use defaults)
# Type conversion safety (coercion with error handling)
# Duplicate deduplication (preserve first occurrence)
```

### User Input Validation
```python
# Connection URL verification
# SQL query validation
# Date range validation
# File type/size limits
```

### Graceful Degradation
```python
# Missing data: Fill with defaults or skip
# Insufficient data: Return empty results with message
# Model failures: Fall back to next model
# Database errors: Suggest retry with connection check
```

---

## Security Considerations

1. **Database Connection** - Use environment variables, never hardcode credentials
2. **File Upload Limits** - Restrict file size to prevent memory exhaustion
3. **SQL Injection Prevention** - Use SQLAlchemy parameterized queries
4. **Data Privacy** - Ensure PII is handled per regulations (GDPR, CCPA)
5. **Report Sensitivity** - Control report access and retention policies

---

## Extension Points

### Adding New Forecasting Models
1. Implement `_predict_*` and `_fit_*` functions in `forecasting.py`
2. Add to `AVAILABLE_MODELS` list
3. Update model selection logic

### Adding New Anomaly Detections
1. Implement detection function in `anomalies.py`
2. Add to `detect_business_anomalies()` output
3. Wire into alert generation

### Custom Data Sources
1. Implement data loader function
2. Add to `_load_dataset()` in `jobs.py`
3. Add schema mapping for new format

### Additional Visualizations
1. Add plot function to `eda.py`
2. Add corresponding tab to Streamlit `app.py`
3. Wire up interactive filter controls

---

## Testing & Validation

### Data Validation Testing
- Test with malformed CSV files
- Test with missing required columns
- Test with mismatched data types
- Test with duplicate records

### Forecasting Validation
- Backtest models on historical data
- Compare RMSE/MAPE across models
- Validate predictions with holdout data
- Test edge cases (no data, single value)

### Quality Gate Validation
- Verify threshold behavior
- Test critical vs warning severity
- Validate fail-fast gates

---

## Deployment Guide

### Local Development
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Docker Deployment
```dockerfile
FROM python:3.11
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501"]
```

### Scheduled Jobs (Cron)
```bash
0 2 * * * cd /path/to/Sales_Data_Analysis && \
  python -m src.jobs --source default --quality-gate strict
```

---

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Encoding errors | File encoding mismatch | System uses UTF-8 fallback, ensure input data encoding |
| Empty forecast | Insufficient historical data | Need minimum 12 months for seasonal models |
| Quality gate fail | Data drift or missing columns | Review data quality report, check upstream source |
| Slow dashboard | Large dataset | Implement date filtering or data sampling |
| DB connection fail | Invalid credentials | Verify connection string format and credentials |

---

## Future Enhancements

1. **Advanced Forecasting**
   - Prophet integration for complex seasonality
   - ARIMA models for time series
   - Ensemble forecasting methods

2. **ML-Based Anomaly Detection**
   - Isolation Forest for multivariate anomalies
   - Autoencoder for pattern detection
   - Clustering for customer segmentation

3. **Real-Time Streaming**
   - Kafka integration for live data
   - Incremental model updates
   - Real-time dashboard refresh

4. **Advanced Reporting**
   - PDF generation with charts
   - Email delivery automation
   - Slack/Teams integration

5. **Data Governance**
   - Data lineage tracking
   - Audit logging
   - Role-based access control (RBAC)

---

## Contributing

For architecture changes or new features:
1. Document changes in this TECH.md
2. Update configuration if adding new parameters
3. Add unit tests for new modules
4. Ensure backward compatibility

---

**Last Updated:** April 2026  
**Version:** 1.0  
**Maintainer:** Sales Analytics Team
