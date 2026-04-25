from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"

RAW_DATA_PATH = DATA_DIR / "amazon.csv"
PROCESSED_DATA_PATH = DATA_DIR / "retail_sales.csv"

REGIONS = ["North", "South", "East", "West", "Central"]
CUSTOMER_SEGMENTS = ["Consumer", "Corporate", "Home Office"]

REQUIRED_COLUMNS = [
    "Order ID",
    "Order Date",
    "Sales",
    "Profit",
    "Category",
    "Sub-Category",
    "Region",
    "Customer Segment",
]

DEFAULT_FORECAST_PERIODS = 6

