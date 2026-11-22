import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"

RAW_DATA_PATH = DATA_DIR / "amazon.csv"
PROCESSED_DATA_PATH = DATA_DIR / "retail_sales.csv"

# Default dataset is Amazon India sales, so amounts are shown in INR.
CURRENCY_SYMBOL = "₹"
CURRENCY_CODE = "INR"

# A table name is only an identifier (optionally schema-qualified), never
# arbitrary SQL. Anything else is rejected before it reaches the database.
_TABLE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")


def validate_table_name(table_name: str) -> str:
    """Return a stripped, validated table name or raise ValueError."""
    name = (table_name or "").strip()
    if not name or len(name) > 128 or not _TABLE_NAME_RE.match(name):
        raise ValueError(
            "Invalid table name. Use only letters, digits and underscores, "
            "optionally as schema.table (for example: sales)."
        )
    return name


def quote_table_name(table_name: str) -> str:
    """Quote each dot-separated part of a validated table name."""
    return ".".join(f'"{part}"' for part in validate_table_name(table_name).split("."))

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

