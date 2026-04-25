import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    CUSTOMER_SEGMENTS,
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
    REGIONS,
    REQUIRED_COLUMNS,
)


def _stable_bucket(value: str, buckets: int) -> int:
    digest = hashlib.md5(value.encode("utf-8")).hexdigest()
    return int(digest, 16) % buckets


def _parse_currency(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace(r"[^0-9.\-]", "", regex=True)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _parse_percentage(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.replace("%", "", regex=False)
    cleaned = cleaned.str.replace(r"[^0-9.\-]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def _parse_count(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.replace(",", "", regex=False)
    cleaned = cleaned.str.replace(r"[^0-9.\-]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def _load_with_fallback(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding_errors="ignore")


def load_raw_dataset(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {path}")
    return _load_with_fallback(path)


def _extract_sub_category(category_value: str) -> str:
    parts = str(category_value).split("|")
    if len(parts) > 1:
        return parts[1].strip()
    return parts[0].strip() if parts else "Unknown"


def _normalize_col_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def _build_column_lookup(columns: list[str]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for col in columns:
        key = _normalize_col_name(col)
        if key and key not in lookup:
            lookup[key] = col
    return lookup


def _pick_column(df: pd.DataFrame, aliases: list[str]) -> str | None:
    lookup = _build_column_lookup(df.columns.tolist())
    for alias in aliases:
        key = _normalize_col_name(alias)
        if key in lookup:
            return lookup[key]
    return None


def _as_text(series: pd.Series, default_value: str = "Unknown") -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .replace({"": default_value, "nan": default_value, "None": default_value})
        .fillna(default_value)
    )


def normalize_sales_dataset(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy().drop_duplicates().reset_index(drop=True)
    if df.empty:
        raise ValueError("Uploaded dataset has no rows.")

    order_id_col = _pick_column(df, ["Order ID", "Invoice ID", "Transaction ID", "OrderNumber"])
    order_date_col = _pick_column(
        df,
        [
            "Order Date",
            "Date",
            "OrderDate",
            "Invoice Date",
            "Transaction Date",
            "Purchase Date",
            "Ship Date",
        ],
    )
    sales_col = _pick_column(df, ["Sales", "Revenue", "Amount", "Net Sales", "Total Sales", "Total"])
    profit_col = _pick_column(df, ["Profit", "Net Profit", "Gross Profit", "Earnings"])
    category_col = _pick_column(df, ["Category", "Product Category", "Main Category", "Department"])
    sub_category_col = _pick_column(
        df, ["Sub-Category", "Sub Category", "SubCategory", "Product Sub-Category", "Subcat"]
    )
    region_col = _pick_column(df, ["Region", "Zone", "Market", "State", "Territory", "Country"])
    segment_col = _pick_column(
        df, ["Customer Segment", "Segment", "Customer Type", "Customer Group"]
    )
    product_name_col = _pick_column(
        df, ["Product Name", "Product", "Item Name", "Item", "SKU", "Product_Name"]
    )
    unit_price_col = _pick_column(df, ["Unit Price", "Price", "Selling Price", "UnitPrice"])
    quantity_col = _pick_column(df, ["Quantity", "Qty", "Units", "Order Quantity"])
    discount_col = _pick_column(
        df, ["Discount", "Discount %", "Discount Percentage", "DiscountPercentage"]
    )
    rating_col = _pick_column(df, ["Rating", "Review Rating", "Score"])
    rating_count_col = _pick_column(df, ["Rating Count", "Review Count", "Reviews", "RatingCount"])

    if sales_col is None:
        lookup = _build_column_lookup(df.columns.tolist())
        is_amazon_like = {
            "discountedprice",
            "actualprice",
            "discountpercentage",
            "ratingcount",
        }.issubset(set(lookup.keys()))
        if is_amazon_like:
            return clean_and_engineer_dataset(df)
        if unit_price_col is None or quantity_col is None:
            raise ValueError(
                "Could not identify a sales column. Include one like 'Sales' or 'Revenue', "
                "or provide both unit price and quantity columns."
            )

    if sales_col is not None:
        sales = _parse_currency(df[sales_col])
    else:
        unit_price_for_sales = _parse_currency(df[unit_price_col])
        quantity_for_sales = _parse_count(df[quantity_col]).fillna(1)
        sales = unit_price_for_sales * quantity_for_sales

    sales = sales.fillna(0).clip(lower=0)

    if order_date_col is not None:
        order_date = pd.to_datetime(df[order_date_col], errors="coerce")
    else:
        order_date = pd.Series(pd.NaT, index=df.index)
    if order_date.notna().sum() == 0:
        # Fallback timeline enables trend charts even when date is missing.
        order_date = pd.Series(
            pd.Timestamp("2023-01-01") + pd.to_timedelta(df.index, unit="D"), index=df.index
        )

    if profit_col is not None:
        profit = _parse_currency(df[profit_col])
    else:
        profit = pd.Series(np.nan, index=df.index, dtype=float)
    profit = profit.fillna((sales * 0.12).round(2))

    if category_col is not None:
        category = _as_text(df[category_col], default_value="Unknown")
    else:
        category = pd.Series("Unknown", index=df.index)

    if sub_category_col is not None:
        sub_category = _as_text(df[sub_category_col], default_value="Unknown")
    else:
        sub_category = category.copy()

    if region_col is not None:
        region = _as_text(df[region_col], default_value="Unknown")
    else:
        row_key = pd.Series(df.index, index=df.index).astype(str)
        region = row_key.apply(lambda x: REGIONS[_stable_bucket(f"{x}-region", len(REGIONS))])

    if segment_col is not None:
        customer_segment = _as_text(df[segment_col], default_value="Consumer")
    else:
        row_key = pd.Series(df.index, index=df.index).astype(str)
        customer_segment = row_key.apply(
            lambda x: CUSTOMER_SEGMENTS[_stable_bucket(f"{x}-segment", len(CUSTOMER_SEGMENTS))]
        )

    if product_name_col is not None:
        product_name = _as_text(df[product_name_col], default_value="Unknown Product")
    else:
        product_name = pd.Series(df.index, index=df.index).map(lambda x: f"Item {x + 1}")

    if quantity_col is not None:
        quantity = _parse_count(df[quantity_col]).fillna(1)
    elif unit_price_col is not None:
        price_base = _parse_currency(df[unit_price_col]).replace(0, np.nan)
        quantity = np.round(sales / price_base).fillna(1)
    else:
        quantity = pd.Series(1, index=df.index, dtype=float)
    quantity = quantity.clip(lower=1).round().astype(int)

    if unit_price_col is not None:
        unit_price = _parse_currency(df[unit_price_col])
    else:
        safe_quantity = quantity.replace(0, 1)
        unit_price = sales / safe_quantity
    unit_price = unit_price.replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=0)

    if discount_col is not None:
        discount_percentage = _parse_percentage(df[discount_col]).fillna(0)
    else:
        discount_percentage = pd.Series(0.0, index=df.index)

    if rating_col is not None:
        rating = pd.to_numeric(df[rating_col], errors="coerce").round(2)
    else:
        rating = pd.Series(np.nan, index=df.index)

    if rating_count_col is not None:
        rating_count = _parse_count(df[rating_count_col]).fillna(0).astype(int)
    else:
        rating_count = pd.Series(0, index=df.index, dtype=int)

    if order_id_col is not None:
        order_id = _as_text(df[order_id_col], default_value="").replace("Unknown", "")
        missing_order_id = order_id.eq("")
        order_id = order_id.where(~missing_order_id, pd.NA)
    else:
        order_id = pd.Series(pd.NA, index=df.index)

    cleaned = pd.DataFrame(
        {
            "Order Date": pd.to_datetime(order_date, errors="coerce"),
            "Sales": sales.round(2),
            "Profit": profit.round(2),
            "Category": category,
            "Sub-Category": sub_category,
            "Region": region,
            "Customer Segment": customer_segment,
            "Product Name": product_name,
            "Rating": rating,
            "Rating Count": rating_count,
            "Discount Percentage": discount_percentage,
            "Unit Price": unit_price.round(2),
            "Quantity": quantity,
            "Order ID": order_id,
        }
    )

    cleaned = cleaned.dropna(subset=["Order Date"]).copy()
    cleaned = cleaned[cleaned["Sales"] >= 0].copy()
    cleaned = cleaned.sort_values("Order Date").reset_index(drop=True)

    missing_order_id = cleaned["Order ID"].isna() | cleaned["Order ID"].astype(str).str.strip().eq("")
    cleaned.loc[missing_order_id, "Order ID"] = [
        f"ORD-{idx + 1:05d}" for idx in cleaned.index[missing_order_id]
    ]

    column_order = REQUIRED_COLUMNS + [
        "Product Name",
        "Unit Price",
        "Quantity",
        "Discount Percentage",
        "Rating",
        "Rating Count",
    ]
    cleaned = cleaned[column_order]
    return cleaned


def prepare_sales_dataset(raw_df: pd.DataFrame) -> pd.DataFrame:
    lookup = _build_column_lookup(raw_df.columns.tolist())
    amazon_like = {
        "discountedprice",
        "actualprice",
        "discountpercentage",
        "ratingcount",
    }.issubset(set(lookup.keys()))
    if amazon_like:
        return clean_and_engineer_dataset(raw_df)
    return normalize_sales_dataset(raw_df)


def clean_and_engineer_dataset(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    df = df.drop_duplicates().reset_index(drop=True)

    discounted_price = _parse_currency(df.get("discounted_price", pd.Series(dtype=float)))
    actual_price = _parse_currency(df.get("actual_price", pd.Series(dtype=float)))
    discount_percentage = _parse_percentage(df.get("discount_percentage", pd.Series(dtype=float)))
    rating = pd.to_numeric(df.get("rating", pd.Series(dtype=float)), errors="coerce")
    rating_count = _parse_count(df.get("rating_count", pd.Series(dtype=float)))

    category_series = df.get("category", pd.Series(dtype=str)).fillna("Unknown").astype(str)
    category = category_series.str.split("|").str[0].str.strip().replace("", "Unknown")
    sub_category = category_series.apply(_extract_sub_category).replace("", "Unknown")

    product_key = df.get("product_id", pd.Series(dtype=str)).fillna("").astype(str)
    product_key = product_key.where(product_key != "", pd.Series(df.index, index=df.index).astype(str))

    start_date = pd.Timestamp("2023-01-01")
    end_date = pd.Timestamp("2025-12-31")
    day_span = int((end_date - start_date).days)
    # Deterministic date mapping keeps runs reproducible for portfolio demos.
    day_offsets = product_key.apply(lambda x: _stable_bucket(f"{x}-date", day_span + 1))
    order_date = start_date + pd.to_timedelta(day_offsets, unit="D")

    region = product_key.apply(lambda x: REGIONS[_stable_bucket(f"{x}-region", len(REGIONS))])
    customer_segment = product_key.apply(
        lambda x: CUSTOMER_SEGMENTS[_stable_bucket(f"{x}-segment", len(CUSTOMER_SEGMENTS))]
    )

    price = discounted_price.fillna(actual_price)
    valid_price_fallback = float(price.dropna().median()) if not price.dropna().empty else 0.0
    price = price.fillna(valid_price_fallback).clip(lower=0)

    # Estimate order quantity from engagement volume (rating count proxy).
    quantity = np.ceil(np.log1p(rating_count.fillna(1)) / 2).astype(int).clip(lower=1, upper=8)
    sales = (price * quantity).round(2)

    # Profit model combines baseline margin, rating signal, and discount pressure.
    margin = 0.10 + (rating.fillna(3.8) - 3.8) * 0.04 - discount_percentage.fillna(0) * 0.0015
    category_adjustment = category.apply(
        lambda x: (_stable_bucket(f"{x}-margin", 9) - 4) / 100
    )  # +/- 4%
    margin = (margin + category_adjustment).clip(lower=-0.2, upper=0.45)
    profit = (sales * margin).round(2)

    cleaned = pd.DataFrame(
        {
            "Order Date": order_date,
            "Sales": sales,
            "Profit": profit,
            "Category": category.fillna("Unknown"),
            "Sub-Category": sub_category.fillna("Unknown"),
            "Region": region,
            "Customer Segment": customer_segment,
            "Product Name": df.get("product_name", pd.Series(dtype=str)).fillna("Unknown"),
            "Rating": rating.round(2),
            "Rating Count": rating_count.fillna(0).astype(int),
            "Discount Percentage": discount_percentage.fillna(0),
            "Unit Price": price,
            "Quantity": quantity,
        }
    )

    cleaned = cleaned.dropna(subset=["Order Date", "Sales", "Profit"])
    cleaned = cleaned[cleaned["Sales"] >= 0].copy()
    cleaned = cleaned.sort_values("Order Date").reset_index(drop=True)
    cleaned["Order ID"] = cleaned.index.map(lambda idx: f"ORD-{idx + 1:05d}")

    column_order = REQUIRED_COLUMNS + [
        "Product Name",
        "Unit Price",
        "Quantity",
        "Discount Percentage",
        "Rating",
        "Rating Count",
    ]
    cleaned = cleaned[column_order]
    return cleaned


def save_processed_dataset(df: pd.DataFrame, path: Path = PROCESSED_DATA_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_processed_dataset(path: Path = PROCESSED_DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {path}")
    df = pd.read_csv(path, parse_dates=["Order Date"])
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Processed dataset is missing required columns: {missing}")
    return df


def ensure_processed_dataset(force_rebuild: bool = False) -> pd.DataFrame:
    if PROCESSED_DATA_PATH.exists() and not force_rebuild:
        return load_processed_dataset(PROCESSED_DATA_PATH)

    raw_df = load_raw_dataset(RAW_DATA_PATH)
    cleaned_df = clean_and_engineer_dataset(raw_df)
    save_processed_dataset(cleaned_df, PROCESSED_DATA_PATH)
    return cleaned_df
