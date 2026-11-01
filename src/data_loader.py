"""Data loading, validation and cleaning for the India housing dataset."""

# Type hints for readable function signatures
from __future__ import annotations

# OS-independent paths
from pathlib import Path

# Numerical helpers
import numpy as np

# Tabular data handling
import pandas as pd

# Project configuration and logger
from src.config import CURRENT_YEAR, RAW_DATA_PATH
from src.logger import get_logger

# Module-level logger
logger = get_logger(__name__)

# Columns that must exist in the raw file (taken from the project brief)
REQUIRED_COLUMNS = [
    "ID",
    "State",
    "City",
    "Locality",
    "Property_Type",
    "BHK",
    "Size_in_SqFt",
    "Price_in_Lakhs",
    "Price_per_SqFt",
    "Year_Built",
    "Furnished_Status",
    "Floor_No",
    "Total_Floors",
    "Age_of_Property",
    "Nearby_Schools",
    "Nearby_Hospitals",
    "Public_Transport_Accessibility",
    "Parking_Space",
    "Security",
    "Amenities",
    "Facing",
    "Owner_Type",
    "Availability_Status",
]
# Numeric columns that must be strictly positive to make business sense
POSITIVE_COLUMNS = ["BHK", "Size_in_SqFt", "Price_in_Lakhs", "Total_Floors"]


class DataValidationError(Exception):
    """Raised when the input data does not meet the expected schema."""


def load_raw_data(path: Path | str = RAW_DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV and verify that every required column is present."""
    try:
        # Read the CSV file from disk
        df = pd.read_csv(path)
    except FileNotFoundError as exc:
        # Give a clear message when the dataset is missing
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Place india_housing_prices.csv in data/raw/."
        ) from exc
    except pd.errors.EmptyDataError as exc:
        # An empty file cannot be analysed
        raise DataValidationError(f"The file '{path}' is empty.") from exc
    except pd.errors.ParserError as exc:
        # A corrupted CSV cannot be parsed
        raise DataValidationError(
            f"The file '{path}' is not a valid CSV: {exc}"
        ) from exc

    # Find any expected column that is absent
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    # Stop early if the schema is wrong
    if missing_cols:
        raise DataValidationError(f"Missing required columns: {missing_cols}")
    # Log what was loaded
    logger.info(
        "Loaded %s rows and %s columns from %s", f"{len(df):,}", df.shape[1], path
    )
    # Return the validated frame
    return df


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean the raw data and return the cleaned frame plus an audit report of every step."""
    # Work on a copy so the raw frame stays untouched
    out = df.copy()
    # Dictionary that records what each cleaning step changed
    report: dict = {"rows_raw": int(len(out))}

    # Names of text columns (anything that is not numeric)
    text_cols = [c for c in out.columns if not pd.api.types.is_numeric_dtype(out[c])]
    # Strip stray whitespace from every text column
    for col in text_cols:
        out[col] = out[col].astype("string").str.strip()
    # Replace empty strings with proper missing values
    out[text_cols] = out[text_cols].replace("", pd.NA)

    # Step 1: remove duplicated listings (ignoring the unique ID column)
    before = len(out)
    out = out.drop_duplicates(subset=[c for c in out.columns if c != "ID"])
    report["duplicates_removed"] = int(before - len(out))

    # Step 2: drop rows where the key numeric fields are unusable
    before = len(out)
    key_cols = ["Price_in_Lakhs", "Size_in_SqFt", "BHK", "Year_Built"]
    out = out.dropna(subset=key_cols)
    report["rows_dropped_missing_key_fields"] = int(before - len(out))

    # Step 3: impute remaining gaps (median for numbers, mode for categories)
    imputed = 0
    for col in out.columns:
        # Count the gaps in this column
        n_missing = int(out[col].isna().sum())
        # Skip complete columns
        if n_missing == 0:
            continue
        # Choose a fill value that matches the column type
        fill = (
            out[col].median()
            if pd.api.types.is_numeric_dtype(out[col])
            else out[col].mode().iloc[0]
        )
        # Fill the gaps
        out[col] = out[col].fillna(fill)
        # Track how many cells were imputed
        imputed += n_missing
    report["cells_imputed"] = imputed

    # Step 4: remove rows that break basic business rules
    before = len(out)
    # Positive price, size, BHK and floor count
    for col in POSITIVE_COLUMNS:
        out = out[out[col] > 0]
    # A property cannot be built in the future
    out = out[out["Year_Built"] <= CURRENT_YEAR]
    report["rows_dropped_invalid_values"] = int(before - len(out))

    # Step 5: recompute property age from Year_Built so it is always consistent
    recomputed_age = CURRENT_YEAR - out["Year_Built"]
    report["age_values_corrected"] = int(
        (recomputed_age != out["Age_of_Property"]).sum()
    )
    out["Age_of_Property"] = recomputed_age

    # Step 6: fix impossible floors (floor number above the building height)
    bad_floor = out["Floor_No"] > out["Total_Floors"]
    # Keep a flag so the correction stays transparent
    out["Floor_Data_Corrected"] = bad_floor.astype(int)
    # Cap the floor at the top floor of the building
    out.loc[bad_floor, "Floor_No"] = out.loc[bad_floor, "Total_Floors"]
    report["floor_values_corrected"] = int(bad_floor.sum())

    # Step 7: rebuild price per sq ft in rupees (the source column is rounded to 2 decimals of lakhs)
    out["Price_per_SqFt_INR"] = out["Price_in_Lakhs"] * 100_000 / out["Size_in_SqFt"]

    # Step 8: standardise the Yes/No columns
    for col in ["Parking_Space", "Security"]:
        out[col] = out[col].str.title()

    # Reset the index after dropping rows
    out = out.reset_index(drop=True)
    # Final row count
    report["rows_clean"] = int(len(out))
    # Log a compact summary
    logger.info(
        "Cleaning complete: %s -> %s rows",
        f"{report['rows_raw']:,}",
        f"{report['rows_clean']:,}",
    )
    # Return the clean frame and the audit trail
    return out, report


def detect_outliers_iqr(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Count outliers per column using the 1.5 x IQR rule."""
    # Collect one result row per column
    rows = []
    for col in columns:
        # First and third quartiles
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        # Inter-quartile range
        iqr = q3 - q1
        # Lower and upper fences
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        # Number of values outside the fences
        n_out = int(((df[col] < low) | (df[col] > high)).sum())
        # Store the summary
        rows.append(
            {
                "Column": col,
                "Lower_Fence": round(float(low), 3),
                "Upper_Fence": round(float(high), 3),
                "Outliers": n_out,
                "Outlier_Pct": round(100 * n_out / len(df), 2),
            }
        )
    # Return as a tidy table
    return pd.DataFrame(rows)


def winsorize(series: pd.Series, low: float, high: float) -> pd.Series:
    """Clip a series to fixed lower and upper bounds."""
    # Replace values beyond the bounds with the bounds themselves
    return series.clip(lower=low, upper=high)


def safe_numeric(value, default: float = np.nan) -> float:
    """Convert a value to float, returning a default when conversion fails."""
    try:
        # Attempt the conversion
        return float(value)
    except (TypeError, ValueError):
        # Fall back to the default on failure
        return default
