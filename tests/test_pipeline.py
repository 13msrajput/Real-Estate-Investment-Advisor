"""Unit tests for cleaning, feature engineering and input validation (run with `pytest`)."""

# Tabular data handling
import pandas as pd

# Test framework
import pytest

# Project modules
from src.data_loader import (
    DataValidationError,
    clean_data,
    detect_outliers_iqr,
    load_raw_data,
)
from src.features import build_training_frame, engineer_features


@pytest.fixture(scope="module")
def sample_raw() -> pd.DataFrame:
    """A 3,000-row sample of the raw dataset keeps the tests fast."""
    # Load the raw file and sample it reproducibly
    return load_raw_data().sample(3000, random_state=0).reset_index(drop=True)


def test_missing_file_raises():
    """A wrong path must give a clear FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_raw_data("does_not_exist.csv")


def test_schema_validation(tmp_path):
    """A CSV without the required columns must be rejected."""
    # Write a tiny invalid CSV
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,2\n")
    with pytest.raises(DataValidationError):
        load_raw_data(bad)


def test_cleaning_fixes_floors_and_adds_pps(sample_raw):
    """No floor may exceed the building height and rupee price per sq ft must exist."""
    clean, report = clean_data(sample_raw)
    assert (clean["Floor_No"] <= clean["Total_Floors"]).all()
    assert "Price_per_SqFt_INR" in clean.columns
    assert report["rows_clean"] == len(clean)


def test_cleaning_removes_invalid_rows(sample_raw):
    """Rows with a non-positive price must be dropped and duplicates removed."""
    dirty = pd.concat([sample_raw, sample_raw.iloc[[0]]], ignore_index=True)
    dirty.loc[1, "Price_in_Lakhs"] = -5
    clean, report = clean_data(dirty)
    assert report["duplicates_removed"] >= 1
    assert (clean["Price_in_Lakhs"] > 0).all()


def test_targets_are_valid(sample_raw):
    """Labels must be binary and the future price must exceed the current price."""
    clean, _ = clean_data(sample_raw)
    df, ref = build_training_frame(clean)
    assert set(df["Good_Investment"].unique()) <= {0, 1}
    assert (df["Future_Price_5Y"] > df["Price_in_Lakhs"]).all()
    assert df["Growth_Rate"].between(0.03, 0.12).all()


def test_inference_features_match_training(sample_raw):
    """Engineering a single row at inference must reproduce the training features."""
    clean, _ = clean_data(sample_raw)
    df, ref = build_training_frame(clean)
    row = engineer_features(clean.iloc[[0]], ref)
    assert row["Infra_Score"].iloc[0] == pytest.approx(df["Infra_Score"].iloc[0])
    assert row["PPS_vs_City_Median"].iloc[0] == pytest.approx(
        df["PPS_vs_City_Median"].iloc[0]
    )


def test_outlier_report_shape(sample_raw):
    """The IQR report must contain one row per requested column."""
    out = detect_outliers_iqr(sample_raw, ["Price_in_Lakhs", "Size_in_SqFt"])
    assert len(out) == 2
