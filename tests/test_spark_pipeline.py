"""Tests for the PySpark ETL pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pyspark.sql import functions as F  # noqa: N812

# Ensure project modules are importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "data"))

from spark_pipeline import (  # noqa: E402
    aggregate_monthly,
    clean_data,
    create_spark_session,
    load_raw_data,
    save_processed,
)


# ---------------------------------------------------------------------------
# create_spark_session
# ---------------------------------------------------------------------------

def test_create_spark_session(spark):
    """SparkSession creates successfully and is usable."""
    # Use the shared fixture; create_spark_session returns the singleton
    session = create_spark_session("TestCreateSession")
    assert session is not None
    assert session.sparkContext is not None
    # Verify the session can execute a trivial query
    result = session.sql("SELECT 1 AS value").collect()
    assert result[0]["value"] == 1
    # Do NOT stop — the session is shared via getOrCreate()


# ---------------------------------------------------------------------------
# load_raw_data
# ---------------------------------------------------------------------------

def test_load_raw_data(spark):
    """Loads actual CSVs, returns 15,181 rows with expected columns."""
    raw_dir = PROJECT_ROOT / "data" / "raw"
    df = load_raw_data(spark, raw_dir)

    assert df.count() == 15_181, f"Expected 15181 rows, got {df.count()}"
    assert "date" in df.columns, "'date' column missing after load"
    for col in ["subcategory", "sizing", "tactical", "quantity", "amount"]:
        assert col in df.columns, f"Expected column '{col}' missing"


# ---------------------------------------------------------------------------
# clean_data
# ---------------------------------------------------------------------------

def test_clean_data_preserves_na_tactical(synthetic_raw_df):
    """The string 'NA' in the tactical column is preserved, NOT converted to 'N'."""
    cleaned = clean_data(synthetic_raw_df)

    na_count = cleaned.filter(F.col("tactical") == "NA").count()
    assert na_count == 1, f"Expected 1 'NA' tactical row, got {na_count}"


def test_clean_data_fills_null_tactical(synthetic_raw_df):
    """True null tactical values are filled with 'N'."""
    # Verify the raw data has nulls
    raw_null_count = synthetic_raw_df.filter(F.col("tactical").isNull()).count()
    assert raw_null_count > 0, "Synthetic data should contain null tactical values"

    cleaned = clean_data(synthetic_raw_df)
    null_count = cleaned.filter(F.col("tactical").isNull()).count()
    assert null_count == 0, f"Expected 0 null tactical values after clean, got {null_count}"

    # The null rows should now be 'N'
    n_count = cleaned.filter(F.col("tactical") == "N").count()
    # Original 'N' rows (2) + filled nulls (2) = 4
    assert n_count == 4, f"Expected 4 'N' tactical rows (2 original + 2 filled), got {n_count}"


def test_clean_data_drops_constant_columns(synthetic_raw_df):
    """Constant/low-signal columns (type, category, trigger_action) are removed."""
    cleaned = clean_data(synthetic_raw_df)

    for col in ["type", "category", "trigger_action"]:
        assert col not in cleaned.columns, f"Column '{col}' should have been dropped"

    # Verify useful columns are retained
    for col in ["subcategory", "sizing", "tactical", "quantity", "amount"]:
        assert col in cleaned.columns, f"Column '{col}' should be retained"


# ---------------------------------------------------------------------------
# aggregate_monthly
# ---------------------------------------------------------------------------

def test_aggregate_monthly_quantity_conservation(spark, synthetic_raw_df):
    """Sum of quantity is conserved through aggregation (pre-zero-fill)."""
    cleaned = clean_data(synthetic_raw_df)
    raw_total = cleaned.agg(F.sum("quantity")).collect()[0][0]

    filled_df, agg_df = aggregate_monthly(cleaned, spark)
    agg_total = agg_df.agg(F.sum("quantity")).collect()[0][0]

    assert raw_total == agg_total, (
        f"Quantity not conserved: cleaned={raw_total}, aggregated={agg_total}"
    )


def test_aggregate_monthly_zero_fill(spark, synthetic_raw_df):
    """Zero-filled panel is complete: n_products * 60 months == total rows."""
    cleaned = clean_data(synthetic_raw_df)
    filled_df, _ = aggregate_monthly(cleaned, spark)

    n_products = filled_df.select("subcategory", "sizing", "tactical").distinct().count()
    expected_rows = n_products * 60  # 60 months from 2019-01 to 2023-12
    actual_rows = filled_df.count()

    assert actual_rows == expected_rows, (
        f"Panel incomplete: {n_products} products * 60 months = {expected_rows}, got {actual_rows}"
    )


def test_aggregate_monthly_zero_fill_values(spark, synthetic_raw_df):
    """Zero-filled rows have quantity=0 and amount=0.0."""
    cleaned = clean_data(synthetic_raw_df)
    filled_df, agg_df = aggregate_monthly(cleaned, spark)

    # Rows that exist only because of zero-fill should have quantity=0
    # Find rows from the filled panel that were NOT in the aggregated data
    zero_filled_rows = filled_df.filter(F.col("transaction_count") == 0)

    if zero_filled_rows.count() > 0:
        # All zero-filled rows should have quantity=0 and amount=0.0
        bad_qty = zero_filled_rows.filter(F.col("quantity") != 0).count()
        bad_amt = zero_filled_rows.filter(F.col("amount") != 0.0).count()

        assert bad_qty == 0, f"Found {bad_qty} zero-filled rows with non-zero quantity"
        assert bad_amt == 0, f"Found {bad_amt} zero-filled rows with non-zero amount"


# ---------------------------------------------------------------------------
# save_processed
# ---------------------------------------------------------------------------

def test_save_processed_creates_parquet(spark, synthetic_raw_df, tmp_path):
    """Parquet file is created at the expected path."""
    cleaned = clean_data(synthetic_raw_df)
    filled_df, _ = aggregate_monthly(cleaned, spark)

    output_path = tmp_path / "output_parquet"
    save_processed(filled_df, output_path)

    assert output_path.exists(), f"Parquet directory not created at {output_path}"
    # Spark writes Parquet as a directory with part files
    parquet_files = list(output_path.glob("*.parquet"))
    assert len(parquet_files) > 0, "No .parquet part files found in output directory"


# ---------------------------------------------------------------------------
# run_pipeline end-to-end
# ---------------------------------------------------------------------------

def test_run_pipeline_end_to_end(spark):
    """Full pipeline runs end-to-end: load -> clean -> aggregate -> verify output."""
    raw_dir = PROJECT_ROOT / "data" / "raw"

    # Load
    raw_df = load_raw_data(spark, raw_dir)
    assert raw_df.count() == 15_181

    # Clean
    clean_df = clean_data(raw_df)
    assert clean_df.count() == 15_181  # Cleaning doesn't drop rows
    assert "type" not in clean_df.columns

    # Aggregate + zero-fill
    filled_df, agg_df = aggregate_monthly(clean_df, spark)

    # Verify output shape
    n_products = filled_df.select("subcategory", "sizing", "tactical").distinct().count()
    assert filled_df.count() == n_products * 60

    # Verify expected columns in final output
    expected_cols = {"date", "subcategory", "sizing", "tactical", "quantity", "amount",
                     "barrel_length", "transaction_count", "avg_price"}
    actual_cols = set(filled_df.columns)
    assert expected_cols.issubset(actual_cols), (
        f"Missing columns: {expected_cols - actual_cols}"
    )

    # Verify no nulls in key columns
    for col in ["date", "subcategory", "sizing", "tactical", "quantity", "amount"]:
        null_count = filled_df.filter(F.col(col).isNull()).count()
        assert null_count == 0, f"Found {null_count} nulls in '{col}'"
