"""Tests for PySpark feature engineering functions."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest
from pyspark.sql import Window
from pyspark.sql import functions as F  # noqa: N812
from pyspark.sql.types import (
    DateType,
    DoubleType,
    FloatType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

# Ensure project modules are importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "features"))

from engineering import (  # noqa: E402
    _product_window,
    add_hunting_season_flags,
    add_lag_features,
    add_product_encodings,
    add_rolling_features,
    add_time_features,
    to_pandas,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_panel_df(spark):
    """Synthetic zero-filled panel: 3 products x 24 months (2019-01 to 2020-12).

    Products have non-overlapping quantity ranges for leakage detection:
      Pump Action / 12 GA / Y  : 100, 110, 120, …, 330
      Over Under  / 20 GA / NA :   1,   2,   3, …,  24
      Semi-Auto   / 10 GA / N  : 500, 505, 510, …, 615
    """
    months = [date(y, m, 1) for y in (2019, 2020) for m in range(1, 13)]

    rows = []
    for i, d in enumerate(months):
        # Product 1
        q1 = 100 + i * 10
        rows.append(
            (
                d,
                "Pump Action",
                "12 GA",
                "Y",
                q1,
                q1 * 50.0,
                50.0,
                20.0,
                max(1, q1 // 10),
            )
        )
        # Product 2
        q2 = 1 + i
        rows.append(
            (
                d,
                "Over Under",
                "20 GA",
                "NA",
                q2,
                q2 * 50.0,
                50.0,
                20.0,
                max(1, q2 // 10),
            )
        )
        # Product 3
        q3 = 500 + i * 5
        rows.append(
            (d, "Semi-Auto", "10 GA", "N", q3, q3 * 50.0, 50.0, 20.0, max(1, q3 // 10))
        )

    schema = StructType(
        [
            StructField("date", DateType(), False),
            StructField("subcategory", StringType(), False),
            StructField("sizing", StringType(), False),
            StructField("tactical", StringType(), False),
            StructField("quantity", IntegerType(), False),
            StructField("amount", DoubleType(), False),
            StructField("avg_price", DoubleType(), False),
            StructField("barrel_length", FloatType(), True),
            StructField("transaction_count", IntegerType(), False),
        ]
    )

    return spark.createDataFrame(rows, schema=schema)


# ---------------------------------------------------------------------------
# add_time_features
# ---------------------------------------------------------------------------


def test_add_time_features_columns_exist(synthetic_panel_df):
    result = add_time_features(synthetic_panel_df)
    expected = {
        "month",
        "quarter",
        "year",
        "month_sin",
        "month_cos",
        "covid_period",
        "post_covid",
        "is_march",
        "off_season",
        "months_to_hunting",
    }
    assert expected.issubset(set(result.columns)), (
        f"Missing columns: {expected - set(result.columns)}"
    )


def test_add_time_features_cyclical_range(synthetic_panel_df):
    result = add_time_features(synthetic_panel_df)
    stats = result.agg(
        F.min("month_sin").alias("sin_min"),
        F.max("month_sin").alias("sin_max"),
        F.min("month_cos").alias("cos_min"),
        F.max("month_cos").alias("cos_max"),
    ).collect()[0]

    assert stats["sin_min"] >= -1.0
    assert stats["sin_max"] <= 1.0
    assert stats["cos_min"] >= -1.0
    assert stats["cos_max"] <= 1.0


def test_add_time_features_covid_period(synthetic_panel_df):
    result = add_time_features(synthetic_panel_df)

    # covid_period=1 for dates 2020-03-01 through 2021-12-01
    bad_covid_on = result.filter(
        (F.col("covid_period") == 1)
        & ((F.col("date") < "2020-03-01") | (F.col("date") > "2021-12-01"))
    ).count()
    assert bad_covid_on == 0, (
        f"{bad_covid_on} rows incorrectly flagged as covid_period=1"
    )

    bad_covid_off = result.filter(
        (F.col("covid_period") == 0)
        & (F.col("date") >= "2020-03-01")
        & (F.col("date") <= "2021-12-01")
    ).count()
    assert bad_covid_off == 0, (
        f"{bad_covid_off} rows should be covid_period=1 but are 0"
    )


# ---------------------------------------------------------------------------
# add_hunting_season_flags
# ---------------------------------------------------------------------------


def test_add_hunting_season_flags_wrap_around(synthetic_panel_df):
    """Waterfowl season (Oct–Jan) wraps around the year boundary."""
    seasons = {"waterfowl": {"start_month": 10, "end_month": 1}}
    df = add_time_features(synthetic_panel_df)
    result = add_hunting_season_flags(df, seasons=seasons)

    on_months = {
        row["month"]
        for row in result.filter(F.col("is_waterfowl_season") == 1)
        .select("month")
        .distinct()
        .collect()
    }
    off_months = {
        row["month"]
        for row in result.filter(F.col("is_waterfowl_season") == 0)
        .select("month")
        .distinct()
        .collect()
    }

    assert on_months == {1, 10, 11, 12}, f"Expected {{1,10,11,12}}, got {on_months}"
    assert off_months == {2, 3, 4, 5, 6, 7, 8, 9}


def test_add_hunting_season_flags_intensity(synthetic_panel_df):
    """hunting_intensity equals the sum of all individual season flags."""
    df = add_time_features(synthetic_panel_df)
    result = add_hunting_season_flags(df)  # uses config's 6 seasons

    flag_cols = [
        c for c in result.columns if c.startswith("is_") and c.endswith("_season")
    ]
    assert len(flag_cols) == 6, f"Expected 6 season flags, got {len(flag_cols)}"

    sum_expr = F.col(flag_cols[0])
    for c in flag_cols[1:]:
        sum_expr = sum_expr + F.col(c)
    result = result.withColumn("expected_intensity", sum_expr)

    mismatch = result.filter(
        F.col("hunting_intensity") != F.col("expected_intensity")
    ).count()
    assert mismatch == 0, f"{mismatch} rows where hunting_intensity != sum of flags"


# ---------------------------------------------------------------------------
# add_lag_features
# ---------------------------------------------------------------------------


def test_add_lag_features_no_leakage(synthetic_panel_df):
    """Lag values come from the same product — no cross-product leakage."""
    df = add_time_features(synthetic_panel_df)
    # Must include lag 12 because add_lag_features hardcodes quantity_yoy_change
    result = add_lag_features(df, lags=[1, 12])

    for subcat, sizing, tact in [
        ("Pump Action", "12 GA", "Y"),
        ("Over Under", "20 GA", "NA"),
        ("Semi-Auto", "10 GA", "N"),
    ]:
        rows = (
            result.filter(
                (F.col("subcategory") == subcat)
                & (F.col("sizing") == sizing)
                & (F.col("tactical") == tact)
            )
            .orderBy("date")
            .select("quantity", "quantity_lag_1")
            .collect()
        )
        for i in range(1, len(rows)):
            lag_val = rows[i]["quantity_lag_1"]
            if lag_val is not None:
                assert lag_val == rows[i - 1]["quantity"], (
                    f"Leakage in ({subcat},{sizing},{tact}) row {i}: "
                    f"lag_1={lag_val} != prev_qty={rows[i - 1]['quantity']}"
                )


def test_add_lag_features_correct_values(synthetic_panel_df):
    """lag_1 matches the previous month's quantity for the same product."""
    df = add_time_features(synthetic_panel_df)
    result = add_lag_features(df, lags=[1, 12])

    rows = (
        result.filter(
            (F.col("subcategory") == "Pump Action")
            & (F.col("sizing") == "12 GA")
            & (F.col("tactical") == "Y")
        )
        .orderBy("date")
        .select("quantity", "quantity_lag_1")
        .collect()
    )

    assert rows[0]["quantity_lag_1"] is None, "First row should have null lag_1"
    for i in range(1, len(rows)):
        assert rows[i]["quantity_lag_1"] == rows[i - 1]["quantity"], (
            f"Row {i}: lag_1={rows[i]['quantity_lag_1']} != prev={rows[i - 1]['quantity']}"
        )


def test_add_lag_features_warmup_nulls(synthetic_panel_df):
    """First N rows per product have null lags for lag_N."""
    df = add_time_features(synthetic_panel_df)
    result = add_lag_features(df, lags=[1, 3, 6, 12])

    rows = (
        result.filter(
            (F.col("subcategory") == "Pump Action")
            & (F.col("sizing") == "12 GA")
            & (F.col("tactical") == "Y")
        )
        .orderBy("date")
        .collect()
    )

    # lag_1: first 1 row null
    assert rows[0]["quantity_lag_1"] is None
    assert rows[1]["quantity_lag_1"] is not None

    # lag_3: first 3 rows null
    for i in range(3):
        assert rows[i]["quantity_lag_3"] is None, f"Expected null lag_3 at row {i}"
    assert rows[3]["quantity_lag_3"] is not None

    # lag_12: first 12 rows null
    for i in range(12):
        assert rows[i]["quantity_lag_12"] is None, f"Expected null lag_12 at row {i}"
    assert rows[12]["quantity_lag_12"] is not None


# ---------------------------------------------------------------------------
# add_rolling_features
# ---------------------------------------------------------------------------


def test_add_rolling_features_backwards_only(synthetic_panel_df):
    """Rolling windows use only current and past rows — no future data."""
    df = add_time_features(synthetic_panel_df)
    result = add_rolling_features(df, windows=[3])

    rows = (
        result.filter(
            (F.col("subcategory") == "Pump Action")
            & (F.col("sizing") == "12 GA")
            & (F.col("tactical") == "Y")
        )
        .orderBy("date")
        .select("quantity", "quantity_ma_3")
        .collect()
    )

    # Row 0: window has 1 value → ma = qty itself
    assert abs(rows[0]["quantity_ma_3"] - rows[0]["quantity"]) < 0.01

    # Row 1: window has 2 values → ma = avg(row0, row1)
    expected = (rows[0]["quantity"] + rows[1]["quantity"]) / 2.0
    assert abs(rows[1]["quantity_ma_3"] - expected) < 0.01

    # Row 2: full window → avg(row0, row1, row2)
    expected = (rows[0]["quantity"] + rows[1]["quantity"] + rows[2]["quantity"]) / 3.0
    assert abs(rows[2]["quantity_ma_3"] - expected) < 0.01

    # Row 3: window slides → avg(row1, row2, row3), NOT row0
    expected = (rows[1]["quantity"] + rows[2]["quantity"] + rows[3]["quantity"]) / 3.0
    assert abs(rows[3]["quantity_ma_3"] - expected) < 0.01


def test_add_rolling_features_std_guard(synthetic_panel_df):
    """std returns 0.0 when the window has fewer than 2 rows."""
    df = add_time_features(synthetic_panel_df)
    result = add_rolling_features(df, windows=[3])

    rows = (
        result.filter(
            (F.col("subcategory") == "Pump Action")
            & (F.col("sizing") == "12 GA")
            & (F.col("tactical") == "Y")
        )
        .orderBy("date")
        .select("quantity_std_3")
        .collect()
    )

    # First row: only 1 data point in window → guarded to 0.0
    assert rows[0]["quantity_std_3"] == 0.0, (
        f"Expected std=0.0 for single-row window, got {rows[0]['quantity_std_3']}"
    )


# ---------------------------------------------------------------------------
# add_product_encodings
# ---------------------------------------------------------------------------


def test_add_product_encodings_rare_gauge_grouping(synthetic_panel_df):
    """Gauges not in _KEEP_GAUGES become OTHER_GAUGE."""
    result = add_product_encodings(synthetic_panel_df)

    # 10 GA is NOT in _KEEP_GAUGES → should become OTHER_GAUGE
    rare = result.filter(F.col("sizing") == "10 GA")
    grouped = {
        row["sizing_grouped"]
        for row in rare.select("sizing_grouped").distinct().collect()
    }
    assert grouped == {"OTHER_GAUGE"}, f"Expected {{'OTHER_GAUGE'}}, got {grouped}"

    # 12 GA IS in _KEEP_GAUGES → should stay as-is
    kept = result.filter(F.col("sizing") == "12 GA")
    kept_vals = {
        row["sizing_grouped"]
        for row in kept.select("sizing_grouped").distinct().collect()
    }
    assert kept_vals == {"12 GA"}

    # 20 GA IS in _KEEP_GAUGES → should stay as-is
    kept20 = result.filter(F.col("sizing") == "20 GA")
    kept20_vals = {
        row["sizing_grouped"]
        for row in kept20.select("sizing_grouped").distinct().collect()
    }
    assert kept20_vals == {"20 GA"}


def test_add_product_encodings_tactical_na(synthetic_panel_df):
    """The string 'NA' is encoded as is_tactical_na=1."""
    result = add_product_encodings(synthetic_panel_df)

    # Rows with tactical='NA' should have is_tactical_na=1
    na_rows = result.filter(F.col("tactical") == "NA")
    assert na_rows.count() > 0, "No tactical='NA' rows in fixture"
    bad_na = na_rows.filter(F.col("is_tactical_na") != 1).count()
    assert bad_na == 0, f"{bad_na} rows with tactical='NA' but is_tactical_na != 1"

    # Rows with tactical != 'NA' should have is_tactical_na=0
    non_na = result.filter(F.col("tactical") != "NA")
    bad_non_na = non_na.filter(F.col("is_tactical_na") != 0).count()
    assert bad_non_na == 0, (
        f"{bad_non_na} rows with tactical!='NA' but is_tactical_na != 0"
    )


# ---------------------------------------------------------------------------
# build_feature_set (column count, without file write)
# ---------------------------------------------------------------------------


def test_build_feature_set_column_count(synthetic_panel_df):
    """Full feature pipeline produces the expected number of columns."""
    # Replicate build_feature_set logic without the parquet write
    df = add_time_features(synthetic_panel_df)
    df = add_hunting_season_flags(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_product_encodings(df)

    # Cross-product features (same as build_feature_set)
    subcat_window = Window.partitionBy("subcategory", "date")
    df = df.withColumn("subcat_total_qty", F.sum("quantity").over(subcat_window))
    df = df.withColumn(
        "subcat_share",
        F.when(
            F.col("subcat_total_qty") > 0, F.col("quantity") / F.col("subcat_total_qty")
        ).otherwise(F.lit(0.0)),
    )
    df = df.withColumn("time_index", F.row_number().over(_product_window()) - 1)

    n_cols = len(df.columns)
    # 9 original + 10 time + 7 hunting + 7 lag + 6 rolling
    # + product encodings (sizing_grouped, is_tactical, is_tactical_na, subcategory dummies, is_12ga)
    # + 3 build extras (subcat_total_qty, subcat_share, time_index)
    # With 3 subcategories → 2 dummies → ~48 columns
    assert 45 <= n_cols <= 55, (
        f"Expected 45-55 columns, got {n_cols}: {sorted(df.columns)}"
    )


# ---------------------------------------------------------------------------
# to_pandas
# ---------------------------------------------------------------------------


def test_to_pandas_drops_warmup(synthetic_panel_df):
    """Warm-up rows with NaN lags are dropped."""
    df = add_time_features(synthetic_panel_df)
    df = add_lag_features(df, lags=[1, 3, 6, 12])

    spark_count = df.count()  # 3 products × 24 months = 72
    pdf = to_pandas(df, drop_na=True)

    # Max lag is 12 → first 12 rows per product are dropped
    expected_dropped = 3 * 12  # 3 products × 12 warm-up rows
    assert len(pdf) == spark_count - expected_dropped, (
        f"Expected {spark_count - expected_dropped} rows, got {len(pdf)}"
    )
    assert len(pdf) < spark_count, "to_pandas should have dropped warm-up rows"


def test_to_pandas_no_nans(synthetic_panel_df):
    """Zero NaN values in the output after dropping warm-up rows."""
    df = add_time_features(synthetic_panel_df)
    df = add_lag_features(df, lags=[1, 3, 6, 12])
    df = add_rolling_features(df, windows=[3, 6, 12])

    pdf = to_pandas(df, drop_na=True)

    total_nans = pdf.isna().sum().sum()
    assert total_nans == 0, (
        f"Found {total_nans} NaN values in output; "
        f"columns with NaN: {pdf.columns[pdf.isna().any()].tolist()}"
    )
