"""PySpark feature engineering for shotgun demand forecasting.

Builds the full feature set using PySpark Window functions for lag and
rolling statistics, then converts to pandas for downstream ML training.
Features include temporal encodings, hunting-season flags, lag/rolling
aggregates, and product-level one-hot encodings.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F  # noqa: N812

from config import load_config

if TYPE_CHECKING:
    import pandas as pd
    from pyspark.sql.window import WindowSpec

logger = logging.getLogger(__name__)

# Gauges to keep as distinct categories; everything else -> OTHER_GAUGE
_KEEP_GAUGES = {"12 GA", "20 GA", "410 GA", "28 GA", "16 GA", "45/410 GA"}


def _product_window() -> WindowSpec:
    """Return the product-level window partitioned by composite key, ordered by date."""
    return Window.partitionBy("subcategory", "sizing", "tactical").orderBy("date")


def add_time_features(df: DataFrame) -> DataFrame:
    """Add temporal features: month, quarter, year, cyclical encoding, and domain events.

    Cyclical encoding uses sin/cos transforms so that December (12) and
    January (1) are numerically close.  Domain event flags capture the
    COVID structural break, March peak, off-season trough, and distance
    to the October hunting-season opener.

    Args:
        df: Spark DataFrame with a ``date`` column.

    Returns:
        DataFrame with additional time-feature columns.
    """
    # Calendar extractions
    df = (
        df.withColumn("month", F.month("date"))
        .withColumn("quarter", F.quarter("date"))
        .withColumn("year", F.year("date"))
    )

    # Cyclical encoding
    df = df.withColumn(
        "month_sin", F.sin(2 * math.pi * F.col("month") / 12)
    ).withColumn("month_cos", F.cos(2 * math.pi * F.col("month") / 12))

    # Domain event flags
    df = (
        df.withColumn(
            "covid_period",
            F.when(
                (F.col("date") >= F.lit("2020-03-01"))
                & (F.col("date") <= F.lit("2021-12-01")),
                F.lit(1),
            ).otherwise(F.lit(0)),
        )
        .withColumn(
            "post_covid",
            F.when(F.col("date") >= F.lit("2022-01-01"), F.lit(1)).otherwise(F.lit(0)),
        )
        .withColumn(
            "is_march",
            F.when(F.col("month") == 3, F.lit(1)).otherwise(F.lit(0)),
        )
        .withColumn(
            "off_season",
            F.when(F.col("month").isin(6, 7, 8), F.lit(1)).otherwise(F.lit(0)),
        )
        .withColumn(
            "months_to_hunting",
            ((F.lit(22) - F.col("month")) % F.lit(12)).cast("int"),
        )
    )

    logger.info("Added time features: month, quarter, year, cyclical, domain events")
    return df


def add_hunting_season_flags(
    df: DataFrame,
    seasons: dict[str, dict[str, int]] | None = None,
) -> DataFrame:
    """Add binary hunting-season indicators and composite intensity score.

    Handles wrap-around seasons (e.g., waterfowl Oct-Jan) where
    ``start_month > end_month`` by using OR logic.

    Args:
        df: Spark DataFrame with a ``month`` column (or derivable from
            ``date``).
        seasons: Mapping of season name to {start_month, end_month}.
            Defaults to values in config/settings.yaml.

    Returns:
        DataFrame with hunting-season flag columns appended.
    """
    if seasons is None:
        config = load_config()
        seasons = config["hunting_seasons"]

    # Ensure month column exists
    if "month" not in df.columns:
        df = df.withColumn("month", F.month("date"))

    flag_cols: list[str] = []
    for name, bounds in seasons.items():
        col_name = f"is_{name}_season"
        start = bounds["start_month"]
        end = bounds["end_month"]

        if start > end:
            # Wrap-around (e.g., Oct-Jan): month >= start OR month <= end
            condition = (F.col("month") >= start) | (F.col("month") <= end)
        else:
            # Normal range (e.g., Mar-May): start <= month <= end
            condition = (F.col("month") >= start) & (F.col("month") <= end)

        df = df.withColumn(col_name, F.when(condition, F.lit(1)).otherwise(F.lit(0)))
        flag_cols.append(col_name)

    # Composite hunting intensity = sum of all season flags
    intensity = F.col(flag_cols[0])
    for c in flag_cols[1:]:
        intensity = intensity + F.col(c)
    df = df.withColumn("hunting_intensity", intensity)

    logger.info("Added hunting season flags: %s + hunting_intensity", flag_cols)
    return df


def add_lag_features(
    df: DataFrame,
    lags: list[int] | None = None,
    target: str = "quantity",
) -> DataFrame:
    """Add lagged values of the target using PySpark Window functions.

    Creates ``{target}_lag_{n}`` for each lag, ``avg_price_lag_1``, and
    derived year-over-year / month-over-month change columns.

    Args:
        df: Spark DataFrame partitioned by product group with ``date`` order.
        lags: List of lag periods in months.  Defaults to config value.
        target: Column name to lag.

    Returns:
        DataFrame with ``{target}_lag_{n}`` columns appended.
    """
    if lags is None:
        config = load_config()
        lags = config["features"]["lags"]

    for n in lags:
        df = df.withColumn(
            f"{target}_lag_{n}",
            F.lag(target, n).over(_product_window()),
        )

    # Price lag
    df = df.withColumn(
        "avg_price_lag_1",
        F.lag("avg_price", 1).over(_product_window()),
    )

    # Derived change features
    df = df.withColumn(
        "quantity_yoy_change",
        F.col("quantity") - F.col("quantity_lag_12"),
    )
    df = df.withColumn(
        "quantity_mom_change",
        F.col("quantity") - F.col("quantity_lag_1"),
    )

    logger.info(
        "Added lag features: lags=%s, target=%s, + avg_price_lag_1, yoy/mom change",
        lags,
        target,
    )
    return df


def add_rolling_features(
    df: DataFrame,
    windows: list[int] | None = None,
    target: str = "quantity",
) -> DataFrame:
    """Add rolling mean and standard deviation using PySpark Window functions.

    Uses backwards-looking windows via ``rowsBetween(-(w-1), 0)``.
    Standard deviation is guarded with a minimum count of 2 to avoid
    undefined values for single-row windows.

    Args:
        df: Spark DataFrame partitioned by product group with ``date`` order.
        windows: List of rolling-window sizes in months.  Defaults to config.
        target: Column name for rolling statistics.

    Returns:
        DataFrame with ``{target}_ma_{w}`` and ``{target}_std_{w}`` columns.
    """
    if windows is None:
        config = load_config()
        windows = config["features"]["rolling_windows"]

    for w in windows:
        rolling_w = _product_window().rowsBetween(-(w - 1), 0)
        df = df.withColumn(
            f"{target}_ma_{w}",
            F.avg(target).over(rolling_w),
        )
        df = df.withColumn(
            f"{target}_std_{w}",
            F.when(
                F.count("*").over(rolling_w) >= 2,
                F.stddev(target).over(rolling_w),
            ).otherwise(F.lit(0.0)),
        )

    logger.info("Added rolling features: windows=%s, target=%s", windows, target)
    return df


def add_product_encodings(df: DataFrame) -> DataFrame:
    """Encode categorical product columns into numeric features.

    Steps:
        1. Group rare gauges into ``OTHER_GAUGE`` (keep top 6 gauges).
        2. Create ``is_tactical`` and ``is_tactical_na`` binary flags.
        3. One-hot encode subcategory (drop most-common as reference).
        4. Create ``is_12ga`` convenience flag.

    Args:
        df: Spark DataFrame with categorical product columns.

    Returns:
        DataFrame with encoded columns appended.
    """
    # 1. Group rare gauges
    df = df.withColumn(
        "sizing_grouped",
        F.when(F.col("sizing").isin(*_KEEP_GAUGES), F.col("sizing")).otherwise(
            F.lit("OTHER_GAUGE")
        ),
    )

    # 2. Tactical flags
    df = df.withColumn(
        "is_tactical",
        F.when(F.col("tactical") == "Y", F.lit(1)).otherwise(F.lit(0)),
    )
    df = df.withColumn(
        "is_tactical_na",
        F.when(F.col("tactical") == "NA", F.lit(1)).otherwise(F.lit(0)),
    )

    # 3. One-hot encode subcategory (drop most common as reference)
    subcat_counts = (
        df.groupBy("subcategory")
        .agg(F.sum("quantity").alias("total_qty"))
        .orderBy(F.desc("total_qty"))
        .collect()
    )
    most_common = subcat_counts[0]["subcategory"]
    other_subcats = [row["subcategory"] for row in subcat_counts[1:]]

    for subcat in other_subcats:
        col_name = "is_" + subcat.lower().replace(" ", "_").replace("/", "_")
        df = df.withColumn(
            col_name,
            F.when(F.col("subcategory") == subcat, F.lit(1)).otherwise(F.lit(0)),
        )

    # 4. Gauge convenience flag
    df = df.withColumn(
        "is_12ga",
        F.when(F.col("sizing") == "12 GA", F.lit(1)).otherwise(F.lit(0)),
    )

    logger.info(
        "Added product encodings: sizing_grouped, tactical flags, "
        "subcategory one-hot (reference=%s, %d dummies), is_12ga",
        most_common,
        len(other_subcats),
    )
    return df


def build_feature_set(df: DataFrame) -> DataFrame:
    """Apply the full feature-engineering pipeline in order.

    Chains all feature functions, adds cross-product features
    (subcategory-level demand and share), adds a per-product time index,
    and saves the result to ``data/features/`` as Parquet.

    Args:
        df: Raw monthly aggregated Spark DataFrame.

    Returns:
        Fully featured Spark DataFrame (may contain NaN rows from lagging).
    """
    # Chain feature functions
    df = add_time_features(df)
    df = add_hunting_season_flags(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_product_encodings(df)

    # Cross-product features
    subcat_window = Window.partitionBy("subcategory", "date")
    df = df.withColumn("subcat_total_qty", F.sum("quantity").over(subcat_window))
    df = df.withColumn(
        "subcat_share",
        F.when(
            F.col("subcat_total_qty") > 0,
            F.col("quantity") / F.col("subcat_total_qty"),
        ).otherwise(F.lit(0.0)),
    )

    # Per-product time index (0-based)
    df = df.withColumn(
        "time_index",
        F.row_number().over(_product_window()) - 1,
    )

    # Save to data/features/ as Parquet
    project_root = Path(__file__).resolve().parent.parent.parent
    config = load_config()
    features_path = project_root / config["paths"]["data_features"]
    features_path.mkdir(parents=True, exist_ok=True)

    (
        df.orderBy("date", "subcategory", "sizing", "tactical")
        .coalesce(1)
        .write.mode("overwrite")
        .parquet(str(features_path))
    )
    logger.info("Saved feature set to %s", features_path)

    return df


def to_pandas(df: DataFrame, drop_na: bool = True) -> pd.DataFrame:
    """Convert a PySpark DataFrame to pandas, optionally dropping NaN rows.

    Drops rows where any lag column contains NaN (warm-up period from
    the 12-month maximum lag).

    Args:
        df: Spark DataFrame to convert.
        drop_na: If True, drop rows with any NaN values in lag columns.

    Returns:
        pandas DataFrame ready for model training.
    """
    pdf = df.toPandas()
    logger.info("Converted to pandas: shape=%s", pdf.shape)

    if drop_na:
        lag_cols = [c for c in pdf.columns if "_lag_" in c]
        before = len(pdf)
        pdf = pdf.dropna(subset=lag_cols)
        logger.info("Dropped %d warm-up rows (NaN in lag columns)", before - len(pdf))

    logger.info("Final pandas shape: %s", pdf.shape)
    return pdf
