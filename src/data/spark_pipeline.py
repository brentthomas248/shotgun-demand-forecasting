"""PySpark ETL pipeline for shotgun demand forecasting.

Loads raw CSV transaction data into PySpark, cleans and validates records,
aggregates to monthly product-level totals, and saves processed output as
Parquet. This is the primary data ingestion layer required by the CIS 731
course (MapReduce/Spark requirement).
"""
from __future__ import annotations

import logging
from functools import reduce
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F  # noqa: N812
from pyspark.sql.types import (
    FloatType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

RAW_SCHEMA = StructType([
    StructField("type", StringType(), True),
    StructField("category", StringType(), True),
    StructField("subcategory", StringType(), True),
    StructField("sizing", StringType(), True),
    StructField("month_name", StringType(), True),
    StructField("month_number", IntegerType(), True),
    StructField("year_", IntegerType(), True),
    StructField("tactical", StringType(), True),
    StructField("barrel_length", FloatType(), True),
    StructField("trigger_action", StringType(), True),
    StructField("quantity", IntegerType(), True),
    StructField("amount", FloatType(), True),
])


def create_spark_session(app_name: str = "ShotgunDemandETL") -> SparkSession:
    """Create and configure a local Spark session.

    Args:
        app_name: Name for the Spark application.

    Returns:
        Configured SparkSession instance.
    """
    spark = (
        SparkSession.builder
        .master("local[*]")
        .appName(app_name)
        .config("spark.driver.memory", "2g")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def load_raw_data(
    spark: SparkSession,
    raw_dir: Path,
    years: list[int] | None = None,
    config: dict | None = None,
) -> DataFrame:
    """Load yearly shotgun CSV files into a single Spark DataFrame.

    Reads each year's CSV using an explicit StructType schema (no inferSchema),
    unions them, and constructs a date column from year_ and month_number.

    Args:
        spark: Active SparkSession.
        raw_dir: Path to directory containing raw CSV files.
        years: List of years to load. Defaults to range from config or 2019-2023.
        config: Optional config dict for deriving default years.

    Returns:
        Combined Spark DataFrame with all years and a constructed date column.
    """
    if years is None:
        if config is not None:
            start = config["data"]["years"]["start"]
            end = config["data"]["years"]["end"]
            years = list(range(start, end + 1))
        else:
            years = [2019, 2020, 2021, 2022, 2023]

    dfs: list[DataFrame] = []
    for year in years:
        path = str(raw_dir / f"dlx_shotgun_datadump_firearms_{year}.csv")
        df = spark.read.csv(path, schema=RAW_SCHEMA, header=True)
        dfs.append(df)
        logger.info("Loaded %s: %d rows", path, df.count())

    combined = reduce(DataFrame.unionByName, dfs)
    combined = combined.withColumn(
        "date", F.make_date(F.col("year_"), F.col("month_number"), F.lit(1))
    )
    logger.info("Combined raw data: %d rows", combined.count())
    return combined


def clean_data(df: DataFrame) -> DataFrame:
    """Clean raw data: standardise tactical values and drop low-signal columns.

    Applies the following transformations:
    - Fill only truly null tactical values with 'N' (non-tactical).
      The string "NA" is preserved as a valid third category (764 rows).
    - Drop constant/low-signal columns: type, category, month_name,
      trigger_action, month_number, year_ (redundant after date construction).
    - Validate that quantities are non-negative; log any violations.

    Args:
        df: Raw Spark DataFrame with a date column already constructed.

    Returns:
        Cleaned Spark DataFrame.
    """
    # Fill only truly null tactical values — preserve "NA" as a valid category
    df = df.withColumn(
        "tactical",
        F.when(F.col("tactical").isNull(), F.lit("N"))
        .otherwise(F.col("tactical")),
    )

    # Drop constant / low-signal / redundant columns
    cols_to_drop = ["type", "category", "month_name", "trigger_action", "month_number", "year_"]
    df = df.drop(*cols_to_drop)

    # Validate non-negative quantities
    neg_count = df.filter(F.col("quantity") < 0).count()
    if neg_count > 0:
        logger.warning("Found %d rows with negative quantity", neg_count)
    else:
        logger.info("Quantity validation passed: no negative values")

    logger.info("Cleaned data: %d rows, columns: %s", df.count(), df.columns)
    return df


def aggregate_monthly(df: DataFrame, spark: SparkSession) -> DataFrame:
    """Aggregate transaction data to monthly product-level totals with zero-fill.

    Groups by (date, subcategory, sizing, tactical) and computes:
    - sum(quantity), sum(amount), avg(barrel_length), count(*) as transaction_count
    - Derived: avg_price = amount / quantity (0.0 when quantity is 0)

    Then zero-fills the panel by cross-joining a complete date spine with all
    distinct products, left-joining actuals, and filling nulls.

    Args:
        df: Cleaned Spark DataFrame with a date column.
        spark: Active SparkSession (needed for date spine SQL).

    Returns:
        Monthly aggregated, zero-filled Spark DataFrame.
    """
    # Aggregate
    agg_df = (
        df.groupBy("date", "subcategory", "sizing", "tactical")
        .agg(
            F.sum("quantity").alias("quantity"),
            F.sum("amount").alias("amount"),
            F.avg("barrel_length").alias("barrel_length"),
            F.count("*").alias("transaction_count"),
        )
        .withColumn(
            "avg_price",
            F.when(F.col("quantity") > 0, F.col("amount") / F.col("quantity"))
            .otherwise(F.lit(0.0)),
        )
    )
    logger.info("Aggregated data: %d rows", agg_df.count())

    # Zero-fill the panel
    # 1. Date spine: all months from 2019-01 to 2023-12
    date_spine = spark.sql(
        "SELECT explode(sequence(to_date('2019-01-01'), to_date('2023-12-01'), interval 1 month)) as date"
    )

    # 2. Distinct products
    products = agg_df.select("subcategory", "sizing", "tactical").distinct()
    logger.info("Distinct products: %d", products.count())

    # 3. Cross-join for complete panel grid
    panel = date_spine.crossJoin(products)

    # 4. Left-join actuals onto grid
    filled = panel.join(
        agg_df,
        on=["date", "subcategory", "sizing", "tactical"],
        how="left",
    )

    # 5. Fill nulls for numeric columns
    filled = filled.fillna({
        "quantity": 0,
        "amount": 0.0,
        "transaction_count": 0,
        "avg_price": 0.0,
    })

    # 6. Fill barrel_length nulls with per-product mean, then global mean
    product_avg_bl = (
        agg_df.filter(F.col("barrel_length").isNotNull())
        .groupBy("subcategory", "sizing", "tactical")
        .agg(F.avg("barrel_length").alias("product_avg_bl"))
    )
    filled = filled.join(
        product_avg_bl,
        on=["subcategory", "sizing", "tactical"],
        how="left",
    )
    filled = filled.withColumn(
        "barrel_length",
        F.coalesce(F.col("barrel_length"), F.col("product_avg_bl")),
    )

    # Global mean for any remaining nulls
    global_avg_bl = (
        agg_df.filter(F.col("barrel_length").isNotNull())
        .agg(F.avg("barrel_length"))
        .collect()[0][0]
    )
    filled = filled.withColumn(
        "barrel_length",
        F.coalesce(F.col("barrel_length"), F.lit(global_avg_bl)),
    ).drop("product_avg_bl")

    logger.info("Zero-filled panel: %d rows", filled.count())
    return filled, agg_df


def save_processed(df: DataFrame, output_path: Path) -> None:
    """Save processed DataFrame as Parquet.

    Args:
        df: Processed Spark DataFrame to persist.
        output_path: Destination path for the Parquet file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    (
        df.orderBy("date", "subcategory", "sizing", "tactical")
        .coalesce(1)
        .write.mode("overwrite")
        .parquet(str(output_path))
    )
    logger.info("Saved processed data to %s", output_path)


def validate_pipeline(
    raw_df: DataFrame,
    clean_df: DataFrame,
    agg_df: DataFrame,
    filled_df: DataFrame,
    config: dict,
) -> None:
    """Assert data quality invariants at each pipeline stage.

    Args:
        raw_df: Raw loaded DataFrame.
        clean_df: Cleaned DataFrame.
        agg_df: Aggregated DataFrame (pre-zero-fill).
        filled_df: Zero-filled panel DataFrame.
        config: Project configuration dict.
    """
    expected_raw = config["pipeline"]["expected_raw_rows"]
    total_months = config["pipeline"]["total_months"]

    # Raw row count
    raw_count = raw_df.count()
    assert raw_count == expected_raw, f"Raw row count mismatch: expected {expected_raw}, got {raw_count}"
    logger.info("VALIDATE: Raw row count OK (%d)", raw_count)

    # Quantity conservation: raw total == aggregated total
    raw_total = raw_df.agg(F.sum("quantity")).collect()[0][0]
    agg_total = agg_df.agg(F.sum("quantity")).collect()[0][0]
    assert raw_total == agg_total, f"Quantity conservation failed: raw={raw_total}, agg={agg_total}"
    logger.info("VALIDATE: Quantity conservation OK (total=%d)", raw_total)

    # No negative quantities in cleaned data
    neg_count = clean_df.filter(F.col("quantity") < 0).count()
    assert neg_count == 0, f"Found {neg_count} negative quantities in cleaned data"
    logger.info("VALIDATE: No negative quantities OK")

    # Panel dimensions: n_products * total_months == filled row count
    n_products = filled_df.select("subcategory", "sizing", "tactical").distinct().count()
    filled_count = filled_df.count()
    expected_filled = n_products * total_months
    assert filled_count == expected_filled, (
        f"Panel dimension mismatch: {n_products} products * {total_months} months = "
        f"{expected_filled}, got {filled_count}"
    )
    logger.info("VALIDATE: Panel dimensions OK (%d products x %d months = %d rows)", n_products, total_months, filled_count)

    # No nulls in key columns
    key_cols = ["date", "subcategory", "sizing", "tactical", "quantity", "amount"]
    for col in key_cols:
        null_count = filled_df.filter(F.col(col).isNull()).count()
        assert null_count == 0, f"Found {null_count} null values in column '{col}'"
    logger.info("VALIDATE: No nulls in key columns OK")

    logger.info("All validations passed!")


def run_pipeline(config_path: Path | None = None) -> None:
    """Execute the full ETL pipeline end-to-end.

    Orchestrates: create_spark_session -> load_raw_data -> clean_data ->
    aggregate_monthly -> save_processed, with validation at each stage.

    Args:
        config_path: Optional path to settings.yaml. Defaults to
            config/settings.yaml relative to project root.
    """
    from config import load_config

    project_root = Path(__file__).resolve().parent.parent.parent
    config = load_config(str(project_root / "config" / "settings.yaml") if config_path is None else str(config_path))

    raw_dir = project_root / config["paths"]["data_raw"]
    output_path = project_root / config["paths"]["data_processed"]

    spark = create_spark_session()
    try:
        # Load
        logger.info("=== Loading raw data ===")
        raw_df = load_raw_data(spark, raw_dir, config=config)

        # Clean
        logger.info("=== Cleaning data ===")
        clean_df = clean_data(raw_df)

        # Aggregate + zero-fill
        logger.info("=== Aggregating monthly ===")
        filled_df, agg_df = aggregate_monthly(clean_df, spark)

        # Validate
        logger.info("=== Validating pipeline ===")
        validate_pipeline(raw_df, clean_df, agg_df, filled_df, config)

        # Save
        logger.info("=== Saving processed data ===")
        save_processed(filled_df, output_path)

        logger.info("Pipeline complete!")
    finally:
        spark.stop()
        logger.info("SparkSession stopped")


if __name__ == "__main__":
    run_pipeline()
