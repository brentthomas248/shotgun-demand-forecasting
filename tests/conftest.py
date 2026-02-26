"""Shared test fixtures for the Spark pipeline test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F  # noqa: N812
from pyspark.sql.types import (
    FloatType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

# Ensure project src is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "data"))


@pytest.fixture(scope="session")
def spark() -> SparkSession:
    """Create a SparkSession shared across the entire test session."""
    session = (
        SparkSession.builder
        .master("local[*]")
        .appName("TestShotgunPipeline")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow")
        .config("spark.executor.extraJavaOptions", "-Djava.security.manager=allow")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


@pytest.fixture()
def synthetic_raw_df(spark: SparkSession):
    """Create a small synthetic DataFrame matching the raw CSV schema.

    Includes a mix of tactical values: 'Y', 'N', 'NA', and null to test
    cleaning logic thoroughly.
    """
    schema = StructType([
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

    data = [
        ("Firearms", "Shotgun", "Pump Action", "12 GA", "Jan", 1, 2019, "Y", 20.0, "OTHER", 100, 5000.0),
        ("Firearms", "Shotgun", "Pump Action", "12 GA", "Feb", 2, 2019, "N", 26.0, "OTHER", 200, 10000.0),
        ("Firearms", "Shotgun", "Pump Action", "12 GA", "Mar", 3, 2019, "NA", 28.0, "OTHER", 50, 2500.0),
        ("Firearms", "Shotgun", "Pump Action", "20 GA", "Jan", 1, 2019, None, 24.0, "OTHER", 80, 4000.0),
        ("Firearms", "Shotgun", "Over Under", "12 GA", "Jan", 1, 2019, "Y", 30.0, "OTHER", 150, 12000.0),
        ("Firearms", "Shotgun", "Over Under", "12 GA", "Feb", 2, 2019, None, 28.0, "OTHER", 120, 9600.0),
        ("Firearms", "Shotgun", "Over Under", "20 GA", "Mar", 3, 2019, "N", 26.0, "OTHER", 90, 4500.0),
        ("Firearms", "Shotgun", "Pump Action", "12 GA", "Apr", 4, 2019, "Y", 20.0, "OTHER", 110, 5500.0),
    ]

    df = spark.createDataFrame(data, schema=schema)
    # Add date column the same way the pipeline does
    df = df.withColumn(
        "date", F.make_date(F.col("year_"), F.col("month_number"), F.lit(1))
    )
    return df
