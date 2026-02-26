# Shotgun Demand Forecasting

**CIS 531/731 — Programming Techniques for Data Science and Analytics**
**Kansas State University | Dr. Hsu**
**Author: Brent Showalter**

## Problem Statement

SKU-level demand forecasting for DLX shotgun product lines. The goal is multi-step time-series regression to predict monthly unit sales at the product level (subcategory x gauge x tactical), enabling improved inventory planning and reduced stockout risk for firearms distributors.

The project uses 5 years of transactional sales data (2019–2023), applies PySpark for ETL and feature engineering, trains baseline and advanced gradient-boosting models, and evaluates them via rolling-origin cross-validation with paired statistical testing.

## Directory Structure

| Path | Purpose |
|------|---------|
| `config/settings.yaml` | All configuration: seeds, model params, paths, hunting seasons |
| `data/raw/` | Raw yearly CSV transaction files (2019–2023) |
| `data/processed/` | PySpark pipeline output (Parquet) |
| `src/data/spark_pipeline.py` | PySpark ETL: load, clean, aggregate, save |
| `src/data/data_splitter.py` | Rolling-origin CV, temporal splits, and active product filtering |
| `src/features/engineering.py` | PySpark feature engineering: time, seasons, lags, rolling, encodings |
| `src/models/baselines.py` | Seasonal Naive, Ridge Regression, SARIMA |
| `src/models/xgboost_model.py` | XGBoost + Optuna hyperparameter optimization |
| `src/models/lightgbm_model.py` | LightGBM + Optuna hyperparameter optimization |
| `src/evaluation/metrics.py` | RMSE, MAE, MAPE computation and fold aggregation |
| `src/evaluation/statistical_tests.py` | Paired t-test and hypothesis report formatting |
| `src/visualization/plots.py` | Publication-quality plots for the final report |
| `notebooks/` | Jupyter notebooks for exploration and experiments |
| `deliverables/` | Plain-text content for proposal, report, and presentation |

## Environment Setup

### Prerequisites

- Python 3.13
- Java 17 (required by PySpark — Java 23 has security manager incompatibility)

### Installation

```bash
# Clone the repository
git clone https://github.com/brentthomas248/shotgun-demand-forecasting.git
cd shotgun-demand-forecasting

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip3 install -r requirements.txt
```

### PySpark Local Mode

PySpark runs in local mode — no cluster setup needed. The pipeline creates a SparkSession configured for the local machine. Ensure Java is installed:

```bash
java -version  # Should report 17.x
```

If Java is not installed on macOS:

```bash
brew install openjdk@17
export JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home
```

## How to Run

### 1. Data Pipeline (Sprint 1)

Run the PySpark ETL pipeline to load, clean, and aggregate raw data:

```bash
python3 -m src.data.spark_pipeline
```

This reads CSVs from `data/raw/`, processes them, and writes Parquet to `data/processed/`.

### 2. Feature Engineering (Sprint 2)

Feature engineering runs within the pipeline or can be invoked separately in notebooks. See `notebooks/02_pyspark_pipeline.ipynb` for the full flow.

### 3. Train Baseline Models (Sprint 3)

Train baseline models across rolling-origin CV folds. Products with fewer than 12 months of non-zero demand are automatically excluded (19 of 74 SKUs removed, 100% volume retained):

```bash
# Run from the model training notebook
jupyter notebook notebooks/04_model_training.ipynb
```

### 4. Train Alternative Models (Sprint 4)

Train XGBoost and LightGBM with Optuna optimization, then run statistical tests:

```bash
# Continue in the model training notebook
jupyter notebook notebooks/04_model_training.ipynb
```

### 5. Generate Visualizations (Sprint 5)

Create all publication-quality figures:

```bash
jupyter notebook notebooks/05_evaluation_analysis.ipynb
```

## Reproducing Experiments

All configuration is centralized in `config/settings.yaml`:

- **Random seed:** 42 (used across all models and splits)
- **CV parameters:** min_train=24 months, horizon=3 months, step=3 months
- **Optuna:** 50 trials, minimize MAPE
- **Statistical test:** Paired t-test, alpha=0.05

To reproduce results exactly:

1. Use the same Python version (3.13) and dependency versions from `requirements.txt`
2. Ensure `config/settings.yaml` is unchanged
3. Run the pipeline and notebooks in order (01 through 05)

## Models

| Role | Model | Description |
|------|-------|-------------|
| Baseline 1 | Seasonal Naive | Same month last year |
| Baseline 2 | Ridge Regression | Linear model on engineered features (alpha=1.0) |
| Baseline 3 | SARIMA | Classical time-series: order (1,1,1), seasonal (1,1,1,12) |
| Alternative 1 | XGBoost + Optuna | Gradient boosting with Bayesian hyperparameter optimization |
| Alternative 2 | LightGBM + Optuna | Second GBM variant for robustness |

## Results Summary

Mean metrics across 8 rolling-origin CV folds:

| Model | RMSE | MAE | MAPE |
|-------|------|-----|------|
| Seasonal Naive | 1735.59 | 534.17 | 907.34% |
| Ridge | 737.12 | 357.52 | 1255.55% |
| SARIMA | 283160.21 | 23288.40 | 11190.90% |
| **XGBoost** | **646.23** | **289.60** | 938.33% |
| LightGBM | 672.95 | 309.10 | 1062.30% |

Paired t-tests on RMSE show XGBoost significantly outperforms Ridge (p=0.042) and Seasonal Naive (p=0.022). MAPE-based tests did not reach significance due to high variance from sparse demand data.

> **Note:** MAPE values are inflated by low-volume SKUs. RMSE and MAE provide more reliable accuracy measures for this dataset.

## Sprint Status

| Sprint | Description | Status |
|--------|-------------|--------|
| 1 | PySpark Data Pipeline | Complete |
| 2 | Feature Engineering | Complete |
| 3 | Baseline Models + Rolling-Origin CV | Complete |
| 4 | Alternative Models + Statistical Testing | Complete |
| 5 | Visualization, Demo, Deliverables | Complete |

