# Shotgun Demand Forecasting Using Machine Learning
## CIS 531/731 Term Project Presentation

---

## Slide 1: Title

**Shotgun Demand Forecasting Using Machine Learning: A Comparative Study**

- Brent Showalter
- CIS 531/731 — Data Mining
- Kansas State University
- Spring 2026

---

## Slide 2: Problem Statement

**Why demand forecasting matters for firearms distributors**

- U.S. shotgun market driven by seasonal hunting cycles (Sep–Jan peak)
- Distributors face dual risk: **overstocking** ties up capital; **stockouts** forfeit revenue during peak seasons
- Traditional forecasting methods (manual heuristics, simple averages) fail to capture complex demand patterns
- SKU-level forecasting is especially difficult — many products have sparse, intermittent demand

---

## Slide 3: Research Question

**Can machine learning models produce statistically significantly better SKU-level demand forecasts than traditional baselines?**

- Compare gradient boosted trees (XGBoost, LightGBM) against classical baselines
- Evaluate using rigorous rolling-origin cross-validation
- Test significance with paired t-tests (alpha = 0.05)

---

## Slide 4: Dataset Overview

**Monthly shotgun sales from a major U.S. firearms distributor**

- **15,181 transactions** aggregated to monthly product-level data
- **74 unique products** (subcategory x gauge x tactical designation)
- **55 active products** retained (>= 12 non-zero months; >99.9% of sales volume)
- **Date range:** January 2020 – December 2023 (48 months)
- **Challenge:** High proportion of zero-demand product-months

---

## Slide 5: Demand Patterns

**Aggregate monthly demand reveals strong seasonality and COVID disruption**

- Clear annual peaks in Sep–Oct (fall hunting opener)
- COVID-19 demand surge in early 2020
- Post-pandemic normalization trend (2021–2023)

> **Figure:** `fig1_demand_time_series.png`

---

## Slide 6: Feature Engineering

**22+ engineered features across 4 categories**

- **Temporal:** Month, quarter, sinusoidal encodings, time index
- **Lag features:** Demand at 1, 3, 6, 12-month offsets; price lag
- **Domain-specific:** Hunting season indicators (waterfowl, upland, turkey, dove, deer), hunting intensity score, COVID-period flag
- **Product:** Binary encodings for subcategory, gauge group, tactical designation

**Key insight:** Domain knowledge (hunting seasons) encoded directly as features

---

## Slide 7: Models Evaluated

**3 baselines + 2 alternatives**

| Model | Type | Description |
|-------|------|-------------|
| Seasonal Naive | Baseline | Repeats value from 12 months ago |
| Ridge Regression | Baseline | Linear model with L2 regularization |
| SARIMA(1,1,1)(1,1,1,12) | Baseline | Per-product univariate time series |
| **XGBoost** | **Alternative** | Gradient boosting, Optuna-tuned (50 trials) |
| **LightGBM** | **Alternative** | Gradient boosting, Optuna-tuned (50 trials) |

---

## Slide 8: Cross-Validation Design

**Rolling-origin CV with expanding training window**

- Minimum training window: **24 months**
- Forecast horizon: **3 months**
- Step size: **3 months**
- Total folds: **8** (Jan 2022 – Dec 2023)
- Each fold evaluates all **55 products** simultaneously (165 predictions/fold)

**Why rolling-origin?** Respects temporal ordering — no future data leakage

---

## Slide 9: Statistical Testing Framework

**One-sided paired t-tests on per-fold error metrics**

- H0: mean(alternative) >= mean(baseline) — alternative is not better
- HA: mean(alternative) < mean(baseline) — alternative outperforms
- Alpha = 0.05, n = 8 folds (7 degrees of freedom)
- Tests run on both MAPE and RMSE to examine metric sensitivity

---

## Slide 10: Overall Results — Model Comparison

**Five-model performance summary (mean +/- std across 8 folds)**

| Model | RMSE | MAE | MAPE (%) |
|-------|------|-----|----------|
| Seasonal Naive | 1,735.6 +/- 1,222.7 | 451.3 +/- 258.3 | 907.3 +/- 547.1 |
| Ridge | 737.1 +/- 407.5 | 360.5 +/- 211.0 | 1,256.5 +/- 399.5 |
| SARIMA | 283,160 +/- 264,948 | 62,627 +/- 56,012 | 1,075.1 +/- 505.1 |
| **XGBoost** | **646.2 +/- 328.1** | **289.6 +/- 142.1** | 1,087.2 +/- 388.6 |
| LightGBM | 673.0 +/- 380.2 | 314.9 +/- 168.9 | 1,047.1 +/- 416.1 |

**XGBoost achieves the lowest RMSE and MAE**

---

## Slide 11: RMSE Comparison

**XGBoost: 12% lower RMSE than Ridge, 63% lower than Seasonal Naive**

- SARIMA excluded from chart (extreme outlier: RMSE > 283,000)
- XGBoost and LightGBM consistently outperform both baselines
- Ridge shows moderate performance but higher sensitivity to large errors

> **Figure:** `fig2_rmse_comparison.png`

---

## Slide 12: Feature Importance

**Top 15 XGBoost features by importance (gain-based)**

- `quantity_lag_1` dominates at **64.4%** — strong autoregressive signal
- `quantity_lag_3` captures quarterly cycles (14.8%)
- `covid_period` ranks 3rd — reflects structural break in 2020
- Hunting season features validate domain-driven engineering (turkey season, hunting intensity in top 15)

> **Figure:** `fig4_feature_importance.png`

---

## Slide 13: RMSE t-Test Results

**2 of 3 comparisons reach statistical significance on RMSE**

| Comparison | t-stat | p-value | Significant? |
|------------|--------|---------|--------------|
| XGBoost vs Ridge | -2.02 | **0.042** | **Yes** |
| LightGBM vs Ridge | -1.54 | 0.084 | No |
| XGBoost vs Seasonal Naive | -2.45 | **0.022** | **Yes** |

**XGBoost significantly outperforms both Ridge and Seasonal Naive on RMSE**

---

## Slide 14: MAPE t-Test Results

**0 of 3 comparisons reach significance on MAPE**

| Comparison | t-stat | p-value | Significant? |
|------------|--------|---------|--------------|
| XGBoost vs Ridge | -1.05 | 0.165 | No |
| LightGBM vs Ridge | -1.21 | 0.133 | No |
| XGBoost vs Seasonal Naive | 0.71 | 0.750 | No |

**Why?** MAPE is inflated by near-zero demand products, creating high variance that masks genuine differences

---

## Slide 15: The MAPE vs RMSE Narrative

**Metric selection critically affects statistical conclusions**

- **MAPE problem:** |actual - predicted| / |actual| explodes when actual is near zero
  - Prediction of 2 when actual is 1 = 100% MAPE
  - Many product-months have zero/near-zero demand
  - Inflates all models equally but amplifies variance
- **RMSE advantage:** Weights errors by squared magnitude
  - Large errors on high-volume products (business-critical) dominate
  - Small errors on sparse products contribute minimally
- **Conclusion:** For sparse demand data, RMSE is the more appropriate primary metric

---

## Slide 16: Limitations

**Important caveats for interpreting these results**

- **Small fold count** (n = 8) limits statistical power — marginal improvements may not reach significance
- **Product aggregation** — per-fold metrics aggregate across all 55 products; high-volume products dominate RMSE/MAE
- **COVID period** — 2020–2023 includes atypical pandemic-era demand; model performance may differ under normal conditions
- **SARIMA convergence failures** — many products have insufficient history for seasonal parameter estimation
- **Single domain** — results specific to one distributor's shotgun lines

---

## Slide 17: Conclusion

**Key takeaways**

1. **XGBoost is the recommended model** — lowest RMSE (646.2) and MAE (289.6), statistically significant improvement over both Ridge and Seasonal Naive
2. **Metric choice matters** — RMSE reveals significant differences that MAPE obscures; sparse demand requires careful metric selection
3. **Domain features add value** — hunting season indicators and COVID flags contribute to model performance
4. **SARIMA is unsuitable** for multi-product sparse demand — convergence failures on most products

---

## Slide 18: Future Work

**Directions for extending this research**

- **Hierarchical forecasting** — reconciliation methods to leverage category-level patterns for SKU-level improvement
- **Intermittent demand models** — Croston's method or SBA for low-volume products, combined with ML for high-volume
- **External data** — NICS background check volumes, economic indicators, hunting license data as exogenous features
- **Conformal prediction intervals** — quantify forecast uncertainty for inventory optimization decisions
- **Extended evaluation** — assess performance beyond the COVID-affected window

---

## Figure Reference Guide

| Slide | Figure File |
|-------|-------------|
| Slide 5 | `deliverables/report/figures/fig1_demand_time_series.png` |
| Slide 11 | `deliverables/report/figures/fig2_rmse_comparison.png` |
| Slide 12 | `deliverables/report/figures/fig4_feature_importance.png` |
| Supplementary | `deliverables/report/figures/fig3_mae_comparison.png` |
| Supplementary | `deliverables/report/figures/fig5_error_distribution.png` |
| Supplementary | `deliverables/report/figures/fig6_rolling_cv_rmse.png` |
| Supplementary | `deliverables/report/figures/fig7_actual_vs_predicted.png` |
