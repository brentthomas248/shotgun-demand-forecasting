# Shotgun Demand Forecasting Using Machine Learning: A Comparative Study

**CIS 531/731 Term Project Proposal** | Brent Showalter | Kansas State University

---

## 1. Introduction

Accurate demand forecasting is a critical capability for firearms distributors navigating a market defined by sharp seasonal cycles, regulatory uncertainty, and volatile consumer sentiment. Shotgun sales in the U.S. follow well-documented seasonal patterns driven by state-level hunting openers — waterfowl, upland bird, turkey, and deer seasons each create distinct demand windows between September and January. Failure to anticipate these cycles leads directly to business losses: overstocking ties up capital and warehouse space, while stockouts during peak hunting seasons forfeit revenue and damage distributor–retailer relationships.

This project investigates whether modern machine learning models — specifically gradient boosted trees — can produce statistically significantly better SKU-level demand forecasts than traditional time-series baselines for a major U.S. firearms distributor's DLX shotgun product lines. We frame this as a supervised regression problem, engineering domain-specific features from monthly transactional data spanning January 2020 through December 2023.

## 2. Related Work

Classical statistical methods for demand forecasting, including ARIMA and exponential smoothing, remain industry staples and are thoroughly surveyed by Hyndman and Athanasopoulos (2021). However, these univariate methods cannot easily incorporate exogenous features such as seasonal indicators or economic signals.

Gradient boosting frameworks have emerged as strong alternatives for tabular time-series problems. XGBoost (Chen & Guestrin, 2016) and LightGBM (Ke et al., 2017) have dominated forecasting competitions and practical applications, particularly when combined with engineered lag and calendar features. Januschowski et al. (2020) provide a comprehensive review, noting that feature-engineered gradient boosting often outperforms deep learning on structured data with moderate sample sizes.

A key challenge in our domain is sparse (intermittent) demand. Many SKUs have months with zero or near-zero sales, which inflates percentage-based error metrics like MAPE and complicates model evaluation. Kolassa (2016) argues against MAPE for intermittent demand, recommending absolute or squared-error metrics instead. We adopt RMSE as our primary metric based on this guidance.

For statistical model comparison, Demsar (2006) establishes the paired t-test framework for comparing predictors across multiple evaluation folds, which we adapt to our rolling-origin cross-validation setting.

## 3. Proposed Methodology

**Data pipeline.** We build a PySpark ETL pipeline to process raw transactional data into monthly product-level aggregations. Products are defined by the combination of subcategory, gauge/sizing, and tactical designation, yielding 74 unique SKUs. Products with fewer than 12 non-zero demand months are filtered out, retaining 55 active products that account for over 99.9% of total sales volume.

**Feature engineering.** We construct 22+ engineered features across four categories: (1) temporal features including sinusoidal month encodings and time indices; (2) lag features capturing demand at 1, 3, 6, and 12-month offsets; (3) domain features including binary hunting season indicators (waterfowl, upland, turkey, dove, deer), a composite hunting intensity score, and a COVID-period flag; and (4) product features encoding subcategory, gauge group, and tactical designation.

**Models.** We evaluate five forecasting approaches:
- *Baselines:* Seasonal Naive (12-month lookback), Ridge Regression, and SARIMA(1,1,1)(1,1,1,12)
- *Alternatives:* XGBoost and LightGBM, each hyperparameter-tuned via Optuna (50 trials)

**Cross-validation.** We employ rolling-origin cross-validation with an expanding training window (minimum 24 months), a 3-month forecast horizon, and a 3-month step size, producing 8 sequential folds covering January 2022 through December 2023.

## 4. Evaluation Plan

Models are compared on three error metrics computed per fold across all active products:
- **RMSE** (primary) — robust to sparse demand, emphasizes high-volume products
- **MAE** — interpretable in original units (quantity)
- **MAPE** — included for completeness but expected to be unreliable due to sparse demand inflation

Statistical significance is assessed via one-sided paired t-tests at alpha = 0.05, comparing each alternative model against each baseline on per-fold metric arrays (n = 8 folds). The null hypothesis states that the alternative model's mean error is greater than or equal to the baseline's.

## 5. Milestones

| Sprint | Deliverable | Timeline |
|--------|-------------|----------|
| Sprint 1 | Data acquisition, EDA, PySpark ETL pipeline | Weeks 1–2 |
| Sprint 2 | Feature engineering, baseline models (Seasonal Naive, Ridge, SARIMA) | Weeks 3–4 |
| Sprint 3 | XGBoost/LightGBM implementation, Optuna hyperparameter tuning | Weeks 5–6 |
| Sprint 4 | Rolling-origin CV evaluation, paired t-tests, statistical analysis | Weeks 7–8 |
| Sprint 5 | Final report notebook, presentation, and proposal document | Weeks 9–10 |

## References

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794.

Demsar, J. (2006). Statistical comparisons of classifiers over multiple data sets. *Journal of Machine Learning Research*, 7, 1–30.

Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts.

Januschowski, T., et al. (2020). Criteria for classifying forecasting methods. *International Journal of Forecasting*, 36(1), 167–177.

Ke, G., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems*, 30, 3146–3154.

Kolassa, S. (2016). Evaluating predictive count data distributions in retail sales forecasting. *International Journal of Forecasting*, 32(3), 788–803.
