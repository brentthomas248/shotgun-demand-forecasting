"""Baseline forecasting models for demand prediction.

Provides three baselines of increasing complexity:
1. SeasonalNaiveForecaster — same month last year (simplest meaningful baseline)
2. RidgeForecaster — linear model on engineered features
3. SARIMAForecaster — classical time-series method (SARIMA)

All classes follow a consistent fit/predict interface.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SeasonalNaiveForecaster:
    """Predict using the value from the same month in the previous year.

    This is the simplest meaningful baseline for monthly seasonal data.
    No parameters to tune — purely deterministic.

    Attributes:
        season_length: Number of periods in one seasonal cycle (default 12
            for monthly data).
        history_: Stored training values after ``fit``.
    """

    def __init__(self, season_length: int = 12) -> None:
        """Initialise the seasonal naive forecaster.

        Args:
            season_length: Periods per seasonal cycle.
        """
        self.season_length = season_length
        self.history_: np.ndarray | None = None

    def fit(self, y: np.ndarray | pd.Series) -> SeasonalNaiveForecaster:
        """Store historical values for look-back prediction.

        Args:
            y: Training target values in temporal order.

        Returns:
            self
        """
        self.history_ = np.asarray(y)
        logger.info("SeasonalNaiveForecaster fit on %d observations", len(self.history_))
        return self

    def predict(self, horizon: int) -> np.ndarray:
        """Generate forecasts by repeating the last seasonal cycle.

        Args:
            horizon: Number of future periods to predict.

        Returns:
            Array of predicted values with length ``horizon``.
        """
        if self.history_ is None:
            msg = "Must call fit() before predict()"
            raise ValueError(msg)
        cycle = self.history_[-self.season_length:]
        return np.tile(cycle, (horizon // self.season_length) + 1)[:horizon]


class RidgeForecaster:
    """Ridge regression on engineered features.

    Wraps scikit-learn's Ridge with a fit/predict interface consistent
    with the other forecasters in this module.

    Attributes:
        alpha: L2 regularisation strength.
        feature_names: Feature columns used during fitting.
        model_: Fitted Ridge estimator.
    """

    def __init__(self, alpha: float = 1.0, random_state: int = 42) -> None:
        """Initialise the Ridge forecaster.

        Args:
            alpha: Regularisation strength (higher = more regularisation).
            random_state: Seed for reproducibility.
        """
        self.alpha = alpha
        self.random_state = random_state
        self.feature_names: list[str] | None = None
        self.model_: Any = None

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray | pd.Series,
        feature_names: list[str] | None = None,
    ) -> RidgeForecaster:
        """Fit the Ridge model on training features and target.

        Args:
            X: Feature matrix.
            y: Target values.
            feature_names: Optional column names for feature importance
                reporting.

        Returns:
            self
        """
        from sklearn.linear_model import Ridge

        self.model_ = Ridge(alpha=self.alpha, random_state=self.random_state)
        self.model_.fit(X, y)
        if feature_names is not None:
            self.feature_names = list(feature_names)
        elif isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
        logger.info("RidgeForecaster fit with alpha=%.4f on %d samples", self.alpha, len(y))
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Generate predictions from feature matrix.

        Args:
            X: Feature matrix (same columns as training).

        Returns:
            Array of predicted values.
        """
        return self.model_.predict(X)


class SARIMAForecaster:
    """Seasonal ARIMA forecaster.

    Wraps statsmodels SARIMAX with a simplified fit/predict interface.

    Attributes:
        order: ARIMA (p, d, q) order.
        seasonal_order: Seasonal (P, D, Q, s) order.
        model_: Fitted SARIMAX results object.
    """

    def __init__(
        self,
        order: tuple[int, int, int] = (1, 1, 1),
        seasonal_order: tuple[int, int, int, int] = (1, 1, 1, 12),
    ) -> None:
        """Initialise the SARIMA forecaster.

        Args:
            order: Non-seasonal ARIMA (p, d, q) order.
            seasonal_order: Seasonal (P, D, Q, s) order.
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.model_: Any = None

    def fit(self, y: np.ndarray | pd.Series) -> SARIMAForecaster:
        """Fit the SARIMA model to training data.

        Args:
            y: Univariate training time series in temporal order.

        Returns:
            self
        """
        import warnings

        from statsmodels.tsa.statespace.sarimax import SARIMAX

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            model = SARIMAX(
                y,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            self.model_ = model.fit(disp=False, maxiter=200)
        logger.info(
            "SARIMAForecaster fit with order=%s seasonal_order=%s",
            self.order,
            self.seasonal_order,
        )
        return self

    def predict(self, horizon: int) -> np.ndarray:
        """Forecast future values.

        Args:
            horizon: Number of periods to forecast.

        Returns:
            Array of predicted values with length ``horizon``.
        """
        return np.asarray(self.model_.forecast(steps=horizon))
