"""Tests for baseline forecasting models."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.baselines import (  # noqa: E402
    RidgeForecaster,
    SARIMAForecaster,
    SeasonalNaiveForecaster,
)


def test_seasonal_naive_fit_stores_history():
    y = np.arange(24, dtype=float)
    model = SeasonalNaiveForecaster(season_length=12)
    model.fit(y)
    assert model.history_ is not None
    assert len(model.history_) == 24


def test_seasonal_naive_predict_length():
    y = np.arange(24, dtype=float)
    model = SeasonalNaiveForecaster(season_length=12)
    model.fit(y)
    preds = model.predict(horizon=3)
    assert len(preds) == 3


def test_seasonal_naive_predict_repeats_cycle():
    y = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120,
                  15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115, 125], dtype=float)
    model = SeasonalNaiveForecaster(season_length=12)
    model.fit(y)
    preds = model.predict(3)
    expected = y[-12:][:3]
    np.testing.assert_array_equal(preds, expected)


def test_ridge_fit_creates_model():
    rng = np.random.RandomState(42)
    X = rng.randn(30, 5)
    y = rng.randn(30)
    model = RidgeForecaster(alpha=1.0)
    model.fit(X, y)
    assert model.model_ is not None


def test_ridge_predict_shape():
    rng = np.random.RandomState(42)
    model = RidgeForecaster(alpha=1.0)
    model.fit(rng.randn(30, 5), rng.randn(30))
    preds = model.predict(rng.randn(10, 5))
    assert preds.shape == (10,)


def test_ridge_stores_feature_names():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    y = np.array([1.0, 2.0, 3.0])
    model = RidgeForecaster(alpha=1.0)
    model.fit(df, y)
    assert model.feature_names == ["a", "b", "c"]


def test_sarima_fit_creates_model():
    y = np.arange(1, 37, dtype=float)
    model = SARIMAForecaster(order=(1, 0, 0), seasonal_order=(0, 0, 0, 12))
    model.fit(y)
    assert model.model_ is not None


def test_sarima_predict_length():
    y = np.arange(1, 37, dtype=float)
    model = SARIMAForecaster(order=(1, 0, 0), seasonal_order=(0, 0, 0, 12))
    model.fit(y)
    preds = model.predict(horizon=3)
    assert len(preds) == 3


def test_sarima_predict_returns_numeric():
    y = np.arange(1, 37, dtype=float)
    model = SARIMAForecaster(order=(1, 0, 0), seasonal_order=(0, 0, 0, 12))
    model.fit(y)
    preds = model.predict(horizon=3)
    assert np.all(np.isfinite(preds)), f"Non-finite predictions: {preds}"
