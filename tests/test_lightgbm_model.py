"""Tests for LightGBM forecasting model."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.lightgbm_model import LightGBMForecaster, LightGBMConfig  # noqa: E402


def test_lightgbm_fit_creates_model():
    rng = np.random.RandomState(42)
    X = rng.randn(30, 5)
    y = rng.randn(30)
    model = LightGBMForecaster()
    model.fit(X, y)
    assert model.model_ is not None


def test_lightgbm_predict_shape():
    rng = np.random.RandomState(42)
    model = LightGBMForecaster()
    model.fit(rng.randn(30, 5), rng.randn(30))
    preds = model.predict(rng.randn(10, 5))
    assert preds.shape == (10,)


def test_lightgbm_stores_feature_names():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    y = np.array([1.0, 2.0, 3.0])
    model = LightGBMForecaster()
    model.fit(df, y)
    assert model.feature_names == ["a", "b", "c"]


def test_lightgbm_get_feature_importance_shape():
    rng = np.random.RandomState(42)
    X = rng.randn(30, 5)
    y = rng.randn(30)
    model = LightGBMForecaster()
    model.fit(X, y)
    importance_df = model.get_feature_importance()
    assert isinstance(importance_df, pd.DataFrame)
    assert "feature" in importance_df.columns
    assert "importance" in importance_df.columns
    assert len(importance_df) == 5


def test_lightgbm_fit_returns_self():
    rng = np.random.RandomState(42)
    model = LightGBMForecaster()
    result = model.fit(rng.randn(30, 5), rng.randn(30))
    assert result is model


def test_lightgbm_fit_with_validation():
    rng = np.random.RandomState(42)
    X_train = rng.randn(30, 5)
    y_train = rng.randn(30)
    X_val = rng.randn(10, 5)
    y_val = rng.randn(10)
    model = LightGBMForecaster()
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    assert model.model_ is not None
