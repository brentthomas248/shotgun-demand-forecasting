"""Tests for evaluation metrics."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import (  # noqa: E402
    compute_mae,
    compute_mape,
    compute_rmse,
    evaluate_across_folds,
    evaluate_model,
)


def test_compute_rmse_perfect():
    y = np.array([1.0, 2.0, 3.0])
    assert compute_rmse(y, y) == 0.0


def test_compute_rmse_known_value():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 3.0, 5.0])
    # errors: [0, 1, 4], mean=5/3, sqrt=1.2909...
    expected = float(np.sqrt(5.0 / 3.0))
    assert abs(compute_rmse(y_true, y_pred) - expected) < 1e-6


def test_compute_mae_known_value():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 2.0, 5.0])
    # |errors|: [1, 0, 2], mean=1.0
    assert abs(compute_mae(y_true, y_pred) - 1.0) < 1e-6


def test_compute_mape_excludes_zeros():
    y_true = np.array([0.0, 10.0, 20.0])
    y_pred = np.array([5.0, 12.0, 18.0])
    # Only non-zero actuals: |10-12|/10=0.2, |20-18|/20=0.1, mean=0.15 => 15%
    result = compute_mape(y_true, y_pred)
    assert abs(result - 15.0) < 1e-6


def test_compute_mape_known_value():
    y_true = np.array([100.0, 200.0])
    y_pred = np.array([110.0, 180.0])
    # |100-110|/100=0.1, |200-180|/200=0.1, mean=0.1 => 10%
    assert abs(compute_mape(y_true, y_pred) - 10.0) < 1e-6


def test_evaluate_model_returns_all_keys():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.5, 2.5, 3.5])
    result = evaluate_model(y_true, y_pred)
    assert set(result.keys()) == {"rmse", "mae", "mape"}
    assert all(isinstance(v, float) for v in result.values())


def test_evaluate_across_folds_summary_shape():
    folds = [
        {"rmse": 1.0, "mae": 0.5, "mape": 10.0},
        {"rmse": 2.0, "mae": 1.5, "mape": 20.0},
        {"rmse": 3.0, "mae": 2.5, "mape": 30.0},
    ]
    summary = evaluate_across_folds(folds)
    assert isinstance(summary, pd.DataFrame)
    assert list(summary.columns) == ["mean", "std", "min", "max"]
    assert len(summary) == 3  # rmse, mae, mape
    assert abs(summary.loc["rmse", "mean"] - 2.0) < 1e-6
