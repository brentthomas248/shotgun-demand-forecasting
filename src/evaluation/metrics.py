"""Evaluation metrics for demand forecasting models.

Provides RMSE, MAE, and MAPE computation plus convenience wrappers for
evaluating a single model or running evaluation across all cross-validation
folds produced by RollingOriginSplitter.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Error.

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values.

    Returns:
        RMSE as a float.
    """
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Absolute Error.

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values.

    Returns:
        MAE as a float.
    """
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Absolute Percentage Error.

    Returns MAPE as a percentage (0-100 scale). Handles zero actual values
    by excluding them from the computation.

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values.

    Returns:
        MAPE as a percentage (e.g. 12.5 means 12.5%).
    """
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute all standard metrics for a single prediction set.

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values.

    Returns:
        Dictionary with keys ``rmse``, ``mae``, ``mape``.
    """
    return {
        "rmse": compute_rmse(y_true, y_pred),
        "mae": compute_mae(y_true, y_pred),
        "mape": compute_mape(y_true, y_pred),
    }


def evaluate_across_folds(
    fold_results: list[dict[str, float]],
) -> pd.DataFrame:
    """Aggregate per-fold metrics into a summary table.

    Computes mean and standard deviation for each metric across folds.

    Args:
        fold_results: List of metric dictionaries, one per fold (as returned
            by ``evaluate_model``).

    Returns:
        DataFrame with rows for each metric and columns
        ``mean``, ``std``, ``min``, ``max``.
    """
    df = pd.DataFrame(fold_results)
    return pd.DataFrame({
        "mean": df.mean(),
        "std": df.std(),
        "min": df.min(),
        "max": df.max(),
    })
