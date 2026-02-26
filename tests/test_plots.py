"""Tests for publication-quality visualisation functions."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization.plots import (  # noqa: E402
    plot_error_distribution,
    plot_feature_importance,
    plot_model_comparison,
    plot_rolling_cv_performance,
    plot_time_series,
    set_publication_style,
)


@pytest.fixture(autouse=True)
def _cleanup():
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# set_publication_style
# ---------------------------------------------------------------------------

def test_set_publication_style_sets_dpi():
    set_publication_style()
    assert matplotlib.rcParams["figure.dpi"] == 300


# ---------------------------------------------------------------------------
# plot_time_series
# ---------------------------------------------------------------------------

def test_plot_time_series_returns_figure():
    dates = pd.date_range("2020-01-01", periods=10, freq="MS")
    actual = np.random.default_rng(42).normal(100, 10, size=10)
    fig = plot_time_series(dates, actual)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1


def test_plot_time_series_with_predictions():
    dates = pd.date_range("2020-01-01", periods=10, freq="MS")
    rng = np.random.default_rng(42)
    actual = rng.normal(100, 10, size=10)
    predicted = rng.normal(100, 10, size=10)
    fig = plot_time_series(dates, actual, predicted=predicted)
    ax = fig.axes[0]
    assert len(ax.get_lines()) == 2


# ---------------------------------------------------------------------------
# plot_model_comparison
# ---------------------------------------------------------------------------

def test_plot_model_comparison_returns_figure():
    df = pd.DataFrame(
        {
            "model": ["Linear", "XGBoost", "LightGBM"],
            "mean": [15.2, 10.1, 9.8],
            "std": [2.1, 1.5, 1.3],
        }
    )
    fig = plot_model_comparison(df)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1


# ---------------------------------------------------------------------------
# plot_feature_importance
# ---------------------------------------------------------------------------

def test_plot_feature_importance_returns_figure():
    df = pd.DataFrame(
        {
            "feature": [f"feat_{i}" for i in range(20)],
            "importance": np.random.default_rng(0).random(20),
        }
    )
    top_n = 10
    fig = plot_feature_importance(df, top_n=top_n)
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    # horizontal bar chart: number of patches equals min(top_n, len(df))
    bars = [p for p in ax.patches]
    assert len(bars) == min(top_n, len(df))


# ---------------------------------------------------------------------------
# plot_error_distribution
# ---------------------------------------------------------------------------

def test_plot_error_distribution_returns_figure():
    rng = np.random.default_rng(7)
    errors = {
        "Linear": rng.normal(5, 2, size=10),
        "XGBoost": rng.normal(3, 1, size=10),
        "LightGBM": rng.normal(2.5, 0.8, size=10),
    }
    fig = plot_error_distribution(errors)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1


# ---------------------------------------------------------------------------
# plot_rolling_cv_performance
# ---------------------------------------------------------------------------

def test_plot_rolling_cv_performance_returns_figure():
    rows = []
    for model in ["XGBoost", "LightGBM"]:
        for fold in range(1, 6):
            rows.append(
                {"fold": fold, "model": model, "mape": np.random.default_rng(fold).random() * 20}
            )
    df = pd.DataFrame(rows)
    fig = plot_rolling_cv_performance(df)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1
