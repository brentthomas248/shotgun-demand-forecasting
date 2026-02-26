"""Tests for rolling-origin data splitter."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_splitter import (  # noqa: E402
    RollingOriginSplitter,
    filter_active_products,
    temporal_train_test_split,
)


@pytest.fixture(scope="module")
def panel_48m():
    """Synthetic panel: 2 products x 48 months (Jan 2020 - Dec 2023)."""
    dates = pd.date_range("2020-01-01", periods=48, freq="MS")
    rows = []
    for d in dates:
        rows.append({"date": d, "subcategory": "Pump", "quantity": 100})
        rows.append({"date": d, "subcategory": "Semi", "quantity": 50})
    return pd.DataFrame(rows)


def test_splitter_fold_count(panel_48m):
    splitter = RollingOriginSplitter(min_train_months=24, horizon_months=3, step_months=3)
    folds = splitter.split(panel_48m)
    assert len(folds) == 8, f"Expected 8 folds, got {len(folds)}"


def test_splitter_no_temporal_leakage(panel_48m):
    splitter = RollingOriginSplitter(min_train_months=24, horizon_months=3, step_months=3)
    folds = splitter.split(panel_48m)
    for i, (train_idx, val_idx) in enumerate(folds):
        max_train_date = panel_48m.iloc[train_idx]["date"].max()
        min_val_date = panel_48m.iloc[val_idx]["date"].min()
        assert max_train_date < min_val_date, f"Fold {i}: temporal leakage"


def test_splitter_expanding_window(panel_48m):
    splitter = RollingOriginSplitter(min_train_months=24, horizon_months=3, step_months=3)
    folds = splitter.split(panel_48m)
    sizes = [len(train_idx) for train_idx, _ in folds]
    for i in range(1, len(sizes)):
        assert sizes[i] > sizes[i - 1], f"Fold {i} train not larger than fold {i-1}"


def test_splitter_consistent_horizon(panel_48m):
    splitter = RollingOriginSplitter(min_train_months=24, horizon_months=3, step_months=3)
    folds = splitter.split(panel_48m)
    n_products = 2
    expected_val_size = 3 * n_products  # 3 months x 2 products
    for i, (_, val_idx) in enumerate(folds):
        assert len(val_idx) == expected_val_size, (
            f"Fold {i}: expected {expected_val_size} val rows, got {len(val_idx)}"
        )


def test_fold_info_matches_split(panel_48m):
    splitter = RollingOriginSplitter(min_train_months=24, horizon_months=3, step_months=3)
    folds = splitter.split(panel_48m)
    infos = splitter.get_fold_info(panel_48m)
    assert len(infos) == len(folds)
    for info, (train_idx, val_idx) in zip(infos, folds):
        assert info.train_size == len(train_idx)
        assert info.val_size == len(val_idx)


def test_temporal_train_test_split_boundaries(panel_48m):
    train_df, val_df, test_df = temporal_train_test_split(
        panel_48m, train_end="2022-12-31", val_end="2023-06-30"
    )
    assert train_df["date"].max() <= pd.Timestamp("2022-12-31")
    assert val_df["date"].min() > pd.Timestamp("2022-12-31")
    assert val_df["date"].max() <= pd.Timestamp("2023-06-30")
    assert test_df["date"].min() > pd.Timestamp("2023-06-30")
    assert len(train_df) + len(val_df) + len(test_df) == len(panel_48m)


# --- filter_active_products tests ---


@pytest.fixture(scope="module")
def mixed_panel():
    """Panel with 3 products: high-volume, low-volume (11 months), zero-volume."""
    dates = pd.date_range("2020-01-01", periods=48, freq="MS")
    rows = []
    for i, d in enumerate(dates):
        # Active: non-zero every month
        rows.append({"date": d, "subcategory": "Pump", "sizing": "12 GA", "tactical": "Y", "quantity": 100})
        # Borderline inactive: only 11 non-zero months
        rows.append({"date": d, "subcategory": "Bolt", "sizing": "16 GA", "tactical": "N", "quantity": 5 if i < 11 else 0})
        # Zero volume: never sells
        rows.append({"date": d, "subcategory": "Revolver", "sizing": "410 GA", "tactical": "N", "quantity": 0})
    return pd.DataFrame(rows)


def test_filter_active_products_removes_zero_volume(mixed_panel):
    active_df, summary = filter_active_products(mixed_panel, min_nonzero_months=1)
    active_products = set(zip(active_df["subcategory"], active_df["sizing"], active_df["tactical"]))
    assert ("Revolver", "410 GA", "N") not in active_products


def test_filter_active_products_threshold(mixed_panel):
    # Threshold 12: Bolt (11 non-zero months) should be excluded
    active_df, summary = filter_active_products(mixed_panel, min_nonzero_months=12)
    active_products = set(zip(active_df["subcategory"], active_df["sizing"], active_df["tactical"]))
    assert ("Pump", "12 GA", "Y") in active_products
    assert ("Bolt", "16 GA", "N") not in active_products

    # Threshold 11: Bolt should now be included
    active_df2, _ = filter_active_products(mixed_panel, min_nonzero_months=11)
    active_products2 = set(zip(active_df2["subcategory"], active_df2["sizing"], active_df2["tactical"]))
    assert ("Bolt", "16 GA", "N") in active_products2


def test_filter_active_products_preserves_volume(mixed_panel):
    active_df, _ = filter_active_products(mixed_panel, min_nonzero_months=12)
    # Active df should have all 48 rows for Pump (the only active product)
    assert len(active_df) == 48


def test_filter_active_products_summary(mixed_panel):
    _, summary = filter_active_products(mixed_panel, min_nonzero_months=12)
    # Summary should have all 3 products
    assert len(summary) == 3
    assert "total_qty" in summary.columns
    assert "nonzero_months" in summary.columns
    assert "total_months" in summary.columns
    assert "is_active" in summary.columns
    assert summary["is_active"].sum() == 1  # Only Pump
