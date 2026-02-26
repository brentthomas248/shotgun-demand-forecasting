"""Rolling-origin temporal data splitting for time-series cross-validation.

Implements expanding-window cross-validation to produce multiple evaluation
folds required for paired t-testing. Ensures strict temporal ordering so
no future data leaks into training sets.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FoldInfo:
    """Metadata for a single cross-validation fold."""

    fold_id: int
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    train_size: int
    val_size: int


class RollingOriginSplitter:
    """Expanding-window time-series cross-validation splitter.

    Produces *k* folds where the training window grows by ``step_months``
    each iteration and the validation window is always ``horizon_months``
    long. This guarantees enough per-fold metric values for a paired t-test.

    Attributes:
        min_train_months: Minimum number of months in the first training fold.
        horizon_months: Number of months in each validation window.
        step_months: How many months to advance between folds.
        date_column: Name of the datetime column in the DataFrame.
    """

    def __init__(
        self,
        min_train_months: int = 24,
        horizon_months: int = 3,
        step_months: int = 3,
        date_column: str = "date",
    ) -> None:
        """Initialise the splitter.

        Args:
            min_train_months: Months in the first training window.
            horizon_months: Months in each validation window.
            step_months: Months to advance the origin between folds.
            date_column: Name of the datetime column.
        """
        self.min_train_months = min_train_months
        self.horizon_months = horizon_months
        self.step_months = step_months
        self.date_column = date_column

    def split(
        self, df: pd.DataFrame
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Generate train/validation index arrays for each fold.

        Args:
            df: DataFrame sorted by ``date_column`` with a DatetimeIndex or
                datetime column.

        Returns:
            List of (train_indices, val_indices) tuples, one per fold.
        """
        df = df.copy()
        df[self.date_column] = pd.to_datetime(df[self.date_column])
        dates = sorted(df[self.date_column].unique())
        logger.info(
            "Splitting %d rows across %d unique dates (min_train=%d, horizon=%d, step=%d)",
            len(df),
            len(dates),
            self.min_train_months,
            self.horizon_months,
            self.step_months,
        )

        folds: list[tuple[np.ndarray, np.ndarray]] = []
        for i in range(len(dates)):
            train_end_idx = self.min_train_months + i * self.step_months - 1
            val_start_idx = train_end_idx + 1
            val_end_idx = val_start_idx + self.horizon_months - 1
            if val_end_idx >= len(dates):
                break
            train_dates = dates[: train_end_idx + 1]
            val_dates = dates[val_start_idx : val_end_idx + 1]
            train_mask = df[self.date_column].isin(train_dates)
            val_mask = df[self.date_column].isin(val_dates)
            folds.append((
                df.index[train_mask].to_numpy(),
                df.index[val_mask].to_numpy(),
            ))
            logger.debug(
                "Fold %d: train %s to %s (%d rows), val %s to %s (%d rows)",
                len(folds),
                str(train_dates[0])[:10],
                str(train_dates[-1])[:10],
                train_mask.sum(),
                str(val_dates[0])[:10],
                str(val_dates[-1])[:10],
                val_mask.sum(),
            )

        logger.info("Created %d folds", len(folds))
        return folds

    def get_fold_info(self, df: pd.DataFrame) -> list[FoldInfo]:
        """Return descriptive metadata for every fold without copying data.

        Args:
            df: DataFrame sorted by ``date_column``.

        Returns:
            List of ``FoldInfo`` dataclass instances.
        """
        df = df.copy()
        df[self.date_column] = pd.to_datetime(df[self.date_column])
        dates = sorted(df[self.date_column].unique())

        infos: list[FoldInfo] = []
        for i in range(len(dates)):
            train_end_idx = self.min_train_months + i * self.step_months - 1
            val_start_idx = train_end_idx + 1
            val_end_idx = val_start_idx + self.horizon_months - 1
            if val_end_idx >= len(dates):
                break
            train_dates = dates[: train_end_idx + 1]
            val_dates = dates[val_start_idx : val_end_idx + 1]
            train_mask = df[self.date_column].isin(train_dates)
            val_mask = df[self.date_column].isin(val_dates)
            infos.append(FoldInfo(
                fold_id=len(infos) + 1,
                train_start=str(train_dates[0])[:10],
                train_end=str(train_dates[-1])[:10],
                val_start=str(val_dates[0])[:10],
                val_end=str(val_dates[-1])[:10],
                train_size=int(train_mask.sum()),
                val_size=int(val_mask.sum()),
            ))

        logger.info("Generated fold info for %d folds", len(infos))
        return infos


def temporal_train_test_split(
    df: pd.DataFrame,
    train_end: str = "2022-12-31",
    val_end: str = "2023-06-30",
    date_column: str = "date",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into train / validation / test by date boundaries.

    A convenience wrapper for a single fixed split (e.g. 48 / 6 / 6 months).

    Args:
        df: DataFrame with a datetime column.
        train_end: Last date (inclusive) for the training set.
        val_end: Last date (inclusive) for the validation set.
            Everything after ``val_end`` becomes the test set.
        date_column: Name of the datetime column.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    train_end_dt = pd.Timestamp(train_end)
    val_end_dt = pd.Timestamp(val_end)

    train_df = df[df[date_column] <= train_end_dt]
    val_df = df[(df[date_column] > train_end_dt) & (df[date_column] <= val_end_dt)]
    test_df = df[df[date_column] > val_end_dt]

    logger.info(
        "Temporal split: train=%d rows (to %s), val=%d rows (to %s), test=%d rows (after %s)",
        len(train_df),
        train_end,
        len(val_df),
        val_end,
        len(test_df),
        val_end,
    )
    return train_df, val_df, test_df


def filter_active_products(
    df: pd.DataFrame,
    min_nonzero_months: int = 12,
    target_column: str = "quantity",
    product_key: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter to products with sufficient non-zero demand for modeling.

    Products with fewer than ``min_nonzero_months`` months of non-zero
    target values are classified as inactive and excluded from the
    returned active DataFrame.

    Args:
        df: Input DataFrame with product and target columns.
        min_nonzero_months: Minimum months with target > 0 to qualify
            as active.
        target_column: Column to check for non-zero values.
        product_key: Columns defining a unique product. Defaults to
            ``["subcategory", "sizing", "tactical"]``.

    Returns:
        Tuple of (active_df, summary_df). ``active_df`` contains only
        rows for active products. ``summary_df`` has one row per product
        with columns: product key columns, ``total_qty``,
        ``nonzero_months``, ``total_months``, ``is_active``.
    """
    if product_key is None:
        product_key = ["subcategory", "sizing", "tactical"]

    summary = (
        df.groupby(product_key)[target_column]
        .agg(
            total_qty="sum",
            nonzero_months=lambda x: (x > 0).sum(),
            total_months="count",
        )
        .reset_index()
    )
    summary["is_active"] = summary["nonzero_months"] >= min_nonzero_months

    active_keys = summary.loc[summary["is_active"], product_key]
    active_df = df.merge(active_keys, on=product_key, how="inner")

    n_active = summary["is_active"].sum()
    n_total = len(summary)
    total_vol = df[target_column].sum()
    active_vol = active_df[target_column].sum()
    vol_pct = (active_vol / total_vol * 100) if total_vol > 0 else 0.0

    logger.info(
        "Product filter: %d/%d active (threshold=%d non-zero months), "
        "%.1f%% volume retained (%d → %d rows)",
        n_active,
        n_total,
        min_nonzero_months,
        vol_pct,
        len(df),
        len(active_df),
    )
    return active_df, summary
