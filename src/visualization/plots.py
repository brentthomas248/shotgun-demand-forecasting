"""Publication-quality visualisations for the demand forecasting project.

All plots use matplotlib/seaborn and are styled for ICML-format inclusion.
Functions return the Figure object for further customisation or saving.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib
import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.figure import Figure

matplotlib.use("Agg")

# Muted colour palette for publication plots
_PALETTE = [
    "#4878D0",  # blue
    "#EE854A",  # orange
    "#6ACC64",  # green
    "#D65F5F",  # red
    "#956CB4",  # purple
    "#8C613C",  # brown
    "#DC7EC0",  # pink
    "#797979",  # gray
    "#D5BB67",  # gold
    "#82C6E2",  # light blue
]


def set_publication_style() -> None:
    """Configure matplotlib/seaborn for publication-quality output.

    Sets font sizes, DPI, grid style, and colour palette suitable for
    an ICML-style paper (6-8 pages).
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.family": "serif",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.figsize": (6, 4),
            "grid.color": "#CCCCCC",
            "grid.linewidth": 0.5,
            "figure.autolayout": True,
            "axes.prop_cycle": plt.cycler("color", _PALETTE),
        }
    )


def plot_time_series(
    dates: pd.Series | np.ndarray,
    actual: np.ndarray,
    predicted: np.ndarray | None = None,
    title: str = "Shotgun Demand Over Time",
    ylabel: str = "Quantity",
) -> Figure:
    """Plot actual vs predicted time series.

    Args:
        dates: Date values for the x-axis.
        actual: Observed demand values.
        predicted: Optional model predictions to overlay.
        title: Plot title.
        ylabel: Label for the y-axis.

    Returns:
        matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    set_publication_style()

    fig, ax = plt.subplots()
    ax.plot(dates, actual, color=_PALETTE[0], linewidth=1.5, label="Actual")
    if predicted is not None:
        ax.plot(
            dates,
            predicted,
            color=_PALETTE[1],
            linewidth=1.5,
            linestyle="--",
            label="Predicted",
        )
        ax.legend()
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    fig.autofmt_xdate()
    return fig


def plot_model_comparison(
    results_df: pd.DataFrame,
    metric: str = "mape",
    title: str = "Model Comparison",
) -> Figure:
    """Bar chart comparing models on a given metric (mean +/- std).

    Args:
        results_df: DataFrame with columns ``model``, ``mean``, ``std``
            (one row per model).
        metric: Metric name for axis labelling.
        title: Plot title.

    Returns:
        matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    set_publication_style()

    df = results_df.sort_values("mean").reset_index(drop=True)
    best_idx = 0  # lowest mean after sorting

    colors = [_PALETTE[0]] * len(df)
    colors[best_idx] = _PALETTE[2]  # highlight best in green

    fig, ax = plt.subplots()
    x = np.arange(len(df))
    ax.bar(
        x,
        df["mean"],
        yerr=df["std"],
        color=colors,
        capsize=4,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], rotation=30, ha="right")
    ax.set_ylabel(metric.upper())
    ax.set_title(title)
    return fig


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 15,
    title: str = "Top Feature Importances",
) -> Figure:
    """Horizontal bar chart of top-N feature importances.

    Args:
        importance_df: DataFrame with ``feature`` and ``importance`` columns.
        top_n: Number of top features to display.
        title: Plot title.

    Returns:
        matplotlib Figure.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    set_publication_style()

    df = (
        importance_df.nlargest(top_n, "importance")
        .sort_values("importance", ascending=True)
        .reset_index(drop=True)
    )

    cmap = sns.light_palette(_PALETTE[0], n_colors=len(df), as_cmap=False)

    fig, ax = plt.subplots()
    ax.barh(
        df["feature"],
        df["importance"],
        color=cmap,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_xlabel("Importance")
    ax.set_title(title)
    return fig


def plot_error_distribution(
    errors: dict[str, np.ndarray],
    title: str = "Forecast Error Distribution",
) -> Figure:
    """Box/violin plot of per-fold errors for each model.

    Args:
        errors: Mapping of model name to array of per-fold error values.
        title: Plot title.

    Returns:
        matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    set_publication_style()

    labels = list(errors.keys())
    data = [errors[k] for k in labels]

    fig, ax = plt.subplots()
    bp = ax.boxplot(
        data,
        tick_labels=labels,
        patch_artist=True,
        widths=0.5,
        medianprops={"color": "black", "linewidth": 1.5},
    )
    for patch, color in zip(bp["boxes"], _PALETTE[: len(labels)], strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Error")
    ax.set_title(title)
    if len(labels) > 4:
        ax.tick_params(axis="x", rotation=30)
    return fig


def plot_rolling_cv_performance(
    fold_metrics: pd.DataFrame,
    metric: str = "mape",
    title: str = "Rolling CV Performance",
) -> Figure:
    """Line plot of per-fold metric values across the evaluation timeline.

    Args:
        fold_metrics: DataFrame with columns ``fold``, ``model``, and the
            metric column (one row per model per fold).
        metric: Which metric column to plot.
        title: Plot title.

    Returns:
        matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    set_publication_style()

    fig, ax = plt.subplots()
    models = fold_metrics["model"].unique()
    for i, model in enumerate(models):
        subset = fold_metrics[fold_metrics["model"] == model].sort_values("fold")
        ax.plot(
            subset["fold"],
            subset[metric],
            marker="o",
            markersize=5,
            linewidth=1.5,
            color=_PALETTE[i % len(_PALETTE)],
            label=model,
        )
    ax.set_xlabel("Fold")
    ax.set_ylabel(metric.upper())
    ax.set_title(title)
    ax.legend()
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    return fig
