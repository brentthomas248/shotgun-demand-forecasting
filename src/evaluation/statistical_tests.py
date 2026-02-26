"""Statistical hypothesis testing for model comparison.

Implements paired t-tests to determine whether an alternative model
(e.g. XGBoost) significantly outperforms a baseline (e.g. Ridge) across
rolling-origin cross-validation folds.
"""
from __future__ import annotations

import numpy as np


def paired_ttest(
    baseline_scores: np.ndarray,
    alternative_scores: np.ndarray,
    alpha: float = 0.05,
    alternative: str = "less",
) -> dict[str, float | bool]:
    """Run a paired t-test comparing per-fold metric arrays.

    Tests whether the alternative model's error scores are significantly
    lower than the baseline's.

    H_0: mean(alternative) >= mean(baseline)  (alternative is not better)
    H_A: mean(alternative) <  mean(baseline)  (alternative outperforms)

    Args:
        baseline_scores: Per-fold metric values for the baseline model
            (e.g. MAPE per fold).
        alternative_scores: Per-fold metric values for the alternative model.
        alpha: Significance level (default 0.05).
        alternative: Sidedness of the test — ``"less"`` tests if
            alternative < baseline, ``"two-sided"`` for any difference.

    Returns:
        Dictionary with keys:
        - ``t_statistic``: t-test statistic
        - ``p_value``: p-value
        - ``reject_null``: True if p_value < alpha
        - ``baseline_mean``: Mean of baseline scores
        - ``baseline_std``: Std of baseline scores
        - ``alternative_mean``: Mean of alternative scores
        - ``alternative_std``: Std of alternative scores
        - ``ci_lower``: Lower bound of 95% CI for the difference
        - ``ci_upper``: Upper bound of 95% CI for the difference
    """
    from scipy.stats import ttest_rel

    baseline_scores = np.asarray(baseline_scores, dtype=float)
    alternative_scores = np.asarray(alternative_scores, dtype=float)

    result = ttest_rel(alternative_scores, baseline_scores, alternative=alternative)
    ci = result.confidence_interval(confidence_level=1 - alpha)

    return {
        "t_statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "reject_null": bool(result.pvalue < alpha),
        "baseline_mean": float(baseline_scores.mean()),
        "baseline_std": float(baseline_scores.std()),
        "alternative_mean": float(alternative_scores.mean()),
        "alternative_std": float(alternative_scores.std()),
        "ci_lower": float(ci.low),
        "ci_upper": float(ci.high),
    }


def format_hypothesis_report(
    test_results: dict[str, float | bool],
    baseline_name: str = "Ridge",
    alternative_name: str = "XGBoost",
    metric_name: str = "MAPE",
) -> str:
    """Format t-test results as a human-readable hypothesis report.

    Args:
        test_results: Output dictionary from ``paired_ttest``.
        baseline_name: Display name for the baseline model.
        alternative_name: Display name for the alternative model.
        metric_name: Name of the metric being compared.

    Returns:
        Multi-line formatted string suitable for inclusion in the final
        report or notebook output.
    """
    r = test_results
    conclusion = (
        f"Reject H0 at alpha=0.05: {alternative_name} significantly outperforms {baseline_name}."
        if r["reject_null"]
        else f"Fail to reject H0 at alpha=0.05: no significant difference between {alternative_name} and {baseline_name}."
    )

    return (
        f"Paired t-test: {alternative_name} vs {baseline_name} ({metric_name})\n"
        f"{'=' * 60}\n"
        f"H0: mean({alternative_name}) >= mean({baseline_name})\n"
        f"HA: mean({alternative_name}) <  mean({baseline_name})\n"
        f"\n"
        f"{baseline_name:>20s}: {r['baseline_mean']:.4f} +/- {r['baseline_std']:.4f}\n"
        f"{alternative_name:>20s}: {r['alternative_mean']:.4f} +/- {r['alternative_std']:.4f}\n"
        f"\n"
        f"t-statistic: {r['t_statistic']:.4f}\n"
        f"p-value:     {r['p_value']:.6f}\n"
        f"95% CI:      [{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]\n"
        f"\n"
        f"Conclusion: {conclusion}"
    )
