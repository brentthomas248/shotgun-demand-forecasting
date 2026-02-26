"""Tests for statistical hypothesis testing."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.statistical_tests import paired_ttest, format_hypothesis_report  # noqa: E402


def test_paired_ttest_returns_all_keys():
    baseline = np.array([10.0, 12.0, 11.0, 13.0, 9.0, 14.0, 10.5, 11.5])
    alternative = np.array([9.0, 11.0, 10.0, 12.0, 8.0, 13.0, 9.5, 10.5])
    result = paired_ttest(baseline, alternative)
    expected_keys = {
        "t_statistic", "p_value", "reject_null",
        "baseline_mean", "baseline_std",
        "alternative_mean", "alternative_std",
        "ci_lower", "ci_upper",
    }
    assert set(result.keys()) == expected_keys


def test_paired_ttest_identical_scores():
    scores = np.array([10.0, 12.0, 11.0, 13.0, 9.0, 14.0, 10.5, 11.5])
    result = paired_ttest(scores, scores)
    assert result["reject_null"] is False
    # p_value is NaN when differences have zero variance (identical arrays)
    assert np.isnan(result["p_value"]) or result["p_value"] > 0.05


def test_paired_ttest_reject_null_with_clear_difference():
    baseline = np.array([100.0] * 8)
    alternative = np.array([10.0] * 8)
    result = paired_ttest(baseline, alternative)
    assert result["reject_null"] is True


def test_format_hypothesis_report_contains_model_names():
    result = paired_ttest(
        np.array([10.0, 12.0, 11.0, 13.0, 9.0, 14.0, 10.5, 11.5]),
        np.array([9.0, 11.0, 10.0, 12.0, 8.0, 13.0, 9.5, 10.5]),
    )
    report = format_hypothesis_report(result, baseline_name="Ridge", alternative_name="XGBoost")
    assert "Ridge" in report
    assert "XGBoost" in report


def test_format_hypothesis_report_contains_conclusion():
    result = paired_ttest(
        np.array([10.0, 12.0, 11.0, 13.0, 9.0, 14.0, 10.5, 11.5]),
        np.array([9.0, 11.0, 10.0, 12.0, 8.0, 13.0, 9.5, 10.5]),
    )
    report = format_hypothesis_report(result)
    assert "Reject" in report or "Fail to reject" in report
