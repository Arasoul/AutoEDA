"""Tests for autoeda.analytics — Analytics, correlation, normality, hypothesis tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from autoeda.analytics import (
    Analytics,
    ConfidenceIntervalResult,
    CorrelationResult,
    CovarianceResult,
    HypothesisTestResult,
    NormalityResult,
    NormalityTest,
    StatisticalAnalysis,
    _interpret_correlation,
    _interpret_strength_label,
)
from autoeda.profiler import DataProfiler

# ---------------------------------------------------------------------------
# Correlation interpretation helpers
# ---------------------------------------------------------------------------


class TestInterpretCorrelation:
    def test_very_strong_positive(self) -> None:
        assert "very strong" in _interpret_correlation(0.95)
        assert "positive" in _interpret_correlation(0.95)

    def test_strong_negative(self) -> None:
        assert "strong" in _interpret_correlation(-0.8)
        assert "negative" in _interpret_correlation(-0.8)

    def test_moderate(self) -> None:
        assert "moderate" in _interpret_correlation(0.6)

    def test_weak(self) -> None:
        assert "weak" in _interpret_correlation(0.35)

    def test_negligible(self) -> None:
        assert "negligible" in _interpret_correlation(0.1)

    def test_zero(self) -> None:
        result = _interpret_correlation(0.0)
        assert "no" in result
        assert "negligible" in result


class TestInterpretStrengthLabel:
    def test_very_strong(self) -> None:
        assert _interpret_strength_label(0.95) == "Very Strong"

    def test_strong(self) -> None:
        assert _interpret_strength_label(0.75) == "Strong"

    def test_moderate(self) -> None:
        assert _interpret_strength_label(0.55) == "Moderate"

    def test_weak(self) -> None:
        assert _interpret_strength_label(0.35) == "Weak"

    def test_negligible(self) -> None:
        assert _interpret_strength_label(0.1) == "Negligible"


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------


class TestAnalytics:
    def test_basic_analysis(self, simple_df: pd.DataFrame) -> None:
        profile = DataProfiler().profile(simple_df)
        analytics = Analytics()
        stats = analytics.analyse(simple_df, profile)
        assert isinstance(stats, StatisticalAnalysis)

    def test_pearson_correlation(self, simple_df: pd.DataFrame) -> None:
        profile = DataProfiler().profile(simple_df)
        stats = Analytics().analyse(simple_df, profile)
        assert isinstance(stats.pearson, CorrelationResult)
        assert stats.pearson.method == "pearson"
        assert isinstance(stats.pearson.matrix, dict)

    def test_spearman_correlation(self, simple_df: pd.DataFrame) -> None:
        profile = DataProfiler().profile(simple_df)
        stats = Analytics().analyse(simple_df, profile)
        assert stats.spearman.method == "spearman"

    def test_kendall_correlation(self, simple_df: pd.DataFrame) -> None:
        profile = DataProfiler().profile(simple_df)
        stats = Analytics().analyse(simple_df, profile)
        assert stats.kendall.method == "kendall"

    def test_covariance(self, simple_df: pd.DataFrame) -> None:
        profile = DataProfiler().profile(simple_df)
        stats = Analytics().analyse(simple_df, profile)
        assert isinstance(stats.covariance, CovarianceResult)
        assert "age" in stats.covariance.variances

    def test_normality(self, medium_df: pd.DataFrame) -> None:
        profile = DataProfiler().profile(medium_df)
        stats = Analytics().analyse(medium_df, profile)
        assert isinstance(stats.normality, NormalityResult)
        assert len(stats.normality.tests) > 0
        for t in stats.normality.tests:
            assert isinstance(t, NormalityTest)
            assert t.column != ""
            assert 0 <= t.shapiro_p <= 1

    def test_hypothesis_tests(self, medium_df: pd.DataFrame) -> None:
        profile = DataProfiler().profile(medium_df)
        stats = Analytics().analyse(medium_df, profile)
        assert len(stats.hypothesis_tests) > 0
        for t in stats.hypothesis_tests:
            assert isinstance(t, HypothesisTestResult)
            assert t.test_name in ("independent_t_test", "one_way_anova", "chi_square")
            assert t.column != ""
            assert t.interpretation != ""

    def test_confidence_intervals(self, simple_df: pd.DataFrame) -> None:
        profile = DataProfiler().profile(simple_df)
        stats = Analytics().analyse(simple_df, profile)
        assert len(stats.confidence_intervals) == 3
        for ci in stats.confidence_intervals:
            assert isinstance(ci, ConfidenceIntervalResult)
            assert ci.lower < ci.mean < ci.upper
            assert ci.margin_of_error > 0

    def test_correlation_pair_has_interpretation(self, simple_df: pd.DataFrame) -> None:
        profile = DataProfiler().profile(simple_df)
        stats = Analytics().analyse(simple_df, profile)
        for p in stats.pearson.significant_pairs:
            assert p.interpretation != ""
            assert p.strength_label != ""

    def test_no_numeric_columns(self, simple_df: pd.DataFrame) -> None:
        analytics = Analytics()
        result = analytics._correlation(simple_df, [], "pearson")
        assert result.matrix == {}
        assert result.significant_pairs == []
        assert result.n_significant == 0

    def test_single_numeric_column(self, simple_df: pd.DataFrame) -> None:
        analytics = Analytics()
        result = analytics._correlation(simple_df, ["age"], "pearson")
        assert result.matrix == {}
        assert result.n_significant == 0

    def test_covariance_single_column(self, simple_df: pd.DataFrame) -> None:
        analytics = Analytics()
        result = analytics._covariance(simple_df, ["age"])
        assert result.matrix == {}
        assert result.variances == {}

    def test_normality_skipped_short_column(self) -> None:
        df = pd.DataFrame({"short": [1.0, 2.0, 3.0]})
        profile = DataProfiler().profile(df)
        stats = Analytics().analyse(df, profile)
        assert len(stats.normality.tests) == 0

    def test_hypothesis_no_cat_columns(self, simple_df: pd.DataFrame) -> None:
        profile = DataProfiler().profile(simple_df)
        analytics = Analytics()
        result = analytics._hypothesis_tests(
            simple_df,
            profile.numerical_columns,
            [],
        )
        assert result == []

    def test_hypothesis_two_groups(self, medium_df: pd.DataFrame) -> None:
        profile = DataProfiler().profile(medium_df)
        stats = Analytics().analyse(medium_df, profile)
        t_tests = [t for t in stats.hypothesis_tests if t.test_name == "independent_t_test"]
        for t in t_tests:
            assert t.group_a is not None
            assert t.group_b is not None

    def test_anova_three_groups(self) -> None:
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "value": np.concatenate(
                    [
                        np.random.randn(30),
                        np.random.randn(30) + 2,
                        np.random.randn(30) + 4,
                    ]
                ),
                "group": ["A"] * 30 + ["B"] * 30 + ["C"] * 30,
            }
        )
        profile = DataProfiler().profile(df)
        stats = Analytics().analyse(df, profile)
        anova = [t for t in stats.hypothesis_tests if t.test_name == "one_way_anova"]
        assert len(anova) > 0

    def test_chi_square(self) -> None:
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "cat_a": np.random.choice(["X", "Y", "Z"], 100),
                "cat_b": np.random.choice(["P", "Q"], 100),
            }
        )
        profile = DataProfiler().profile(df)
        stats = Analytics().analyse(df, profile)
        chi2 = [t for t in stats.hypothesis_tests if t.test_name == "chi_square"]
        assert len(chi2) > 0
        for t in chi2:
            assert t.interpretation != ""

    def test_confidence_level_affects_width(self, simple_df: pd.DataFrame) -> None:
        from autoeda.config import AutoEDAConfig

        profile = DataProfiler().profile(simple_df)
        narrow = Analytics(AutoEDAConfig(confidence_level=0.90)).analyse(simple_df, profile)
        wide = Analytics(AutoEDAConfig(confidence_level=0.99)).analyse(simple_df, profile)
        for n_ci, w_ci in zip(narrow.confidence_intervals, wide.confidence_intervals, strict=True):
            n_width = n_ci.upper - n_ci.lower
            w_width = w_ci.upper - w_ci.lower
            assert w_width > n_width
