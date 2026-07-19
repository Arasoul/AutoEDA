"""Tests for autoeda.insight_engine — InsightEngine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autoeda.analytics import Analytics
from autoeda.insight_engine import (
    ExecutiveSummary,
    Insight,
    InsightEngine,
    InsightResult,
    Recommendation,
    VisualizationRecommendation,
)
from autoeda.profiler import DataProfiler

# ---------------------------------------------------------------------------
# InsightEngine
# ---------------------------------------------------------------------------


class TestInsightEngine:
    @pytest.fixture()
    def medium_results(self, medium_df):
        profile = DataProfiler().profile(medium_df)
        stats = Analytics().analyse(medium_df, profile)
        engine = InsightEngine()
        result = engine.generate(medium_df, profile, stats)
        return result

    def test_returns_insight_result(self, medium_results) -> None:
        assert isinstance(medium_results, InsightResult)

    def test_has_insights(self, medium_results) -> None:
        assert len(medium_results.insights) > 0
        for ins in medium_results.insights:
            assert isinstance(ins, Insight)
            assert ins.category != ""
            assert ins.severity in ("info", "warning", "critical")
            assert ins.title != ""
            assert ins.detail != ""
            assert ins.source_metric != ""

    def test_has_recommendations(self, medium_results) -> None:
        assert len(medium_results.recommendations) > 0
        for rec in medium_results.recommendations:
            assert isinstance(rec, Recommendation)
            assert rec.priority in ("high", "medium", "low")
            assert rec.title != ""
            assert rec.detail != ""

    def test_has_executive_summary(self, medium_results) -> None:
        es = medium_results.executive_summary
        assert isinstance(es, ExecutiveSummary)
        assert es.dataset_shape != ""
        assert len(es.key_findings) > 0
        assert es.narrative != ""

    def test_has_viz_recommendations(self, medium_results) -> None:
        assert len(medium_results.viz_recommendations) > 0
        for vr in medium_results.viz_recommendations:
            assert isinstance(vr, VisualizationRecommendation)
            assert vr.plot_type != ""
            assert vr.reason != ""
            assert len(vr.columns) > 0

    def test_distribution_insights(self, medium_df) -> None:
        profile = DataProfiler().profile(medium_df)
        engine = InsightEngine()
        insights = engine._distribution_insights(profile)
        # revenue is lognormal → should be skewed
        assert any(i.category == "distribution" for i in insights)

    def test_missing_insights_no_missing(self, simple_df) -> None:
        profile = DataProfiler().profile(simple_df)
        engine = InsightEngine()
        insights = engine._missing_insights(profile)
        assert len(insights) == 1
        assert "No missing" in insights[0].title

    def test_missing_insights_with_missing(self, df_with_missing) -> None:
        profile = DataProfiler().profile(df_with_missing)
        engine = InsightEngine()
        insights = engine._missing_insights(profile)
        assert len(insights) >= 1
        assert any(i.severity in ("warning", "critical") for i in insights)

    def test_outlier_insights(self, medium_df) -> None:
        profile = DataProfiler().profile(medium_df)
        engine = InsightEngine()
        insights = engine._outlier_insights(profile)
        if any(s.n_outliers > 0 for s in profile.numerical_stats):
            assert len(insights) > 0

    def test_normality_insights(self, medium_df) -> None:
        profile = DataProfiler().profile(medium_df)
        stats = Analytics().analyse(medium_df, profile)
        engine = InsightEngine()
        insights = engine._normality_insights(stats)
        non_normal = [t for t in stats.normality.tests if not t.is_normal]
        if non_normal:
            assert len(insights) > 0

    def test_correlation_insights(self, simple_df) -> None:
        profile = DataProfiler().profile(simple_df)
        stats = Analytics().analyse(simple_df, profile)
        engine = InsightEngine()
        insights = engine._correlation_insights(stats)
        for ins in insights:
            assert ins.category == "correlation"

    def test_categorical_insights(self, medium_df) -> None:
        profile = DataProfiler().profile(medium_df)
        engine = InsightEngine()
        insights = engine._categorical_insights(profile)
        for ins in insights:
            assert ins.category == "categorical"

    def test_hypothesis_insights(self, medium_df) -> None:
        profile = DataProfiler().profile(medium_df)
        stats = Analytics().analyse(medium_df, profile)
        engine = InsightEngine()
        insights = engine._hypothesis_insights(stats)
        sig = [t for t in stats.hypothesis_tests if t.significant]
        if sig:
            assert len(insights) > 0

    def test_recommendations_missing_data(self, df_with_missing) -> None:
        profile = DataProfiler().profile(df_with_missing)
        stats = Analytics().analyse(df_with_missing, profile)
        engine = InsightEngine()
        recs = engine._recommendations(profile, stats, [])
        if profile.missing_pct > 10:
            assert any("missing" in r.title.lower() for r in recs)

    def test_recommendations_skewed(self, medium_df) -> None:
        profile = DataProfiler().profile(medium_df)
        stats = Analytics().analyse(medium_df, profile)
        engine = InsightEngine()
        recs = engine._recommendations(profile, stats, [])
        skewed = [s for s in profile.numerical_stats if abs(s.skewness) > 1]
        if skewed:
            assert any("skew" in r.title.lower() or "transform" in r.title.lower() for r in recs)

    def test_executive_summary_narrative(self, medium_df) -> None:
        profile = DataProfiler().profile(medium_df)
        stats = Analytics().analyse(medium_df, profile)
        engine = InsightEngine()
        es = engine._executive_summary(profile, stats, [])
        assert "rows" in es.narrative
        assert "columns" in es.narrative
        assert "health score" in es.narrative.lower()

    def test_executive_summary_health_label(self, simple_df) -> None:
        profile = DataProfiler().profile(simple_df)
        stats = Analytics().analyse(simple_df, profile)
        engine = InsightEngine()
        es = engine._executive_summary(profile, stats, [])
        health_finding = [f for f in es.key_findings if "health score" in f.lower()]
        assert len(health_finding) > 0
        assert profile.health_score.label in ("Excellent", "Good", "Fair", "Poor")

    def test_viz_recommendations_histograms(self, simple_df) -> None:
        profile = DataProfiler().profile(simple_df)
        stats = Analytics().analyse(simple_df, profile)
        engine = InsightEngine()
        viz_recs = engine._viz_recommendations(profile, stats)
        assert any(vr.plot_type == "Histograms" for vr in viz_recs)

    def test_viz_recommendations_heatmap(self, medium_df) -> None:
        profile = DataProfiler().profile(medium_df)
        stats = Analytics().analyse(medium_df, profile)
        engine = InsightEngine()
        viz_recs = engine._viz_recommendations(profile, stats)
        if len(profile.numerical_columns) >= 3:
            assert any(vr.plot_type == "Correlation Heatmap" for vr in viz_recs)

    def test_viz_recommendations_countplots(self, medium_df) -> None:
        profile = DataProfiler().profile(medium_df)
        stats = Analytics().analyse(medium_df, profile)
        engine = InsightEngine()
        viz_recs = engine._viz_recommendations(profile, stats)
        if profile.categorical_columns:
            assert any(vr.plot_type == "Count Plots" for vr in viz_recs)

    def test_viz_recommendations_pairplot(self) -> None:
        df = pd.DataFrame(
            {
                "a": np.random.randn(100),
                "b": np.random.randn(100),
                "c": np.random.randn(100),
            }
        )
        profile = DataProfiler().profile(df)
        stats = Analytics().analyse(df, profile)
        engine = InsightEngine()
        viz_recs = engine._viz_recommendations(profile, stats)
        assert any(vr.plot_type == "Pair Plot" for vr in viz_recs)

    def test_insight_source_metric_traceable(self, medium_df) -> None:
        profile = DataProfiler().profile(medium_df)
        stats = Analytics().analyse(medium_df, profile)
        engine = InsightEngine()
        result = engine.generate(medium_df, profile, stats)
        for ins in result.insights:
            assert ins.source_metric != ""

    def test_empty_dataset_no_crash(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        profile = DataProfiler().profile(df)
        stats = Analytics().analyse(df, profile)
        engine = InsightEngine()
        result = engine.generate(df, profile, stats)
        assert isinstance(result, InsightResult)
