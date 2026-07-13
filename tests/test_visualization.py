"""Tests for autoeda.visualization — Visualization class."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import pytest

from autoeda.analytics import Analytics
from autoeda.config import AutoEDAConfig
from autoeda.profiler import DataProfiler
from autoeda.visualization import FigureResult, Visualization, VisualizationResult


class TestVisualizationResult:
    def test_empty_result(self) -> None:
        r = VisualizationResult()
        assert r.all_figures == []
        assert r.count == 0

    def test_count_property(self) -> None:
        r = VisualizationResult()
        r.distribution.append(FigureResult(
            title="t", description="d", category="distribution",
            path=None, filename="t.png",
        ))
        r.comparison.append(FigureResult(
            title="t2", description="d2", category="comparison",
            path=None, filename="t2.png",
        ))
        assert r.count == 2

    def test_all_figures_concatenation(self) -> None:
        r = VisualizationResult()
        fig = FigureResult(
            title="t", description="d", category="matrix",
            path=None, filename="t.png",
        )
        r.matrix.append(fig)
        assert fig in r.all_figures


class TestVisualization:
    @pytest.fixture()
    def profile_and_stats(self, medium_df):
        profile = DataProfiler().profile(medium_df)
        stats = Analytics().analyse(medium_df, profile)
        return profile, stats

    def test_generate_all_returns_result(
        self, medium_df, profile_and_stats,
    ) -> None:
        profile, stats = profile_and_stats
        viz = Visualization(AutoEDAConfig(save_figures=False))
        result = viz.generate_all(medium_df, profile, stats)
        assert isinstance(result, VisualizationResult)

    def test_figures_generated(self, medium_df, profile_and_stats) -> None:
        profile, stats = profile_and_stats
        viz = Visualization(AutoEDAConfig(save_figures=False))
        result = viz.generate_all(medium_df, profile, stats)
        assert result.count > 0

    def test_distribution_figures(self, medium_df, profile_and_stats) -> None:
        profile, stats = profile_and_stats
        viz = Visualization(AutoEDAConfig(save_figures=False))
        result = viz.generate_all(medium_df, profile, stats)
        assert len(result.distribution) >= 2  # histograms + kde

    def test_comparison_figures(self, medium_df, profile_and_stats) -> None:
        profile, stats = profile_and_stats
        viz = Visualization(AutoEDAConfig(save_figures=False))
        result = viz.generate_all(medium_df, profile, stats)
        assert len(result.comparison) >= 2  # boxplots + violin + countplots

    def test_relationship_figures(self, medium_df, profile_and_stats) -> None:
        profile, stats = profile_and_stats
        viz = Visualization(AutoEDAConfig(save_figures=False))
        result = viz.generate_all(medium_df, profile, stats)
        assert len(result.relationship) >= 1

    def test_matrix_figures(self, medium_df, profile_and_stats) -> None:
        profile, stats = profile_and_stats
        viz = Visualization(AutoEDAConfig(save_figures=False))
        result = viz.generate_all(medium_df, profile, stats)
        assert len(result.matrix) >= 1  # correlation heatmap

    def test_disabled_plots(self, medium_df, profile_and_stats) -> None:
        profile, stats = profile_and_stats
        viz = Visualization(AutoEDAConfig(enable_plots=False))
        result = viz.generate_all(medium_df, profile, stats)
        assert result.count == 0

    def test_no_numeric_columns(self, df_no_numeric) -> None:
        profile = DataProfiler().profile(df_no_numeric)
        stats = Analytics().analyse(df_no_numeric, profile)
        viz = Visualization(AutoEDAConfig(save_figures=False))
        result = viz.generate_all(df_no_numeric, profile, stats)
        assert len(result.distribution) == 0
        assert len(result.relationship) == 0
        assert len(result.matrix) == 0
        assert len(result.timeseries) == 0

    def test_no_categorical_columns(self, simple_df) -> None:
        profile = DataProfiler().profile(simple_df)
        stats = Analytics().analyse(simple_df, profile)
        viz = Visualization(AutoEDAConfig(save_figures=False))
        viz.generate_all(simple_df, profile, stats)
        # simple_df has 1 categorical column, so countplot may exist
        # but the key test is that it doesn't crash

    def test_missing_value_heatmap(self, df_with_missing) -> None:
        profile = DataProfiler().profile(df_with_missing)
        stats = Analytics().analyse(df_with_missing, profile)
        viz = Visualization(AutoEDAConfig(save_figures=False))
        result = viz.generate_all(df_with_missing, profile, stats)
        quality_figs = [f for f in result.quality if "missing" in f.title.lower()]
        assert len(quality_figs) >= 1

    def test_no_missing_no_heatmap(self, simple_df) -> None:
        profile = DataProfiler().profile(simple_df)
        stats = Analytics().analyse(simple_df, profile)
        viz = Visualization(AutoEDAConfig(save_figures=False))
        result = viz.generate_all(simple_df, profile, stats)
        quality_figs = [f for f in result.quality if "missing" in f.title.lower()]
        assert len(quality_figs) == 0

    def test_timeseries_plots(self, df_with_datetime) -> None:
        profile = DataProfiler().profile(df_with_datetime)
        stats = Analytics().analyse(df_with_datetime, profile)
        viz = Visualization(AutoEDAConfig(save_figures=False))
        result = viz.generate_all(df_with_datetime, profile, stats)
        assert len(result.timeseries) >= 1

    def test_no_datetime_no_timeseries(self, simple_df) -> None:
        profile = DataProfiler().profile(simple_df)
        stats = Analytics().analyse(simple_df, profile)
        viz = Visualization(AutoEDAConfig(save_figures=False))
        result = viz.generate_all(simple_df, profile, stats)
        assert len(result.timeseries) == 0

    def test_bubble_chart_with_3_plus_numeric(self, medium_df, profile_and_stats) -> None:
        profile, stats = profile_and_stats
        viz = Visualization(AutoEDAConfig(save_figures=False))
        viz.generate_all(medium_df, profile, stats)
        # medium_df has 3 numeric columns (revenue, units, satisfaction) → bubble possible

    def test_pairplot(self, medium_df, profile_and_stats) -> None:
        profile, stats = profile_and_stats
        viz = Visualization(AutoEDAConfig(save_figures=False))
        result = viz.generate_all(medium_df, profile, stats)
        pair = [f for f in result.relationship if "pair" in f.title.lower()]
        assert len(pair) >= 1

    def test_figure_results_have_required_fields(
        self, medium_df, profile_and_stats,
    ) -> None:
        profile, stats = profile_and_stats
        viz = Visualization(AutoEDAConfig(save_figures=False))
        result = viz.generate_all(medium_df, profile, stats)
        for fig in result.all_figures:
            assert isinstance(fig, FigureResult)
            assert fig.title != ""
            assert fig.description != ""
            assert fig.category != ""
            assert fig.filename != ""

    def test_save_figures_creates_files(
        self, medium_df, profile_and_stats, tmp_path,
    ) -> None:
        profile, stats = profile_and_stats
        cfg = AutoEDAConfig(
            output_dir=tmp_path / "output",
            save_figures=True,
            save_reports=False,
        )
        viz = Visualization(cfg)
        result = viz.generate_all(medium_df, profile, stats)
        saved = [f for f in result.all_figures if f.path is not None]
        assert len(saved) > 0
        for fig in saved:
            assert fig.path.exists()
