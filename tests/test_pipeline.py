"""End-to-end pipeline test for AutoEDA."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autoeda import AutoEDA
from autoeda.analytics import StatisticalAnalysis
from autoeda.config import AutoEDAConfig
from autoeda.insight_engine import InsightResult
from autoeda.profiler import DatasetProfile, HealthScore
from autoeda.report_generator import ReportResult
from autoeda.visualization import VisualizationResult


class TestAutoEDAPipeline:
    @pytest.fixture()
    def full_pipeline(self, medium_df, tmp_path):
        cfg = AutoEDAConfig(
            output_dir=tmp_path / "autoeda_pipeline_test",
            report_formats=["html", "markdown"],
            save_figures=True,
            save_reports=True,
        )
        pipeline = AutoEDA(cfg)
        results = pipeline.run(medium_df)
        return results

    def test_pipeline_returns_dict(self, full_pipeline) -> None:
        assert isinstance(full_pipeline, dict)
        assert "profile" in full_pipeline
        assert "statistics" in full_pipeline
        assert "figures" in full_pipeline
        assert "insights" in full_pipeline
        assert "report_paths" in full_pipeline

    def test_profile_type(self, full_pipeline) -> None:
        assert isinstance(full_pipeline["profile"], DatasetProfile)

    def test_statistics_type(self, full_pipeline) -> None:
        assert isinstance(full_pipeline["statistics"], StatisticalAnalysis)

    def test_figures_type(self, full_pipeline) -> None:
        assert isinstance(full_pipeline["figures"], VisualizationResult)

    def test_insights_type(self, full_pipeline) -> None:
        assert isinstance(full_pipeline["insights"], InsightResult)

    def test_report_paths_type(self, full_pipeline) -> None:
        assert isinstance(full_pipeline["report_paths"], ReportResult)

    def test_profile_contents(self, full_pipeline) -> None:
        p = full_pipeline["profile"]
        assert p.n_rows == 200
        assert p.n_columns == 5
        assert len(p.numerical_stats) > 0
        assert len(p.categorical_stats) > 0
        assert len(p.variables) == 5
        assert isinstance(p.health_score, HealthScore)

    def test_statistics_contents(self, full_pipeline) -> None:
        s = full_pipeline["statistics"]
        assert s.pearson.method == "pearson"
        assert s.spearman.method == "spearman"
        assert s.kendall.method == "kendall"
        assert len(s.confidence_intervals) > 0
        assert len(s.normality.tests) > 0

    def test_insights_contents(self, full_pipeline) -> None:
        ins = full_pipeline["insights"]
        assert len(ins.insights) > 0
        assert len(ins.recommendations) > 0
        assert len(ins.viz_recommendations) > 0
        assert ins.executive_summary.narrative != ""

    def test_reports_created(self, full_pipeline) -> None:
        r = full_pipeline["report_paths"]
        assert r.html is not None and r.html.exists()
        assert r.markdown is not None and r.markdown.exists()

    def test_pipeline_with_no_categorical(self, tmp_path) -> None:
        df = pd.DataFrame(
            {
                "x": np.random.randn(100),
                "y": np.random.randn(100),
                "z": np.random.randn(100),
            }
        )
        cfg = AutoEDAConfig(output_dir=tmp_path / "no_cat")
        results = AutoEDA(cfg).run(df)
        assert results["profile"].n_rows == 100
        assert results["profile"].categorical_columns == []

    def test_pipeline_with_no_numeric(self, tmp_path) -> None:
        df = pd.DataFrame(
            {
                "color": ["red", "blue", "green"] * 30,
                "size": ["S", "M", "L"] * 30,
            }
        )
        cfg = AutoEDAConfig(output_dir=tmp_path / "no_num")
        results = AutoEDA(cfg).run(df)
        assert results["profile"].numerical_columns == []

    def test_pipeline_small_dataset(self, tmp_path) -> None:
        df = pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5],
                "b": ["x", "y", "x", "y", "x"],
            }
        )
        cfg = AutoEDAConfig(output_dir=tmp_path / "small")
        results = AutoEDA(cfg).run(df)
        assert results["profile"].n_rows == 5

    def test_pipeline_with_missing_data(self, tmp_path) -> None:
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "a": np.random.randn(100),
                "b": np.random.randn(100),
                "cat": np.random.choice(["X", "Y"], 100),
            }
        )
        df.loc[df.sample(frac=0.20).index, "a"] = np.nan
        cfg = AutoEDAConfig(output_dir=tmp_path / "missing")
        results = AutoEDA(cfg).run(df)
        assert results["profile"].missing_pct > 0

    def test_pipeline_invalid_input_type(self) -> None:
        with pytest.raises(TypeError):
            AutoEDA().run([1, 2, 3])  # type: ignore[arg-type]

    def test_pipeline_invalid_input_empty(self) -> None:
        with pytest.raises(ValueError):
            AutoEDA().run(pd.DataFrame())

    def test_pipeline_disabled_plots(self, medium_df, tmp_path) -> None:
        cfg = AutoEDAConfig(
            enable_plots=False,
            output_dir=tmp_path / "no_plots",
        )
        results = AutoEDA(cfg).run(medium_df)
        assert results["figures"].count == 0

    def test_pipeline_disabled_reports(self, medium_df, tmp_path) -> None:
        cfg = AutoEDAConfig(
            save_reports=False,
            output_dir=tmp_path / "no_reports",
        )
        results = AutoEDA(cfg).run(medium_df)
        assert results["report_paths"].html is None
        assert results["report_paths"].markdown is None

    def test_pipeline_custom_config(self, medium_df, tmp_path) -> None:
        cfg = AutoEDAConfig(
            correlation_threshold=0.5,
            significance_level=0.01,
            confidence_level=0.99,
            output_dir=tmp_path / "custom_cfg",
        )
        results = AutoEDA(cfg).run(medium_df)
        assert results["statistics"].pearson.n_significant >= 0

    def test_pipeline_idempotent(self, medium_df, tmp_path) -> None:
        cfg = AutoEDAConfig(output_dir=tmp_path / "idem")
        r1 = AutoEDA(cfg).run(medium_df)
        r2 = AutoEDA(cfg).run(medium_df)
        assert r1["profile"].n_rows == r2["profile"].n_rows
        assert r1["profile"].health_score.overall == r2["profile"].health_score.overall
