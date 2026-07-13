"""Tests for autoeda.report_generator — ReportGenerator, HTML, Markdown."""

from __future__ import annotations

import pytest

from autoeda.analytics import Analytics
from autoeda.config import AutoEDAConfig
from autoeda.insight_engine import InsightEngine
from autoeda.profiler import DataProfiler
from autoeda.report_generator import ReportGenerator, ReportResult
from autoeda.visualization import Visualization


@pytest.fixture()
def full_results(medium_df, tmp_path):
    """Run the full pipeline and return results for report generation."""
    cfg = AutoEDAConfig(
        output_dir=tmp_path / "reports",
        save_figures=True,
        save_reports=True,
        report_formats=["html", "markdown"],
    )
    profile = DataProfiler(cfg).profile(medium_df)
    stats = Analytics(cfg).analyse(medium_df, profile)
    figures = Visualization(cfg).generate_all(medium_df, profile, stats)
    insights = InsightEngine(cfg).generate(medium_df, profile, stats)
    return profile, stats, figures, insights, cfg


class TestReportResult:
    def test_default_none(self) -> None:
        r = ReportResult()
        assert r.html is None
        assert r.pdf is None
        assert r.markdown is None

    def test_with_paths(self, tmp_path) -> None:
        r = ReportResult(html=tmp_path / "a.html", markdown=tmp_path / "a.md")
        assert r.html == tmp_path / "a.html"
        assert r.markdown == tmp_path / "a.md"
        assert r.pdf is None


class TestReportGenerator:
    def test_generate_returns_result(self, full_results) -> None:
        profile, stats, figures, insights, cfg = full_results
        gen = ReportGenerator(cfg)
        result = gen.generate(profile, stats, figures, insights)
        assert isinstance(result, ReportResult)

    def test_html_report_created(self, full_results) -> None:
        profile, stats, figures, insights, cfg = full_results
        result = ReportGenerator(cfg).generate(profile, stats, figures, insights)
        assert result.html is not None
        assert result.html.exists()
        assert result.html.suffix == ".html"

    def test_markdown_report_created(self, full_results) -> None:
        profile, stats, figures, insights, cfg = full_results
        result = ReportGenerator(cfg).generate(profile, stats, figures, insights)
        assert result.markdown is not None
        assert result.markdown.exists()
        assert result.markdown.suffix == ".md"

    def test_html_contains_key_sections(self, full_results) -> None:
        profile, stats, figures, insights, cfg = full_results
        result = ReportGenerator(cfg).generate(profile, stats, figures, insights)
        content = result.html.read_text(encoding="utf-8")
        assert "Executive Summary" in content
        assert "Dataset Overview" in content
        assert "Data Health Score" in content
        assert "Variable Summary" in content
        assert "Statistical Analysis" in content
        assert "Correlation Analysis" in content
        assert "Distribution Analysis" in content
        assert "Outlier Analysis" in content
        assert "Visualisations" in content
        assert "Key Insights" in content
        assert "Appendix" in content

    def test_html_contains_kpi_cards(self, full_results) -> None:
        profile, stats, figures, insights, cfg = full_results
        result = ReportGenerator(cfg).generate(profile, stats, figures, insights)
        content = result.html.read_text(encoding="utf-8")
        assert "Health Score" in content
        assert "Numeric" in content
        assert "Categorical" in content

    def test_html_contains_health_bar(self, full_results) -> None:
        profile, stats, figures, insights, cfg = full_results
        result = ReportGenerator(cfg).generate(profile, stats, figures, insights)
        content = result.html.read_text(encoding="utf-8")
        assert "health-bar" in content

    def test_html_contains_narrative(self, full_results) -> None:
        profile, stats, figures, insights, cfg = full_results
        result = ReportGenerator(cfg).generate(profile, stats, figures, insights)
        content = result.html.read_text(encoding="utf-8")
        narrative = insights.executive_summary.narrative
        assert narrative[:50] in content

    def test_html_contains_variable_table(self, full_results) -> None:
        profile, stats, figures, insights, cfg = full_results
        result = ReportGenerator(cfg).generate(profile, stats, figures, insights)
        content = result.html.read_text(encoding="utf-8")
        assert "semantic_type" in content or "suggested_role" in content or "Variable" in content

    def test_html_contains_correlation_interpretation(self, full_results) -> None:
        profile, stats, figures, insights, cfg = full_results
        result = ReportGenerator(cfg).generate(profile, stats, figures, insights)
        content = result.html.read_text(encoding="utf-8")
        for p in stats.pearson.significant_pairs:
            if p.interpretation:
                assert p.interpretation in content

    def test_html_contains_viz_recommendations(self, full_results) -> None:
        profile, stats, figures, insights, cfg = full_results
        result = ReportGenerator(cfg).generate(profile, stats, figures, insights)
        content = result.html.read_text(encoding="utf-8")
        assert "Suggested Visualisations" in content

    def test_markdown_contains_executive_summary(self, full_results) -> None:
        profile, stats, figures, insights, cfg = full_results
        result = ReportGenerator(cfg).generate(profile, stats, figures, insights)
        content = result.markdown.read_text(encoding="utf-8")
        assert "Executive Summary" in content
        assert insights.executive_summary.narrative[:50] in content

    def test_markdown_contains_variable_summary(self, full_results) -> None:
        profile, stats, figures, insights, cfg = full_results
        result = ReportGenerator(cfg).generate(profile, stats, figures, insights)
        content = result.markdown.read_text(encoding="utf-8")
        assert "Variable Summary" in content

    def test_disabled_saving(self, medium_df) -> None:
        profile = DataProfiler().profile(medium_df)
        stats = Analytics().analyse(medium_df, profile)
        figures = Visualization().generate_all(medium_df, profile, stats)
        insights = InsightEngine().generate(medium_df, profile, stats)
        cfg = AutoEDAConfig(save_reports=False)
        result = ReportGenerator(cfg).generate(profile, stats, figures, insights)
        assert result.html is None
        assert result.pdf is None
        assert result.markdown is None

    def test_html_version_in_footer(self, full_results) -> None:
        profile, stats, figures, insights, cfg = full_results
        result = ReportGenerator(cfg).generate(profile, stats, figures, insights)
        content = result.html.read_text(encoding="utf-8")
        assert "AutoEDA v" in content

    def test_html_inline_base64_images(self, full_results) -> None:
        profile, stats, figures, insights, cfg = full_results
        result = ReportGenerator(cfg).generate(profile, stats, figures, insights)
        content = result.html.read_text(encoding="utf-8")
        if figures.all_figures:
            assert "data:image/png;base64," in content

    def test_markdown_contains_correlations(self, full_results) -> None:
        profile, stats, figures, insights, cfg = full_results
        result = ReportGenerator(cfg).generate(profile, stats, figures, insights)
        content = result.markdown.read_text(encoding="utf-8")
        if stats.pearson.significant_pairs:
            assert "Correlation Analysis" in content

    def test_markdown_contains_insights(self, full_results) -> None:
        profile, stats, figures, insights, cfg = full_results
        result = ReportGenerator(cfg).generate(profile, stats, figures, insights)
        content = result.markdown.read_text(encoding="utf-8")
        assert "Business Insights" in content

    def test_markdown_contains_viz_recommendations(self, full_results) -> None:
        profile, stats, figures, insights, cfg = full_results
        result = ReportGenerator(cfg).generate(profile, stats, figures, insights)
        content = result.markdown.read_text(encoding="utf-8")
        assert "Suggested Visualisations" in content
