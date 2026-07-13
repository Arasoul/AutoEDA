"""Tests for autoeda.config."""

from __future__ import annotations

from pathlib import Path

import pytest

from autoeda.config import AutoEDAConfig


class TestAutoEDAConfig:
    """Test AutoEDAConfig creation and validation."""

    def test_defaults(self) -> None:
        cfg = AutoEDAConfig()
        assert cfg.enable_plots is True
        assert cfg.figure_width == 12.0
        assert cfg.figure_height == 6.0
        assert cfg.figure_dpi == 150
        assert cfg.color_palette == "viridis"
        assert cfg.correlation_threshold == 0.7
        assert cfg.confidence_level == 0.95
        assert cfg.significance_level == 0.05
        assert isinstance(cfg.output_dir, Path)
        assert cfg.save_figures is True
        assert cfg.save_reports is True
        assert "html" in cfg.report_formats
        assert cfg.max_categories == 20
        assert cfg.top_n_categories == 10

    def test_custom_values(self) -> None:
        cfg = AutoEDAConfig(
            enable_plots=False,
            figure_width=16.0,
            figure_height=8.0,
            figure_dpi=300,
            color_palette="Set2",
            correlation_threshold=0.8,
            confidence_level=0.99,
            significance_level=0.01,
            output_dir="custom_output",
            save_figures=False,
            save_reports=False,
            report_formats=["html", "pdf", "markdown"],
            max_categories=30,
            top_n_categories=5,
        )
        assert cfg.enable_plots is False
        assert cfg.figure_width == 16.0
        assert cfg.output_dir == Path("custom_output")
        assert cfg.save_figures is False
        assert cfg.report_formats == ["html", "pdf", "markdown"]
        assert cfg.max_categories == 30

    def test_output_dir_string_coerced_to_path(self) -> None:
        cfg = AutoEDAConfig(output_dir="some/string/path")
        assert isinstance(cfg.output_dir, Path)
        assert cfg.output_dir == Path("some/string/path")

    def test_invalid_confidence_level(self) -> None:
        with pytest.raises(ValueError, match="confidence_level"):
            AutoEDAConfig(confidence_level=0.0)
        with pytest.raises(ValueError, match="confidence_level"):
            AutoEDAConfig(confidence_level=1.0)

    def test_invalid_significance_level(self) -> None:
        with pytest.raises(ValueError, match="significance_level"):
            AutoEDAConfig(significance_level=0.0)
        with pytest.raises(ValueError, match="significance_level"):
            AutoEDAConfig(significance_level=1.0)

    def test_invalid_figure_dimensions(self) -> None:
        with pytest.raises(ValueError, match="figure_width and figure_height"):
            AutoEDAConfig(figure_width=0)
        with pytest.raises(ValueError, match="figure_width and figure_height"):
            AutoEDAConfig(figure_height=-1)

    def test_invalid_dpi(self) -> None:
        with pytest.raises(ValueError, match="figure_dpi"):
            AutoEDAConfig(figure_dpi=0)

    def test_invalid_correlation_threshold(self) -> None:
        with pytest.raises(ValueError, match="correlation_threshold"):
            AutoEDAConfig(correlation_threshold=1.5)
        with pytest.raises(ValueError, match="correlation_threshold"):
            AutoEDAConfig(correlation_threshold=-2)

    def test_invalid_report_format(self) -> None:
        with pytest.raises(ValueError, match="Unsupported report format"):
            AutoEDAConfig(report_formats=["xlsx"])
