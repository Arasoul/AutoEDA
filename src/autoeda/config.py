"""Configuration objects for AutoEDA."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class AutoEDAConfig:
    """Central configuration for all AutoEDA analyses and reports.

    Attributes:
        enable_plots: Whether to generate visualizations.
        figure_width: Width of generated figures in inches.
        figure_height: Height of generated figures in inches.
        figure_dpi: Dots per inch for saved figures.
        color_palette: Seaborn / matplotlib color palette name.
        correlation_threshold: Absolute correlation above which a pair is
            flagged as strongly correlated.
        near_perfect_correlation_threshold: Absolute correlation above
            which a pair is considered near-perfect (redundancy risk).
        confidence_level: Confidence level for interval estimates (0-1).
        significance_level: p-value threshold for statistical tests.
        output_dir: Directory for saving reports and figures.
        save_figures: Whether to persist generated figures to disk.
        save_reports: Whether to persist generated reports to disk.
        report_formats: Which report formats to generate.
            Supported: ``"html"``, ``"pdf"``, ``"markdown"``.
        report_filename: Base filename for generated reports (no extension).
        max_categories: Maximum unique values a numeric column can have
            before it is treated as categorical in some analyses.
        top_n_categories: Number of top categories to display in counts.
        outlier_iqr_multiplier: IQR multiplier for outlier detection.
        identifier_unique_ratio: Unique ratio threshold for identifier
            detection (numeric columns with ratio above this are flagged).
        identifier_min_count: Minimum non-null count for identifier
            detection.
        text_unique_ratio_threshold: Unique ratio threshold for text
            detection (string columns with ratio above this are flagged).
        text_avg_length_threshold: Average string length threshold for
            text detection.
        normality_min_sample_shapiro: Minimum sample size to run
            Shapiro-Wilk normality test.
        normality_min_sample_dagostino: Minimum sample size to run
            D'Agostino normality test.
        skewness_threshold: Absolute skewness above which a distribution
            is considered heavily skewed.
        missing_critical_threshold: Missing percentage above which a
            column is considered critically incomplete.
        missing_drop_threshold: Missing percentage above which a column
            should be considered for dropping.
        missing_warning_threshold: Missing percentage above which a
            warning is issued.
        dominant_category_threshold: Percentage above which a single
            category is considered dominant.
        max_insights_per_category: Maximum insights shown per category.
        max_recommendations: Maximum recommendations shown.
        max_missing_insights: Maximum missing-data insights shown.
        histogram_bins: Number of bins for histograms.
        max_scatter_pairs: Maximum scatterplot pairs shown.
        pairplot_max_columns: Maximum columns in a pairplot.
        timeseries_max_columns: Maximum timeseries subplots.
        max_grid_columns: Maximum columns in subplot grids.
        title_fontsize: Font size for plot titles.
        label_fontsize: Font size for axis labels.
    """

    enable_plots: bool = True
    figure_width: float = 12.0
    figure_height: float = 6.0
    figure_dpi: int = 150
    color_palette: str = "viridis"
    correlation_threshold: float = 0.7
    near_perfect_correlation_threshold: float = 0.9
    confidence_level: float = 0.95
    significance_level: float = 0.05
    output_dir: Path = field(default_factory=lambda: Path("reports"))
    save_figures: bool = True
    save_reports: bool = True
    report_formats: list[str] = field(
        default_factory=lambda: ["html", "markdown"],
    )
    report_filename: str = "autoeda_report"
    max_categories: int = 20
    top_n_categories: int = 10

    # Outlier detection
    outlier_iqr_multiplier: float = 1.5

    # Variable classification thresholds
    identifier_unique_ratio: float = 0.98
    identifier_min_count: int = 10
    text_unique_ratio_threshold: float = 0.9
    text_avg_length_threshold: int = 20

    # Normality test minimums
    normality_min_sample_shapiro: int = 8
    normality_min_sample_dagostino: int = 20

    # Insight thresholds
    skewness_threshold: float = 1.0
    missing_critical_threshold: float = 30.0
    missing_drop_threshold: float = 50.0
    missing_warning_threshold: float = 10.0
    dominant_category_threshold: float = 80.0
    max_insights_per_category: int = 5
    max_recommendations: int = 3
    max_missing_insights: int = 3

    # Visualization layout
    histogram_bins: int = 30
    max_scatter_pairs: int = 4
    pairplot_max_columns: int = 5
    timeseries_max_columns: int = 4
    max_grid_columns: int = 3
    title_fontsize: int = 14
    label_fontsize: int = 11

    def __post_init__(self) -> None:
        """Validate configuration values after initialisation."""
        if isinstance(self.output_dir, str):
            object.__setattr__(self, "output_dir", Path(self.output_dir))
        if not 0 < self.confidence_level < 1:
            msg = "confidence_level must be between 0 and 1 (exclusive)"
            raise ValueError(msg)
        if not 0 < self.significance_level < 1:
            msg = "significance_level must be between 0 and 1 (exclusive)"
            raise ValueError(msg)
        if self.figure_width <= 0 or self.figure_height <= 0:
            msg = "figure_width and figure_height must be positive"
            raise ValueError(msg)
        if self.figure_dpi <= 0:
            msg = "figure_dpi must be positive"
            raise ValueError(msg)
        if not -1 <= self.correlation_threshold <= 1:
            msg = "correlation_threshold must be between -1 and 1"
            raise ValueError(msg)
        if not -1 <= self.near_perfect_correlation_threshold <= 1:
            msg = "near_perfect_correlation_threshold must be between -1 and 1"
            raise ValueError(msg)
        if self.outlier_iqr_multiplier <= 0:
            msg = "outlier_iqr_multiplier must be positive"
            raise ValueError(msg)
        if self.max_categories < 1:
            msg = "max_categories must be at least 1"
            raise ValueError(msg)
        if self.top_n_categories < 1:
            msg = "top_n_categories must be at least 1"
            raise ValueError(msg)
        if self.skewness_threshold <= 0:
            msg = "skewness_threshold must be positive"
            raise ValueError(msg)
        if not 0 < self.missing_critical_threshold <= 100:
            msg = "missing_critical_threshold must be between 0 and 100"
            raise ValueError(msg)
        if not 0 < self.missing_drop_threshold <= 100:
            msg = "missing_drop_threshold must be between 0 and 100"
            raise ValueError(msg)
        if not 0 < self.missing_warning_threshold <= 100:
            msg = "missing_warning_threshold must be between 0 and 100"
            raise ValueError(msg)
        if not 0 < self.dominant_category_threshold <= 100:
            msg = "dominant_category_threshold must be between 0 and 100"
            raise ValueError(msg)
        if self.max_insights_per_category < 1:
            msg = "max_insights_per_category must be at least 1"
            raise ValueError(msg)
        if self.max_recommendations < 1:
            msg = "max_recommendations must be at least 1"
            raise ValueError(msg)
        if self.max_missing_insights < 1:
            msg = "max_missing_insights must be at least 1"
            raise ValueError(msg)
        if self.histogram_bins < 1:
            msg = "histogram_bins must be at least 1"
            raise ValueError(msg)
        if self.max_scatter_pairs < 1:
            msg = "max_scatter_pairs must be at least 1"
            raise ValueError(msg)
        if self.pairplot_max_columns < 2:
            msg = "pairplot_max_columns must be at least 2"
            raise ValueError(msg)
        if self.normality_min_sample_shapiro < 3:
            msg = "normality_min_sample_shapiro must be at least 3"
            raise ValueError(msg)
        if self.normality_min_sample_dagostino < 3:
            msg = "normality_min_sample_dagostino must be at least 3"
            raise ValueError(msg)
        supported = {"html", "pdf", "markdown"}
        for fmt in self.report_formats:
            if fmt not in supported:
                msg = (
                    f"Unsupported report format '{fmt}'. "
                    f"Choose from: {sorted(supported)}"
                )
                raise ValueError(msg)
        logger.debug("AutoEDAConfig initialised: %s", self)
