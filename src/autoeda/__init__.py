"""AutoEDA - Automated Exploratory Data Analysis.

A production-quality Python package that performs professional EDA,
statistical analysis, visualisation, business insight generation,
and executive reporting.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from autoeda._version import __version__
from autoeda.analytics import (
    Analytics,
    ConfidenceIntervalResult,
    CorrelationPair,
    CorrelationResult,
    CovarianceResult,
    HypothesisTestResult,
    NormalityResult,
    NormalityTest,
    StatisticalAnalysis,
)
from autoeda.config import AutoEDAConfig
from autoeda.exceptions import (
    AnalysisError,
    AutoEDAError,
    EmptyDataFrameError,
    InvalidConfigError,
    ReportGenerationError,
    VisualizationError,
)
from autoeda.insight_engine import (
    ExecutiveSummary,
    Insight,
    InsightEngine,
    InsightResult,
    Recommendation,
    VisualizationRecommendation,
)
from autoeda.profiler import (
    CategoricalColumnStats,
    DataProfiler,
    DatasetProfile,
    HealthScore,
    NumericalColumnStats,
    VariableInfo,
)
from autoeda.report_generator import ReportGenerator, ReportResult
from autoeda.utils import validate_dataframe
from autoeda.visualization import FigureResult, Visualization, VisualizationResult

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "__version__",
    "Analytics",
    "AutoEDA",
    "AutoEDAConfig",
    "AutoEDAError",
    "AnalysisError",
    "CategoricalColumnStats",
    "ConfidenceIntervalResult",
    "CorrelationPair",
    "CorrelationResult",
    "CovarianceResult",
    "DataProfiler",
    "DatasetProfile",
    "EmptyDataFrameError",
    "ExecutiveSummary",
    "FigureResult",
    "HealthScore",
    "HypothesisTestResult",
    "Insight",
    "InsightEngine",
    "InsightResult",
    "InvalidConfigError",
    "NormalityResult",
    "NormalityTest",
    "NumericalColumnStats",
    "Recommendation",
    "ReportGenerationError",
    "ReportResult",
    "ReportGenerator",
    "StatisticalAnalysis",
    "VariableInfo",
    "Visualization",
    "VisualizationError",
    "VisualizationRecommendation",
    "VisualizationResult",
]


class AutoEDA:
    """High-level entry point that orchestrates the full EDA pipeline.

    Example::

        from autoeda import AutoEDA

        results = AutoEDA().run(df)
    """

    def __init__(self, config: AutoEDAConfig | None = None) -> None:
        """Initialise AutoEDA with an optional configuration.

        Args:
            config: Analysis configuration.  Uses defaults when *None*.
        """
        self.config = config or AutoEDAConfig()
        self._profiler = DataProfiler(self.config)
        self._analytics = Analytics(self.config)
        self._visualization = Visualization(self.config)
        self._insight_engine = InsightEngine(self.config)
        self._report_generator = ReportGenerator(self.config)
        logger.info("AutoEDA v%s initialised", __version__)

    def run(self, df: pd.DataFrame) -> dict[str, Any]:
        """Execute the complete EDA pipeline on a cleaned DataFrame.

        Args:
            df: A cleaned pandas DataFrame ready for analysis.

        Returns:
            A dictionary containing every analysis result produced by
            the pipeline (profile, statistics, insights, report paths).

        Raises:
            TypeError: If *df* is not a :class:`pandas.DataFrame`.
            ValueError: If *df* is empty.
        """
        validate_dataframe(df)
        logger.info(
            "Starting EDA on DataFrame with %d rows x %d columns",
            len(df),
            len(df.columns),
        )

        results: dict[str, Any] = {}

        # Phase 2 — profiling
        results["profile"] = self._profiler.profile(df)

        # Phase 3 — statistics
        results["statistics"] = self._analytics.analyse(df, results["profile"])

        # Phase 4 — visualisations
        results["figures"] = self._visualization.generate_all(
            df,
            results["profile"],
            results["statistics"],
        )

        # Phase 5 — business insights
        results["insights"] = self._insight_engine.generate(
            df,
            results["profile"],
            results["statistics"],
            results["figures"],
        )

        # Phase 6 — reports
        results["report_paths"] = self._report_generator.generate(
            results["profile"],
            results["statistics"],
            results["figures"],
            results["insights"],
        )

        logger.info("EDA pipeline completed successfully")
        return results
