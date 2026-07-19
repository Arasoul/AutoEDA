"""Business insight generation: rule-based insights, recommendations, and executive summary.

This module performs the fourth stage of the EDA pipeline.  It consumes
the results of Phases 2-4 and produces deterministic, traceable
business insights and actionable recommendations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from autoeda.config import AutoEDAConfig
from autoeda.utils import validate_dataframe

if TYPE_CHECKING:
    import pandas as pd

    from autoeda.analytics import StatisticalAnalysis
    from autoeda.profiler import DatasetProfile
    from autoeda.visualization import VisualizationResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Insight:
    """A single data-driven business insight.

    Every insight references the metric or statistical result that
    supports it, ensuring full traceability.

    Attributes:
        category: Domain this insight belongs to (``"distribution"``,
            ``"correlation"``, ``"categorical"``, ``"missing"``,
            ``"outlier"``, ``"normality"``, ``"hypothesis"``).
        severity: Importance level (``"info"``, ``"warning"``,
            ``"critical"``).
        title: Short headline.
        detail: Longer explanation with supporting numbers.
        source_metric: The exact metric or test that produced this insight.
    """

    category: str
    severity: str
    title: str
    detail: str
    source_metric: str


@dataclass(frozen=True)
class Recommendation:
    """An actionable recommendation derived from the analysis.

    Attributes:
        priority: ``"high"``, ``"medium"``, or ``"low"``.
        title: Short headline.
        detail: Explanation of the recommended action.
        related_insight: Title of the insight that triggered this
            recommendation.
    """

    priority: str
    title: str
    detail: str
    related_insight: str


@dataclass(frozen=True)
class ExecutiveSummary:
    """High-level snapshot of the most important findings.

    Attributes:
        dataset_shape: ``"(rows, columns)"`` string.
        total_missing_pct: Percentage of missing cells.
        n_numerical / n_categorical / n_datetime / n_boolean: Column
            type counts.
        n_strong_correlations: Number of strongly correlated pairs.
        n_skewed_features: Features with ``|skewness| > threshold``.
        n_outlier_features: Features with > 0 outliers.
        n_significant_tests: Hypothesis tests that rejected the null.
        key_findings: Top-3 most important findings as strings.
    """

    dataset_shape: str
    total_missing_pct: float
    n_numerical: int
    n_categorical: int
    n_datetime: int
    n_boolean: int
    n_strong_correlations: int
    n_skewed_features: int
    n_outlier_features: int
    n_significant_tests: int
    key_findings: list[str]
    narrative: str


@dataclass(frozen=True)
class VisualizationRecommendation:
    """A recommended visualisation for the dataset.

    Attributes:
        plot_type: Type of plot (e.g. ``"Scatter Plot"``).
        reason: Why this plot is recommended.
        columns: Columns involved.
        priority: ``"high"``, ``"medium"``, or ``"low"``.
    """

    plot_type: str
    reason: str
    columns: list[str]
    priority: str


@dataclass(frozen=True)
class InsightResult:
    """Container for all generated insights and recommendations.

    Attributes:
        insights: All discovered insights.
        recommendations: Actionable recommendations.
        executive_summary: High-level summary.
        viz_recommendations: Suggested visualisations.
    """

    insights: list[Insight]
    recommendations: list[Recommendation]
    executive_summary: ExecutiveSummary
    viz_recommendations: list[VisualizationRecommendation]


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class InsightEngine:
    """Generate rule-based business insights from EDA results.

    Example::

        engine = InsightEngine(config)
        result = engine.generate(df, profile, stats, figures)
    """

    def __init__(self, config: AutoEDAConfig | None = None) -> None:
        self.config = config or AutoEDAConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        df: pd.DataFrame,
        profile: DatasetProfile,
        stats: StatisticalAnalysis,
        figures: VisualizationResult | None = None,
    ) -> InsightResult:
        """Generate all insights from the completed analysis.

        Args:
            df: The cleaned DataFrame.
            profile: Profiling result from Phase 2.
            stats: Statistical analysis from Phase 3.
            figures: Visualisation result from Phase 4.

        Returns:
            An :class:`InsightResult` with insights, recommendations,
            and an executive summary.
        """
        validate_dataframe(df)
        logger.info("Generating business insights")
        insights: list[Insight] = []

        insights.extend(self._distribution_insights(profile))
        insights.extend(self._correlation_insights(stats))
        insights.extend(self._categorical_insights(profile))
        insights.extend(self._missing_insights(profile))
        insights.extend(self._outlier_insights(profile))
        insights.extend(self._normality_insights(stats))
        insights.extend(self._hypothesis_insights(stats))

        recs = self._recommendations(profile, stats, insights)
        viz_recs = self._viz_recommendations(profile, stats)
        exec_sum = self._executive_summary(profile, stats, insights)

        result = InsightResult(
            insights=insights,
            recommendations=recs,
            executive_summary=exec_sum,
            viz_recommendations=viz_recs,
        )
        logger.info(
            "Generated %d insights, %d recommendations, %d viz suggestions",
            len(insights),
            len(recs),
            len(viz_recs),
        )
        return result

    # ------------------------------------------------------------------
    # Distribution insights
    # ------------------------------------------------------------------

    def _distribution_insights(self, profile: DatasetProfile) -> list[Insight]:
        """Detect skewed distributions and high zero-inflation."""
        insights: list[Insight] = []
        skew_thresh = self.config.skewness_threshold
        for s in profile.numerical_stats:
            if abs(s.skewness) > skew_thresh:
                direction = "right" if s.skewness > 0 else "left"
                insights.append(
                    Insight(
                        category="distribution",
                        severity="warning",
                        title=f"'{s.column}' is {direction}-skewed (skewness={s.skewness:.2f})",
                        detail=(
                            f"The distribution of '{s.column}' is heavily "
                            f"skewed to the {direction} with a skewness of {s.skewness:.2f}. "
                            f"Mean ({s.mean:.2f}) differs from median ({s.median:.2f}). "
                            f"Consider a log or Box-Cox transformation for modelling."
                        ),
                        source_metric=f"NumericalColumnStats({s.column}).skewness={s.skewness:.4f}",
                    )
                )
            if s.zero_pct > 20:
                insights.append(
                    Insight(
                        category="distribution",
                        severity="warning",
                        title=f"'{s.column}' has {s.zero_pct:.1f}% zero values",
                        detail=(
                            f"Column '{s.column}' contains {s.n_zeros} zero values "
                            f"({s.zero_pct:.1f}% of non-null entries). This may indicate "
                            f"inflated zeros or data quality issues."
                        ),
                        source_metric=f"NumericalColumnStats({s.column}).zero_pct={s.zero_pct}",
                    )
                )
        return insights

    # ------------------------------------------------------------------
    # Correlation insights
    # ------------------------------------------------------------------

    def _correlation_insights(self, stats: StatisticalAnalysis) -> list[Insight]:
        """Detect strong correlations and potential multicollinearity."""
        insights: list[Insight] = []
        threshold = self.config.correlation_threshold
        np_threshold = self.config.near_perfect_correlation_threshold

        strong_pairs = [
            p for p in stats.pearson.significant_pairs if abs(p.coefficient) >= threshold
        ]
        if strong_pairs:
            top = max(strong_pairs, key=lambda p: abs(p.coefficient))
            insights.append(
                Insight(
                    category="correlation",
                    severity="info",
                    title=f"{len(strong_pairs)} strong correlation(s) detected",
                    detail=(
                        f"Found {len(strong_pairs)} pairs with |r| >= {threshold}. "
                        f"Strongest: {top.col_a} <-> {top.col_b} "
                        f"(r={top.coefficient:.4f}, p={top.p_value:.6f})."
                    ),
                    source_metric=f"CorrelationResult(pearson).significant_pairs (|r| >= {threshold})",
                )
            )

        for p in strong_pairs:
            if abs(p.coefficient) >= np_threshold:
                insights.append(
                    Insight(
                        category="correlation",
                        severity="warning",
                        title=f"Near-perfect correlation: {p.col_a} <-> {p.col_b}",
                        detail=(
                            f"Variables '{p.col_a}' and '{p.col_b}' show a "
                            f"near-perfect correlation (r={p.coefficient:.4f}). "
                            f"This may indicate redundancy or multicollinearity."
                        ),
                        source_metric=f"CorrelationPair({p.col_a}, {p.col_b}, r={p.coefficient})",
                    )
                )
        return insights

    # ------------------------------------------------------------------
    # Categorical insights
    # ------------------------------------------------------------------

    def _categorical_insights(self, profile: DatasetProfile) -> list[Insight]:
        """Detect dominant categories and high-cardinality features."""
        insights: list[Insight] = []
        for s in profile.categorical_stats:
            if s.n_unique == 0:
                continue
            if s.top_pct > self.config.dominant_category_threshold:
                insights.append(
                    Insight(
                        category="categorical",
                        severity="info",
                        title=f"'{s.column}' dominated by '{s.top_value}' ({s.top_pct:.1f}%)",
                        detail=(
                            f"The category '{s.top_value}' accounts for "
                            f"{s.top_pct:.1f}% of all values in '{s.column}'. "
                            f"This heavy concentration may limit the predictive "
                            f"usefulness of this feature."
                        ),
                        source_metric=f"CategoricalColumnStats({s.column}).top_pct={s.top_pct}",
                    )
                )
            if s.n_unique > self.config.max_categories:
                insights.append(
                    Insight(
                        category="categorical",
                        severity="warning",
                        title=f"'{s.column}' has high cardinality ({s.n_unique} unique values)",
                        detail=(
                            f"Column '{s.column}' has {s.n_unique} unique values "
                            f"(unique ratio: {s.unique_ratio:.4f}). High-cardinality "
                            f"categorical features may need grouping for analysis."
                        ),
                        source_metric=f"CategoricalColumnStats({s.column}).n_unique={s.n_unique}",
                    )
                )
        return insights

    # ------------------------------------------------------------------
    # Missing data insights
    # ------------------------------------------------------------------

    def _missing_insights(self, profile: DatasetProfile) -> list[Insight]:
        """Summarise missing data patterns."""
        insights: list[Insight] = []
        if profile.missing_total == 0:
            insights.append(
                Insight(
                    category="missing",
                    severity="info",
                    title="No missing values detected",
                    detail="The dataset is complete with no missing values across all columns.",
                    source_metric="DatasetProfile.missing_total=0",
                )
            )
            return insights

        crit_thresh = self.config.missing_critical_threshold
        drop_thresh = self.config.missing_drop_threshold
        warn_thresh = self.config.missing_warning_threshold

        if profile.missing_pct > crit_thresh:
            insights.append(
                Insight(
                    category="missing",
                    severity="critical",
                    title=f"High overall missing data ({profile.missing_pct:.1f}%)",
                    detail=(
                        f"Total missing cells: {profile.missing_total} "
                        f"({profile.missing_pct:.1f}% of all data). "
                        f"This level of missingness may significantly bias analyses."
                    ),
                    source_metric=f"DatasetProfile.missing_pct={profile.missing_pct}",
                )
            )

        for _, row in profile.missing_by_column.head(self.config.max_missing_insights).iterrows():
            col_name = str(row["column"])
            pct = float(row["missing_pct"])
            if pct > drop_thresh:
                insights.append(
                    Insight(
                        category="missing",
                        severity="critical",
                        title=f"'{col_name}' is >{drop_thresh:.0f}% missing ({pct:.1f}%)",
                        detail=(
                            f"Column '{col_name}' has {pct:.1f}% missing values. "
                            f"Consider dropping this column or imputing carefully."
                        ),
                        source_metric=f"missing_by_column({col_name}).missing_pct={pct}",
                    )
                )
            elif pct > warn_thresh:
                insights.append(
                    Insight(
                        category="missing",
                        severity="warning",
                        title=f"'{col_name}' has {pct:.1f}% missing values",
                        detail=(
                            f"Column '{col_name}' has {pct:.1f}% missing values. "
                            f"Imputation or exclusion should be considered."
                        ),
                        source_metric=f"missing_by_column({col_name}).missing_pct={pct}",
                    )
                )
        return insights

    # ------------------------------------------------------------------
    # Outlier insights
    # ------------------------------------------------------------------

    def _outlier_insights(self, profile: DatasetProfile) -> list[Insight]:
        """Summarise outlier prevalence across numeric columns."""
        insights: list[Insight] = []
        cols_with_outliers = [s for s in profile.numerical_stats if s.n_outliers > 0]
        if not cols_with_outliers:
            return insights

        worst = max(cols_with_outliers, key=lambda s: s.outlier_pct)
        insights.append(
            Insight(
                category="outlier",
                severity="warning",
                title=f"{len(cols_with_outliers)} feature(s) contain outliers",
                detail=(
                    f"{len(cols_with_outliers)} numeric columns have outliers "
                    f"(IQR method). Highest: '{worst.column}' with "
                    f"{worst.n_outliers} outliers ({worst.outlier_pct:.1f}%)."
                ),
                source_metric=(
                    f"NumericalColumnStats({worst.column})."
                    f"n_outliers={worst.n_outliers}, outlier_pct={worst.outlier_pct}"
                ),
            )
        )
        return insights

    # ------------------------------------------------------------------
    # Normality insights
    # ------------------------------------------------------------------

    def _normality_insights(self, stats: StatisticalAnalysis) -> list[Insight]:
        """Summarise normality test findings."""
        insights: list[Insight] = []
        norm = stats.normality
        if not norm.tests:
            return insights

        non_normal = [t for t in norm.tests if not t.is_normal]
        if non_normal:
            max_show = self.config.max_insights_per_category
            cols = ", ".join(f"'{t.column}'" for t in non_normal[:max_show])
            insights.append(
                Insight(
                    category="normality",
                    severity="info",
                    title=f"{len(non_normal)} feature(s) are not normally distributed",
                    detail=(
                        f"Normality tests (Shapiro-Wilk, D'Agostino, KS) rejected "
                        f"normality for: {cols}. "
                        f"Parametric tests may be less reliable for these features."
                    ),
                    source_metric=(
                        f"NormalityResult: n_normal={norm.n_normal}, "
                        f"n_not_normal={norm.n_not_normal}"
                    ),
                )
            )
        return insights

    # ------------------------------------------------------------------
    # Hypothesis test insights
    # ------------------------------------------------------------------

    def _hypothesis_insights(self, stats: StatisticalAnalysis) -> list[Insight]:
        """Summarise significant hypothesis test results."""
        insights: list[Insight] = []
        sig_tests = [t for t in stats.hypothesis_tests if t.significant]
        if not sig_tests:
            return insights

        max_show = self.config.max_insights_per_category
        for t in sig_tests[:max_show]:
            if t.test_name == "chi_square":
                insights.append(
                    Insight(
                        category="hypothesis",
                        severity="info",
                        title=f"Significant association: {t.column} vs {t.categorical_column}",
                        detail=(
                            f"Chi-square test reveals a significant association "
                            f"between '{t.column}' and '{t.categorical_column}' "
                            f"(chi2={t.statistic:.4f}, p={t.p_value:.6f})."
                        ),
                        source_metric=f"HypothesisTestResult(chi_square, {t.column}, {t.categorical_column})",
                    )
                )
            else:
                groups = f"{t.group_a} vs {t.group_b}" if t.group_a else "multiple groups"
                insights.append(
                    Insight(
                        category="hypothesis",
                        severity="info",
                        title=f"Significant difference: {t.column} by {t.categorical_column} ({groups})",
                        detail=(
                            f"{t.test_name.replace('_', ' ').title()} shows a "
                            f"significant difference in '{t.column}' across "
                            f"'{t.categorical_column}' ({groups}, "
                            f"stat={t.statistic:.4f}, p={t.p_value:.6f})."
                        ),
                        source_metric=f"HypothesisTestResult({t.test_name}, {t.column})",
                    )
                )
        return insights

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------

    def _recommendations(
        self,
        profile: DatasetProfile,
        stats: StatisticalAnalysis,
        insights: list[Insight],
    ) -> list[Recommendation]:
        """Derive actionable recommendations from insights."""
        recs: list[Recommendation] = []
        max_recs = self.config.max_recommendations
        skew_thresh = self.config.skewness_threshold
        np_thresh = self.config.near_perfect_correlation_threshold

        # Missing data recommendations
        if profile.missing_pct > self.config.missing_warning_threshold:
            recs.append(
                Recommendation(
                    priority="high",
                    title="Address missing data before analysis",
                    detail=(
                        f"The dataset has {profile.missing_pct:.1f}% missing values. "
                        f"Use DataPrepToolkit to impute or remove affected rows/columns."
                    ),
                    related_insight="missing_insights",
                )
            )

        # Skewness recommendations
        skewed = [s for s in profile.numerical_stats if abs(s.skewness) > skew_thresh]
        if skewed:
            cols = ", ".join(f"'{s.column}'" for s in skewed[:max_recs])
            recs.append(
                Recommendation(
                    priority="medium",
                    title="Consider transforming skewed features",
                    detail=(
                        f"Features {cols} are heavily skewed. "
                        f"Log, square-root, or Box-Cox transformations may improve "
                        f"distributional properties for downstream modelling."
                    ),
                    related_insight="distribution_insights",
                )
            )

        # Correlation recommendations
        strong = [p for p in stats.pearson.significant_pairs if abs(p.coefficient) >= np_thresh]
        if strong:
            recs.append(
                Recommendation(
                    priority="high",
                    title="Investigate near-perfect correlations for redundancy",
                    detail=(
                        f"{len(strong)} pair(s) show |r| >= {np_thresh}. "
                        f"Review whether both variables are needed or if one "
                        f"should be dropped to avoid multicollinearity."
                    ),
                    related_insight="correlation_insights",
                )
            )

        # High-cardinality recommendations
        high_card = [
            s for s in profile.categorical_stats if s.n_unique > self.config.max_categories
        ]
        if high_card:
            cols = ", ".join(f"'{s.column}'" for s in high_card[:max_recs])
            recs.append(
                Recommendation(
                    priority="medium",
                    title="Group high-cardinality categorical features",
                    detail=(
                        f"Features {cols} have more than "
                        f"{self.config.max_categories} unique values. "
                        f"Consider grouping rare categories or using target encoding."
                    ),
                    related_insight="categorical_insights",
                )
            )

        # Outlier recommendations
        outlier_cols = [s for s in profile.numerical_stats if s.n_outliers > 0]
        if outlier_cols:
            recs.append(
                Recommendation(
                    priority="medium",
                    title="Review outlier treatment strategy",
                    detail=(
                        f"{len(outlier_cols)} numeric column(s) contain outliers. "
                        f"Determine whether to cap, transform, or keep them "
                        f"based on domain knowledge."
                    ),
                    related_insight="outlier_insights",
                )
            )

        return recs[:max_recs]

    # ------------------------------------------------------------------
    # Visualisation recommendations
    # ------------------------------------------------------------------

    def _viz_recommendations(
        self,
        profile: DatasetProfile,
        stats: StatisticalAnalysis,
    ) -> list[VisualizationRecommendation]:
        """Suggest which visualisations would be most informative."""
        recs: list[VisualizationRecommendation] = []
        num_cols = profile.numerical_columns
        cat_cols = profile.categorical_columns

        # Always recommend histograms for numeric data
        if num_cols:
            recs.append(
                VisualizationRecommendation(
                    plot_type="Histograms",
                    reason="Reveal distribution shape, central tendency, and spread.",
                    columns=num_cols,
                    priority="high",
                )
            )

        # Correlation heatmap if >= 3 numeric columns
        if len(num_cols) >= 3:
            recs.append(
                VisualizationRecommendation(
                    plot_type="Correlation Heatmap",
                    reason="Identify relationships between all numeric features at a glance.",
                    columns=num_cols,
                    priority="high",
                )
            )

        # Scatterplots if strong correlations exist
        strong = [
            p
            for p in stats.pearson.significant_pairs
            if abs(p.coefficient) >= self.config.correlation_threshold
        ]
        if strong:
            cols = list({c for p in strong[:4] for c in (p.col_a, p.col_b)})
            recs.append(
                VisualizationRecommendation(
                    plot_type="Scatter Plots",
                    reason=f"{len(strong)} strong correlation(s) detected — scatter plots reveal the nature of these relationships.",
                    columns=cols,
                    priority="high",
                )
            )

        # Boxplots for outlier detection
        outlier_cols = [s for s in profile.numerical_stats if s.n_outliers > 0]
        if outlier_cols:
            cols = [s.column for s in outlier_cols]
            recs.append(
                VisualizationRecommendation(
                    plot_type="Box Plots",
                    reason=f"{len(cols)} feature(s) contain outliers — box plots visualise their distribution and extremity.",
                    columns=cols,
                    priority="medium",
                )
            )

        # Countplots for categorical data
        if cat_cols:
            recs.append(
                VisualizationRecommendation(
                    plot_type="Count Plots",
                    reason="Show value distribution across categorical features.",
                    columns=cat_cols,
                    priority="medium",
                )
            )

        # Time series if datetime columns exist
        if profile.datetime_columns and num_cols:
            recs.append(
                VisualizationRecommendation(
                    plot_type="Time Series Line Charts",
                    reason="Datetime column detected — temporal trends may exist.",
                    columns=profile.datetime_columns + num_cols[:3],
                    priority="medium",
                )
            )

        # Pairplot for small numeric sets
        if 2 <= len(num_cols) <= 5:
            recs.append(
                VisualizationRecommendation(
                    plot_type="Pair Plot",
                    reason="Small number of numeric features — pair plot shows all pairwise relationships.",
                    columns=num_cols,
                    priority="low",
                )
            )

        return recs

    # ------------------------------------------------------------------
    # Executive summary
    # ------------------------------------------------------------------

    def _executive_summary(
        self,
        profile: DatasetProfile,
        stats: StatisticalAnalysis,
        insights: list[Insight],
    ) -> ExecutiveSummary:
        """Build a business-focused executive summary with narrative."""
        skew_thresh = self.config.skewness_threshold
        np_thresh = self.config.near_perfect_correlation_threshold

        n_skewed = sum(1 for s in profile.numerical_stats if abs(s.skewness) > skew_thresh)
        n_outlier = sum(1 for s in profile.numerical_stats if s.n_outliers > 0)
        n_sig = sum(1 for t in stats.hypothesis_tests if t.significant)

        key_findings: list[str] = []

        # Health score
        health = profile.health_score
        key_findings.append(f"Dataset health score: {health.overall}/100 ({health.label}).")

        # Missing data
        if profile.missing_pct > 0:
            key_findings.append(f"{profile.missing_pct:.1f}% of cells are missing.")

        # Strong correlations with interpretation
        strong = [p for p in stats.pearson.significant_pairs if abs(p.coefficient) >= np_thresh]
        if strong:
            top = max(strong, key=lambda p: abs(p.coefficient))
            key_findings.append(
                f"Strongest relationship: {top.col_a} and {top.col_b} — "
                f"{top.interpretation} (r={top.coefficient:.2f})."
            )

        # Skewed features
        if n_skewed:
            skewed_names = [
                s.column for s in profile.numerical_stats if abs(s.skewness) > skew_thresh
            ][:3]
            key_findings.append(
                f"Skewed distributions: {', '.join(skewed_names)} — "
                f"consider transformations for modelling."
            )

        # Significant tests
        if n_sig:
            key_findings.append(f"{n_sig} statistically significant group difference(s) found.")

        # Outliers
        if n_outlier:
            key_findings.append(f"{n_outlier} feature(s) contain outliers requiring review.")

        # Build narrative
        narrative_parts: list[str] = []
        narrative_parts.append(
            f"The dataset contains {profile.n_rows} rows and "
            f"{profile.n_columns} columns "
            f"({len(profile.numerical_columns)} numerical, "
            f"{len(profile.categorical_columns)} categorical)."
        )
        if health.overall >= 80:
            narrative_parts.append(
                f"Data quality is {health.label.lower()} "
                f"(health score: {health.overall}/100), "
                f"suitable for analysis."
            )
        else:
            narrative_parts.append(
                f"Data quality is {health.label.lower()} "
                f"(health score: {health.overall}/100) — "
                f"review issues before deep analysis."
            )
        if strong:
            narrative_parts.append(f"{len(strong)} strong correlation(s) were identified.")
        if n_skewed:
            narrative_parts.append(f"{n_skewed} feature(s) show non-normal distributions.")
        narrative_parts.append(
            f"The analysis generated {len(insights)} insight(s) and recommendations."
        )
        narrative = " ".join(narrative_parts)

        max_findings = self.config.max_insights_per_category
        return ExecutiveSummary(
            dataset_shape=f"({profile.n_rows}, {profile.n_columns})",
            total_missing_pct=profile.missing_pct,
            n_numerical=len(profile.numerical_columns),
            n_categorical=len(profile.categorical_columns),
            n_datetime=len(profile.datetime_columns),
            n_boolean=len(profile.boolean_columns),
            n_strong_correlations=len(strong),
            n_skewed_features=n_skewed,
            n_outlier_features=n_outlier,
            n_significant_tests=n_sig,
            key_findings=key_findings[:max_findings],
            narrative=narrative,
        )
