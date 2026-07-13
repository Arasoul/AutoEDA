"""Statistical analytics: correlation, covariance, hypothesis tests, normality.

This module performs the second stage of the EDA pipeline.  It consumes
the :class:`~autoeda.profiler.DatasetProfile` produced by Phase 2 and
produces structured result objects consumed by downstream modules.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from autoeda.config import AutoEDAConfig
from autoeda.utils import validate_dataframe

if TYPE_CHECKING:
    from autoeda.profiler import DatasetProfile

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Correlation interpretation
# ---------------------------------------------------------------------------

# Thresholds are ordered highest to lowest for label lookup.
_STRENGTH_THRESHOLDS: list[tuple[float, str]] = [
    (0.9, "Very Strong"),
    (0.7, "Strong"),
    (0.5, "Moderate"),
    (0.3, "Weak"),
]
_STRENGTH_LABEL_DEFAULT = "Negligible"


def _interpret_strength_label(coefficient: float) -> str:
    """Return a short label for correlation strength."""
    abs_r = abs(coefficient)
    for threshold, label in _STRENGTH_THRESHOLDS:
        if abs_r >= threshold:
            return label
    return _STRENGTH_LABEL_DEFAULT


def _interpret_correlation(coefficient: float) -> str:
    """Return a human-readable interpretation of a correlation coefficient.

    Args:
        coefficient: Pearson/Spearman/Kendall r value (-1 to 1).

    Returns:
        A descriptive string like
        ``"strong positive linear relationship"``.
    """
    abs_r = abs(coefficient)
    if abs_r >= 0.9:
        strength = "very strong"
    elif abs_r >= 0.7:
        strength = "strong"
    elif abs_r >= 0.5:
        strength = "moderate"
    elif abs_r >= 0.3:
        strength = "weak"
    else:
        strength = "very weak / negligible"

    if coefficient > 0:
        direction = "positive"
    elif coefficient < 0:
        direction = "negative"
    else:
        direction = "no"

    return f"{strength} {direction} linear relationship"


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorrelationPair:
    """A single statistically significant correlation between two columns."""

    col_a: str
    col_b: str
    method: str
    coefficient: float
    p_value: float
    significant: bool
    interpretation: str
    strength_label: str


@dataclass(frozen=True)
class CorrelationResult:
    """Full correlation analysis across all numeric columns."""

    method: str
    matrix: dict[str, dict[str, float]]
    significant_pairs: list[CorrelationPair]
    n_significant: int


@dataclass(frozen=True)
class CovarianceResult:
    """Covariance matrix and basic diagnostics."""

    matrix: dict[str, dict[str, float]]
    variances: dict[str, float]


@dataclass(frozen=True)
class NormalityTest:
    """Normality test result for a single numeric column."""

    column: str
    shapiro_stat: float
    shapiro_p: float
    dagostino_stat: float
    dagostino_p: float
    ks_stat: float
    ks_p: float
    is_normal: bool


@dataclass(frozen=True)
class NormalityResult:
    """Aggregated normality test results for all numeric columns."""

    tests: list[NormalityTest]
    n_normal: int
    n_not_normal: int


@dataclass(frozen=True)
class HypothesisTestResult:
    """Result of a single hypothesis test."""

    test_name: str
    statistic: float
    p_value: float
    significant: bool
    column: str
    categorical_column: str
    group_a: str | None = None
    group_b: str | None = None
    interpretation: str = ""


@dataclass(frozen=True)
class ConfidenceIntervalResult:
    """Confidence interval for the mean of a single numeric column."""

    column: str
    confidence_level: float
    mean: float
    lower: float
    upper: float
    margin_of_error: float


@dataclass(frozen=True)
class StatisticalAnalysis:
    """Complete statistical analysis result."""

    pearson: CorrelationResult
    spearman: CorrelationResult
    kendall: CorrelationResult
    covariance: CovarianceResult
    normality: NormalityResult
    hypothesis_tests: list[HypothesisTestResult]
    confidence_intervals: list[ConfidenceIntervalResult]


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------


class Analytics:
    """Run statistical analyses on a profiled DataFrame.

    Example::

        analytics = Analytics(config)
        stats = analytics.analyse(df, profile)
    """

    def __init__(self, config: AutoEDAConfig | None = None) -> None:
        self.config = config or AutoEDAConfig()

    def analyse(
        self,
        df: pd.DataFrame,
        profile: DatasetProfile,
    ) -> StatisticalAnalysis:
        """Run every statistical analysis available for *df*.

        Args:
            df: The cleaned DataFrame.
            profile: The profiling result from Phase 2.

        Returns:
            A :class:`StatisticalAnalysis` containing all results.
        """
        validate_dataframe(df)
        logger.info("Starting statistical analysis")

        num_cols = profile.numerical_columns
        cat_cols = profile.categorical_columns

        pearson = self._correlation(df, num_cols, "pearson")
        spearman = self._correlation(df, num_cols, "spearman")
        kendall = self._correlation(df, num_cols, "kendall")
        cov = self._covariance(df, num_cols)
        norm = self._normality(df, num_cols)
        hyp = self._hypothesis_tests(df, num_cols, cat_cols)
        ci = self._confidence_intervals(df, num_cols)

        result = StatisticalAnalysis(
            pearson=pearson,
            spearman=spearman,
            kendall=kendall,
            covariance=cov,
            normality=norm,
            hypothesis_tests=hyp,
            confidence_intervals=ci,
        )
        logger.info("Statistical analysis complete")
        return result

    def _correlation(
        self,
        df: pd.DataFrame,
        num_cols: list[str],
        method: str,
    ) -> CorrelationResult:
        """Compute correlation matrix and extract significant pairs."""
        if len(num_cols) < 2:
            return CorrelationResult(
                method=method,
                matrix={},
                significant_pairs=[],
                n_significant=0,
            )

        sub = df[num_cols].dropna()
        if len(sub) < 3:
            return CorrelationResult(
                method=method,
                matrix={},
                significant_pairs=[],
                n_significant=0,
            )

        corr_matrix = sub.corr(method=method)  # type: ignore[arg-type]

        matrix: dict[str, dict[str, float]] = {}
        pairs: list[CorrelationPair] = []
        alpha = self.config.significance_level

        for i, ca in enumerate(num_cols):
            matrix[ca] = {}
            for j, cb in enumerate(num_cols):
                val = float(corr_matrix.loc[ca, cb])  # type: ignore[arg-type]
                matrix[ca][cb] = round(val, 4)
                if j > i:
                    if method == "pearson":
                        _, p = sp_stats.pearsonr(sub[ca], sub[cb])
                    elif method == "spearman":
                        _, p = sp_stats.spearmanr(sub[ca], sub[cb])
                    else:
                        _, p = sp_stats.kendalltau(sub[ca], sub[cb])
                    sig = p < alpha
                    interp = _interpret_correlation(val)
                    strength = _interpret_strength_label(val)
                    pairs.append(
                        CorrelationPair(
                            col_a=ca,
                            col_b=cb,
                            method=method,
                            coefficient=round(val, 4),
                            p_value=round(float(p), 6),
                            significant=sig,
                            interpretation=interp,
                            strength_label=strength,
                        )
                    )

        sig_count = sum(1 for p in pairs if p.significant)
        logger.info(
            "%s: %d significant pairs out of %d",
            method.capitalize(),
            sig_count,
            len(pairs),
        )
        return CorrelationResult(
            method=method,
            matrix=matrix,
            significant_pairs=pairs,
            n_significant=sig_count,
        )

    def _covariance(
        self,
        df: pd.DataFrame,
        num_cols: list[str],
    ) -> CovarianceResult:
        """Compute covariance matrix."""
        if len(num_cols) < 2:
            return CovarianceResult(matrix={}, variances={})

        sub = df[num_cols].dropna()
        cov_matrix = sub.cov()

        matrix: dict[str, dict[str, float]] = {}
        variances: dict[str, float] = {}
        for ca in num_cols:
            matrix[ca] = {}
            for cb in num_cols:
                matrix[ca][cb] = round(float(cov_matrix.loc[ca, cb]), 4)  # type: ignore[arg-type]
            variances[ca] = round(float(cov_matrix.loc[ca, ca]), 4)  # type: ignore[arg-type]

        return CovarianceResult(matrix=matrix, variances=variances)

    def _normality(
        self,
        df: pd.DataFrame,
        num_cols: list[str],
    ) -> NormalityResult:
        """Run normality tests on every numeric column."""
        tests: list[NormalityTest] = []
        alpha = self.config.significance_level
        min_shapiro = self.config.normality_min_sample_shapiro
        min_dagostino = self.config.normality_min_sample_dagostino

        for col in num_cols:
            s = df[col].dropna()
            n = len(s)
            if n < min_shapiro:
                continue

            try:
                s_stat, s_p = sp_stats.shapiro(s.head(5000))
            except Exception as exc:
                logger.warning("Shapiro-Wilk failed for column '%s': %s", col, exc)
                s_stat, s_p = float("nan"), float("nan")

            if n >= min_dagostino:
                try:
                    d_stat, d_p = sp_stats.normaltest(s)
                except Exception as exc:
                    logger.warning("D'Agostino failed for column '%s': %s", col, exc)
                    d_stat, d_p = float("nan"), float("nan")
            else:
                d_stat, d_p = float("nan"), float("nan")

            mu, sigma = float(s.mean()), float(s.std())
            if sigma > 0:
                try:
                    k_stat, k_p = sp_stats.kstest(s, "norm", args=(mu, sigma))
                except Exception as exc:
                    logger.warning("KS test failed for column '%s': %s", col, exc)
                    k_stat, k_p = float("nan"), float("nan")
            else:
                k_stat, k_p = float("nan"), float("nan")

            is_normal = all(p >= alpha for p in (s_p, d_p, k_p) if not np.isnan(p))

            tests.append(
                NormalityTest(
                    column=col,
                    shapiro_stat=round(float(s_stat), 4),
                    shapiro_p=round(float(s_p), 6),
                    dagostino_stat=round(float(d_stat), 4),
                    dagostino_p=round(float(d_p), 6),
                    ks_stat=round(float(k_stat), 4),
                    ks_p=round(float(k_p), 6),
                    is_normal=is_normal,
                )
            )

        n_normal = sum(1 for t in tests if t.is_normal)
        return NormalityResult(
            tests=tests,
            n_normal=n_normal,
            n_not_normal=len(tests) - n_normal,
        )

    def _hypothesis_tests(
        self,
        df: pd.DataFrame,
        num_cols: list[str],
        cat_cols: list[str],
    ) -> list[HypothesisTestResult]:
        """Run appropriate hypothesis tests for each num x cat pair."""
        results: list[HypothesisTestResult] = []
        alpha = self.config.significance_level

        for num_col in num_cols:
            for cat_col in cat_cols:
                groups = [
                    group[num_col].dropna().values
                    for _, group in df.groupby(cat_col)
                    if len(group[num_col].dropna()) >= 2
                ]
                if len(groups) < 2:
                    continue

                group_names = [
                    str(name)
                    for name, group in df.groupby(cat_col)
                    if len(group[num_col].dropna()) >= 2
                ]

                if len(groups) == 2:
                    stat, p = sp_stats.ttest_ind(groups[0], groups[1])
                    interp = (
                        f"Significant difference in '{num_col}' between "
                        f"'{group_names[0]}' and '{group_names[1]}' "
                        f"(p={float(p):.6f})."
                        if p < alpha
                        else f"No significant difference in '{num_col}' "
                        f"between '{group_names[0]}' and '{group_names[1]}'."
                    )
                    results.append(
                        HypothesisTestResult(
                            test_name="independent_t_test",
                            statistic=round(float(stat), 4),
                            p_value=round(float(p), 6),
                            significant=p < alpha,
                            column=num_col,
                            categorical_column=cat_col,
                            group_a=group_names[0],
                            group_b=group_names[1],
                            interpretation=interp,
                        )
                    )
                elif len(groups) > 2:
                    stat, p = sp_stats.f_oneway(*groups)
                    interp = (
                        f"Significant difference in '{num_col}' across "
                        f"groups of '{cat_col}' (p={float(p):.6f})."
                        if p < alpha
                        else f"No significant difference in '{num_col}' "
                        f"across groups of '{cat_col}'."
                    )
                    results.append(
                        HypothesisTestResult(
                            test_name="one_way_anova",
                            statistic=round(float(stat), 4),
                            p_value=round(float(p), 6),
                            significant=p < alpha,
                            column=num_col,
                            categorical_column=cat_col,
                            interpretation=interp,
                        )
                    )

        for i, cat_a in enumerate(cat_cols):
            for cat_b in cat_cols[i + 1 :]:
                contingency = pd.crosstab(df[cat_a], df[cat_b])
                if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                    continue
                chi2, p, dof, _ = sp_stats.chi2_contingency(contingency)
                interp = (
                    f"Significant association between '{cat_a}' and '{cat_b}' "
                    f"(chi2={float(chi2):.4f}, p={float(p):.6f})."
                    if p < alpha
                    else f"No significant association between '{cat_a}' and '{cat_b}'."
                )
                results.append(
                    HypothesisTestResult(
                        test_name="chi_square",
                        statistic=round(float(chi2), 4),
                        p_value=round(float(p), 6),
                        significant=p < alpha,
                        column=cat_a,
                        categorical_column=cat_b,
                        interpretation=interp,
                    )
                )

        logger.info("Hypothesis tests: %d performed", len(results))
        return results

    def _confidence_intervals(
        self,
        df: pd.DataFrame,
        num_cols: list[str],
    ) -> list[ConfidenceIntervalResult]:
        """Compute confidence intervals for the mean of each numeric column."""
        results: list[ConfidenceIntervalResult] = []
        cl = self.config.confidence_level

        for col in num_cols:
            s = df[col].dropna()
            n = len(s)
            if n < 2:
                continue
            mean = float(s.mean())
            se = float(s.std(ddof=1)) / np.sqrt(n)
            z = sp_stats.t.ppf((1 + cl) / 2, df=n - 1)
            margin = z * se
            results.append(
                ConfidenceIntervalResult(
                    column=col,
                    confidence_level=cl,
                    mean=round(mean, 4),
                    lower=round(mean - margin, 4),
                    upper=round(mean + margin, 4),
                    margin_of_error=round(margin, 4),
                )
            )

        logger.info("Confidence intervals: %d computed", len(results))
        return results
