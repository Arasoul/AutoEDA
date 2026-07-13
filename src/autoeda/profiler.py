"""Dataset profiling: overview, variable classification, and health scoring.

This module performs the first stage of the EDA pipeline.  It inspects
the raw DataFrame and produces structured, immutable result objects
that downstream modules consume directly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from autoeda.config import AutoEDAConfig
from autoeda.utils import (
    compute_memory_usage,
    compute_missing_summary,
    format_bytes,
    identify_column_types,
    safe_kurtosis,
    safe_max,
    safe_mean,
    safe_median,
    safe_min,
    safe_skew,
    safe_std,
    validate_dataframe,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Variable classification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VariableInfo:
    """Semantic classification of a single column.

    Attributes:
        column: Column name.
        dtype: Raw pandas dtype string.
        semantic_type: One of ``numeric``, ``binary``, ``categorical``,
            ``ordinal``, ``datetime``, ``boolean``, ``identifier``,
            ``text``, ``constant``, ``other``.
        n_unique: Number of unique non-null values.
        unique_ratio: ``n_unique / count``.
        is_constant: True if only one unique value.
        is_identifier: True if unique ratio is ~1.0 and numeric.
        suggested_role: Recommended role in analysis.
    """

    column: str
    dtype: str
    semantic_type: str
    n_unique: int
    unique_ratio: float
    is_constant: bool
    is_identifier: bool
    suggested_role: str


def classify_variables(
    df: pd.DataFrame,
    config: AutoEDAConfig | None = None,
) -> list[VariableInfo]:
    """Classify every column into semantic types.

    Goes beyond dtype-based detection to identify binary, constant,
    identifier, and high-cardinality columns.

    Args:
        df: The DataFrame to classify.
        config: Optional configuration for threshold values.

    Returns:
        A list of :class:`VariableInfo` for every column.
    """
    validate_dataframe(df)
    cfg = config or AutoEDAConfig()
    results: list[VariableInfo] = []
    for col in df.columns:
        series = df[col]
        dtype_str = str(series.dtype)
        non_null = series.dropna()
        n_unique = int(non_null.nunique())
        count = len(non_null)
        unique_ratio = n_unique / count if count > 0 else 0.0
        is_constant = n_unique <= 1

        # Detect identifier: numeric, near-unique, high count
        is_identifier = (
            n_unique > 1
            and unique_ratio > cfg.identifier_unique_ratio
            and count > cfg.identifier_min_count
            and pd.api.types.is_numeric_dtype(series)
        )

        # Detect binary
        is_binary = n_unique == 2

        # Detect text: object/string with high unique ratio and long values
        is_text = False
        if (
            pd.api.types.is_object_dtype(series) or isinstance(series.dtype, pd.StringDtype)
        ) and count > 0:
            avg_len = non_null.astype(str).str.len().mean()
            if (
                unique_ratio > cfg.text_unique_ratio_threshold
                and avg_len > cfg.text_avg_length_threshold
            ):
                is_text = True

        # Detect constant
        if is_constant:
            semantic = "constant"
            role = "exclude"
        elif is_identifier:
            semantic = "identifier"
            role = "index"
        elif is_text:
            semantic = "text"
            role = "exclude_from_stats"
        elif pd.api.types.is_bool_dtype(series):
            semantic = "boolean"
            role = "feature"
        elif is_binary:
            semantic = "binary"
            role = "feature"
        elif pd.api.types.is_numeric_dtype(series):
            semantic = "numeric"
            role = "feature"
        elif pd.api.types.is_datetime64_any_dtype(series):
            semantic = "datetime"
            role = "temporal"
        elif pd.api.types.is_object_dtype(series) or isinstance(
            series.dtype, (pd.CategoricalDtype, pd.StringDtype)
        ):
            semantic = "categorical"
            role = "feature"
        else:
            semantic = "other"
            role = "review"

        results.append(
            VariableInfo(
                column=col,
                dtype=dtype_str,
                semantic_type=semantic,
                n_unique=n_unique,
                unique_ratio=round(unique_ratio, 4),
                is_constant=is_constant,
                is_identifier=is_identifier,
                suggested_role=role,
            )
        )
    return results


# ---------------------------------------------------------------------------
# Dataset Health Score
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HealthScore:
    """Dataset readiness assessment for analysis.

    Attributes:
        overall: Score 0-100.
        label: Qualitative label (``"Excellent"``, ``"Good"``,
            ``"Fair"``, ``"Poor"``).
        completeness: Score for missing data (0-100).
        uniqueness: Score for duplicate rows (0-100).
        consistency: Score for dtype consistency (0-100).
        analysis_readiness: Score for how ready the data is for
            statistical analysis (0-100).
        issues: List of issues found.
        strengths: List of strengths found.
    """

    overall: int
    label: str
    completeness: int
    uniqueness: int
    consistency: int
    analysis_readiness: int
    issues: list[str]
    strengths: list[str]


def compute_health_score(
    df: pd.DataFrame,
    variables: list[VariableInfo],
    config: AutoEDAConfig | None = None,
) -> HealthScore:
    """Compute a dataset health / analysis-readiness score.

    The score measures how ready the dataset is for professional
    analysis, not how clean it is (that is DataPrepToolkit's job).

    Args:
        df: The DataFrame to assess.
        variables: Pre-computed variable classifications.
        config: Optional configuration for threshold values.

    Returns:
        A :class:`HealthScore` with breakdown and narrative.
    """
    issues: list[str] = []
    strengths: list[str] = []

    # Completeness (0-100): based on missing cell ratio
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = int(df.isna().sum().sum())
    completeness = max(0, round(100 - (missing_cells / total_cells * 100))) if total_cells else 100
    if completeness == 100:
        strengths.append("No missing values detected.")
    elif completeness < 80:
        issues.append(f"Dataset has {100 - completeness}% missing values.")
    else:
        strengths.append(f"Missing data is minimal ({100 - completeness}%).")

    # Uniqueness (0-100): based on duplicate row ratio
    n_dupes = int(df.duplicated().sum())
    n_rows = len(df)
    uniqueness = max(0, round(100 - (n_dupes / n_rows * 100))) if n_rows else 100
    if uniqueness == 100:
        strengths.append("No duplicate rows found.")
    elif uniqueness < 90:
        issues.append(f"{n_dupes} duplicate rows detected ({100 - uniqueness}%).")

    # Consistency (0-100): based on identifier/constant/text columns
    n_constant = sum(1 for v in variables if v.semantic_type == "constant")
    n_identifier = sum(1 for v in variables if v.semantic_type == "identifier")
    n_text = sum(1 for v in variables if v.semantic_type == "text")
    bad_cols = n_constant + n_identifier + n_text
    total_cols = len(variables) if variables else 1
    consistency = max(0, round(100 - (bad_cols / total_cols * 100)))
    if n_constant:
        issues.append(f"{n_constant} constant column(s) provide no information.")
    if n_identifier:
        issues.append(f"{n_identifier} identifier column(s) should not be used as features.")
    if consistency == 100:
        strengths.append("All columns are analytically useful.")

    # Analysis readiness (0-100): based on having numeric features,
    # sufficient rows, and reasonable cardinality
    n_numeric = sum(1 for v in variables if v.semantic_type == "numeric")
    readiness_score = 0

    if n_numeric > 0:
        readiness_score += 30
        strengths.append(f"{n_numeric} numeric feature(s) available for analysis.")
    else:
        issues.append("No numeric features detected.")

    if n_rows >= 100:
        readiness_score += 25
    elif n_rows >= 30:
        readiness_score += 15
    else:
        issues.append(f"Only {n_rows} rows — statistical tests may lack power.")

    n_cat = sum(1 for v in variables if v.semantic_type == "categorical")
    if n_cat > 0:
        readiness_score += 15

    n_datetime = sum(1 for v in variables if v.semantic_type == "datetime")
    if n_datetime > 0:
        readiness_score += 10
        strengths.append("Datetime column(s) enable temporal analysis.")

    if completeness > 90:
        readiness_score += 20
    elif completeness > 70:
        readiness_score += 10

    analysis_readiness = min(100, readiness_score)

    overall = round(
        completeness * 0.3 + uniqueness * 0.2 + consistency * 0.2 + analysis_readiness * 0.3
    )
    overall = max(0, min(100, overall))

    if overall >= 90:
        label = "Excellent"
    elif overall >= 75:
        label = "Good"
    elif overall >= 50:
        label = "Fair"
    else:
        label = "Poor"

    return HealthScore(
        overall=overall,
        label=label,
        completeness=completeness,
        uniqueness=uniqueness,
        consistency=consistency,
        analysis_readiness=analysis_readiness,
        issues=issues,
        strengths=strengths,
    )


# ---------------------------------------------------------------------------
# Per-column result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NumericalColumnStats:
    """Descriptive statistics for a single numerical column."""

    column: str
    dtype: str
    count: int
    missing: int
    missing_pct: float
    mean: float
    median: float
    std: float
    min: float
    q25: float
    q50: float
    q75: float
    max: float
    skewness: float
    kurtosis: float
    n_unique: int
    n_zeros: int
    zero_pct: float
    range: float
    iqr: float
    outlier_lower: float
    outlier_upper: float
    n_outliers: int
    outlier_pct: float


@dataclass(frozen=True)
class CategoricalColumnStats:
    """Descriptive statistics for a single categorical column."""

    column: str
    dtype: str
    count: int
    missing: int
    missing_pct: float
    n_unique: int
    unique_ratio: float
    top_value: str
    top_freq: int
    top_pct: float
    values: dict[str, int]


@dataclass(frozen=True)
class DatasetProfile:
    """Complete profiling result for a DataFrame.

    Attributes:
        n_rows: Number of rows.
        n_columns: Number of columns.
        memory_bytes: Total memory usage in bytes.
        memory_human: Human-readable memory string.
        dtypes: Count of columns per dtype.
        column_types: Semantic type classification.
        missing_total: Total missing cells across all columns.
        missing_pct: Percentage of cells that are missing.
        missing_by_column: Per-column missing counts (DataFrame).
        duplicate_rows: Number of fully duplicated rows.
        duplicate_pct: Percentage of duplicated rows.
        numerical_stats: Per-column statistics for numeric features.
        categorical_stats: Per-column statistics for categorical features.
        numerical_columns: List of numeric column names.
        categorical_columns: List of categorical column names.
        datetime_columns: List of datetime column names.
        boolean_columns: List of boolean column names.
        variables: Variable classification for every column.
        health_score: Dataset readiness assessment.
    """

    n_rows: int
    n_columns: int
    memory_bytes: int
    memory_human: str
    dtypes: dict[str, int]
    column_types: dict[str, list[str]]
    missing_total: int
    missing_pct: float
    missing_by_column: pd.DataFrame
    duplicate_rows: int
    duplicate_pct: float
    numerical_stats: list[NumericalColumnStats]
    categorical_stats: list[CategoricalColumnStats]
    numerical_columns: list[str]
    categorical_columns: list[str]
    datetime_columns: list[str]
    boolean_columns: list[str]
    variables: list[VariableInfo]
    health_score: HealthScore


# ---------------------------------------------------------------------------
# Profiler
# ---------------------------------------------------------------------------


class DataProfiler:
    """Produce a :class:`DatasetProfile` from a cleaned DataFrame.

    Example::

        profiler = DataProfiler(config)
        profile = profiler.profile(df)
    """

    def __init__(self, config: AutoEDAConfig | None = None) -> None:
        self.config = config or AutoEDAConfig()

    def profile(self, df: pd.DataFrame) -> DatasetProfile:
        """Run the full profiling pass on *df*.

        Args:
            df: A cleaned DataFrame.

        Returns:
            A :class:`DatasetProfile` with every metric pre-computed.
        """
        validate_dataframe(df)
        logger.info("Profiling DataFrame (%d x %d)", len(df), len(df.columns))

        col_types = identify_column_types(df)
        mem_bytes = compute_memory_usage(df)
        missing_df = compute_missing_summary(df)
        missing_total = int(df.isna().sum().sum())
        total_cells = df.shape[0] * df.shape[1]
        missing_pct_val = (missing_total / total_cells * 100) if total_cells else 0.0

        variables = classify_variables(df, self.config)
        health = compute_health_score(df, variables, self.config)

        num_stats = [self._profile_numerical(df, col) for col in col_types["numeric"]]
        cat_stats = [self._profile_categorical(df, col) for col in col_types["categorical"]]

        dup_count = int(df.duplicated().sum())
        n_rows = len(df)

        profile = DatasetProfile(
            n_rows=n_rows,
            n_columns=len(df.columns),
            memory_bytes=mem_bytes,
            memory_human=format_bytes(mem_bytes),
            dtypes=dict(df.dtypes.value_counts().astype(int)),
            column_types=col_types,
            missing_total=missing_total,
            missing_pct=round(missing_pct_val, 2),
            missing_by_column=missing_df,
            duplicate_rows=dup_count,
            duplicate_pct=round(dup_count / n_rows * 100, 2) if n_rows else 0.0,
            numerical_stats=num_stats,
            categorical_stats=cat_stats,
            numerical_columns=col_types["numeric"],
            categorical_columns=col_types["categorical"],
            datetime_columns=col_types["datetime"],
            boolean_columns=col_types["boolean"],
            variables=variables,
            health_score=health,
        )

        logger.info(
            "Profiling complete: %d numerical, %d categorical, health=%d/100 (%s)",
            len(num_stats),
            len(cat_stats),
            health.overall,
            health.label,
        )
        return profile

    def _profile_numerical(
        self,
        df: pd.DataFrame,
        col: str,
    ) -> NumericalColumnStats:
        """Compute descriptive statistics for one numeric column."""
        s = df[col].dropna()
        count = int(s.count())
        missing = int(df[col].isna().sum())
        total = len(df)
        missing_pct = round(missing / total * 100, 2) if total else 0.0

        q25 = float(s.quantile(0.25)) if count else float("nan")
        q75 = float(s.quantile(0.75)) if count else float("nan")
        iqr = q75 - q25 if count else float("nan")
        multiplier = self.config.outlier_iqr_multiplier
        outlier_lower = q25 - multiplier * iqr if count else float("nan")
        outlier_upper = q75 + multiplier * iqr if count else float("nan")
        n_outliers = int(((s < outlier_lower) | (s > outlier_upper)).sum()) if count else 0
        n_zeros = int((s == 0).sum())

        return NumericalColumnStats(
            column=col,
            dtype=str(df[col].dtype),
            count=count,
            missing=missing,
            missing_pct=missing_pct,
            mean=safe_mean(s),
            median=safe_median(s),
            std=safe_std(s),
            min=safe_min(s),
            q25=q25,
            q50=float(s.quantile(0.50)) if count else float("nan"),
            q75=q75,
            max=safe_max(s),
            skewness=safe_skew(s),
            kurtosis=safe_kurtosis(s),
            n_unique=int(s.nunique()),
            n_zeros=n_zeros,
            zero_pct=round(n_zeros / count * 100, 2) if count else 0.0,
            range=(safe_max(s) - safe_min(s)) if count else float("nan"),
            iqr=iqr,
            outlier_lower=outlier_lower,
            outlier_upper=outlier_upper,
            n_outliers=n_outliers,
            outlier_pct=round(n_outliers / count * 100, 2) if count else 0.0,
        )

    def _profile_categorical(
        self,
        df: pd.DataFrame,
        col: str,
    ) -> CategoricalColumnStats:
        """Compute descriptive statistics for one categorical column."""
        s = df[col]
        total = len(df)
        missing = int(s.isna().sum())
        missing_pct = round(missing / total * 100, 2) if total else 0.0
        non_null = s.dropna()
        n_unique = int(non_null.nunique())

        if n_unique > 0:
            value_counts = non_null.value_counts()
            top_value = str(value_counts.index[0])
            top_freq = int(value_counts.iloc[0])
            top_pct = round(top_freq / len(non_null) * 100, 2)
            values = {str(k): int(v) for k, v in value_counts.items()}
        else:
            top_value = ""
            top_freq = 0
            top_pct = 0.0
            values = {}

        return CategoricalColumnStats(
            column=col,
            dtype=str(df[col].dtype),
            count=int(non_null.count()),
            missing=missing,
            missing_pct=missing_pct,
            n_unique=n_unique,
            unique_ratio=round(n_unique / len(non_null), 4) if len(non_null) else 0.0,
            top_value=top_value,
            top_freq=top_freq,
            top_pct=top_pct,
            values=values,
        )
