"""Internal utility functions shared across AutoEDA modules."""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DataFrame helpers
# ---------------------------------------------------------------------------


def validate_dataframe(df: pd.DataFrame) -> None:
    """Raise :class:`TypeError` or :class:`ValueError` for invalid input.

    Args:
        df: The DataFrame to validate.

    Raises:
        TypeError: If *df* is not a :class:`pandas.DataFrame`.
        ValueError: If *df* is empty.
    """
    if not isinstance(df, pd.DataFrame):
        msg = f"Expected a pandas DataFrame, got {type(df).__name__}"
        raise TypeError(msg)
    if df.empty:
        msg = "DataFrame is empty; nothing to analyse"
        raise ValueError(msg)


def identify_column_types(
    df: pd.DataFrame,
) -> dict[str, list[str]]:
    """Classify every column into semantic type groups.

    Types: ``numeric``, ``categorical``, ``datetime``, ``boolean``,
    ``other``.

    Args:
        df: The DataFrame to classify.

    Returns:
        Mapping from type name to list of column names.
    """
    result: dict[str, list[str]] = {
        "numeric": [],
        "categorical": [],
        "datetime": [],
        "boolean": [],
        "other": [],
    }
    for col in df.columns:
        dtype = df[col].dtype
        if pd.api.types.is_bool_dtype(dtype):
            result["boolean"].append(col)
        elif pd.api.types.is_numeric_dtype(dtype):
            result["numeric"].append(col)
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            result["datetime"].append(col)
        elif pd.api.types.is_object_dtype(dtype) or isinstance(
            dtype, (pd.CategoricalDtype, pd.StringDtype)
        ):
            result["categorical"].append(col)
        else:
            result["other"].append(col)
    return result


# ---------------------------------------------------------------------------
# Safe statistical computations
# ---------------------------------------------------------------------------


def safe_mean(series: pd.Series) -> float:  # type: ignore[type-arg]
    """Compute mean, returning NaN on failure."""
    try:
        result = series.mean()
        return float(result) if result is not None else float("nan")  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def safe_median(series: pd.Series) -> float:  # type: ignore[type-arg]
    """Compute median, returning NaN on failure."""
    try:
        result = series.median()
        return float(result) if result is not None else float("nan")  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def safe_std(series: pd.Series) -> float:  # type: ignore[type-arg]
    """Compute standard deviation, returning NaN on failure."""
    try:
        return float(series.std())
    except (TypeError, ValueError):
        return float("nan")


def safe_skew(series: pd.Series) -> float:  # type: ignore[type-arg]
    """Compute skewness, returning NaN on failure."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return float(series.skew())  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def safe_kurtosis(series: pd.Series) -> float:  # type: ignore[type-arg]
    """Compute kurtosis, returning NaN on failure."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return float(series.kurtosis())  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def safe_min(series: pd.Series) -> float:  # type: ignore[type-arg]
    """Compute minimum, returning NaN on failure."""
    try:
        result = series.min()
        return float(result) if result is not None else float("nan")  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def safe_max(series: pd.Series) -> float:  # type: ignore[type-arg]
    """Compute maximum, returning NaN on failure."""
    try:
        result = series.max()
        return float(result) if result is not None else float("nan")  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def format_number(value: float, decimals: int = 2) -> str:
    """Format a number for display in tables and reports.

    Args:
        value: The number to format.
        decimals: Decimal places to keep.

    Returns:
        A human-readable string representation.
    """
    if np.isnan(value) or np.isinf(value):
        return "N/A"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:,.{decimals}f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:,.{decimals}f}K"
    return f"{value:,.{decimals}f}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format a ratio (0-1) as a percentage string.

    Args:
        value: Ratio between 0 and 1.
        decimals: Decimal places for the percentage.

    Returns:
        A string like ``"42.3%"``.
    """
    if np.isnan(value):
        return "N/A"
    return f"{value * 100:.{decimals}f}%"


def format_bytes(size_bytes: int) -> str:
    """Format byte count into human-readable string.

    Args:
        size_bytes: Size in bytes.

    Returns:
        A string like ``"1.5 MB"``.
    """
    if size_bytes < 0:
        return "N/A"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size_bytes) < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes = int(size_bytes / 1024)
    return f"{size_bytes:.1f} PB"


def compute_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame summarising missing values per column.

    Args:
        df: The DataFrame to inspect.

    Returns:
        A DataFrame with columns ``column``, ``missing_count``,
        ``missing_pct``.
    """
    total = len(df)
    if total == 0:
        return pd.DataFrame(columns=["column", "missing_count", "missing_pct"])
    missing = df.isna().sum()
    missing_count = missing.to_numpy().astype(int)
    missing_pct = np.round(
        np.where(total > 0, missing_count / total * 100, 0.0),
        2,
    )
    result = pd.DataFrame(
        {
            "column": missing.index,
            "missing_count": missing_count,
            "missing_pct": missing_pct,
        }
    )
    return (
        result[result["missing_count"] > 0]
        .sort_values(
            "missing_count",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def compute_memory_usage(df: pd.DataFrame) -> int:
    """Return total memory usage of a DataFrame in bytes.

    Args:
        df: The DataFrame to measure.

    Returns:
        Memory usage in bytes.
    """
    return int(df.memory_usage(deep=True).sum())


def flatten_list(nested: list[list[str]]) -> list[str]:
    """Flatten a list of lists into a single list.

    Args:
        nested: A list containing lists of strings.

    Returns:
        A flat list of strings.
    """
    return [item for sublist in nested for item in sublist]
