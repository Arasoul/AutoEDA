"""Tests for autoeda.utils."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autoeda.utils import (
    compute_memory_usage,
    compute_missing_summary,
    flatten_list,
    format_bytes,
    format_number,
    format_percentage,
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


class TestValidateDataframe:
    def test_valid_dataframe(self, simple_df: pd.DataFrame) -> None:
        validate_dataframe(simple_df)

    def test_not_dataframe(self) -> None:
        with pytest.raises(TypeError, match="Expected a pandas DataFrame"):
            validate_dataframe([1, 2, 3])  # type: ignore[arg-type]

    def test_empty_dataframe(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            validate_dataframe(pd.DataFrame())


class TestIdentifyColumnTypes:
    def test_mixed_types(self, simple_df: pd.DataFrame) -> None:
        result = identify_column_types(simple_df)
        assert "age" in result["numeric"]
        assert "income" in result["numeric"]
        assert "score" in result["numeric"]
        assert "category" in result["categorical"]
        assert "is_active" in result["boolean"]

    def test_all_numeric(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        result = identify_column_types(df)
        assert result["numeric"] == ["a", "b"]
        assert result["categorical"] == []

    def test_datetime_detection(self) -> None:
        df = pd.DataFrame({"d": pd.date_range("2023-01-01", periods=3)})
        result = identify_column_types(df)
        assert "d" in result["datetime"]

    def test_categorical_detection(self) -> None:
        df = pd.DataFrame({"c": pd.Categorical(["a", "b", "c"])})
        result = identify_column_types(df)
        assert "c" in result["categorical"]


class TestSafeStatistics:
    def test_safe_mean(self) -> None:
        s = pd.Series([1.0, 2.0, 3.0])
        assert safe_mean(s) == 2.0

    def test_safe_mean_with_nan(self) -> None:
        s = pd.Series([1.0, np.nan, 3.0])
        assert safe_mean(s) == 2.0

    def test_safe_mean_error(self) -> None:
        s = pd.Series(["a", "b", "c"])
        assert np.isnan(safe_mean(s))

    def test_safe_median(self) -> None:
        s = pd.Series([1.0, 2.0, 3.0])
        assert safe_median(s) == 2.0

    def test_safe_std(self) -> None:
        s = pd.Series([1.0, 2.0, 3.0])
        assert safe_std(s) == pytest.approx(1.0, abs=0.01)

    def test_safe_skew(self) -> None:
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 100.0])
        assert safe_skew(s) > 0

    def test_safe_kurtosis(self) -> None:
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 100.0])
        result = safe_kurtosis(s)
        assert isinstance(result, float)

    def test_safe_min(self) -> None:
        s = pd.Series([5.0, 1.0, 3.0])
        assert safe_min(s) == 1.0

    def test_safe_max(self) -> None:
        s = pd.Series([5.0, 1.0, 3.0])
        assert safe_max(s) == 5.0

    def test_safe_min_error(self) -> None:
        s = pd.Series(["a", "b"])
        assert np.isnan(safe_min(s))


class TestFormatting:
    def test_format_number_normal(self) -> None:
        assert format_number(123.45) == "123.45"

    def test_format_number_thousands(self) -> None:
        assert format_number(5000) == "5.00K"

    def test_format_number_millions(self) -> None:
        assert format_number(1_500_000) == "1.50M"

    def test_format_number_nan(self) -> None:
        assert format_number(float("nan")) == "N/A"

    def test_format_number_inf(self) -> None:
        assert format_number(float("inf")) == "N/A"

    def test_format_percentage(self) -> None:
        assert format_percentage(0.423) == "42.3%"

    def test_format_percentage_nan(self) -> None:
        assert format_percentage(float("nan")) == "N/A"

    def test_format_bytes_bytes(self) -> None:
        assert format_bytes(500) == "500.0 B"

    def test_format_bytes_kb(self) -> None:
        assert format_bytes(2048) == "2.0 KB"

    def test_format_bytes_mb(self) -> None:
        assert format_bytes(1_048_576) == "1.0 MB"

    def test_format_bytes_gb(self) -> None:
        assert format_bytes(1_073_741_824) == "1.0 GB"


class TestComputeMissingSummary:
    def test_no_missing(self) -> None:
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = compute_missing_summary(df)
        assert len(result) == 0

    def test_with_missing(self) -> None:
        df = pd.DataFrame({"a": [1, np.nan, 3], "b": [np.nan, np.nan, np.nan]})
        result = compute_missing_summary(df)
        assert len(result) == 2
        assert result.iloc[0]["column"] == "b"
        assert result.iloc[0]["missing_count"] == 3

    def test_single_column_missing(self) -> None:
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [4, 5, 6]})
        result = compute_missing_summary(df)
        assert len(result) == 1
        assert result.iloc[0]["column"] == "a"


class TestComputeMemoryUsage:
    def test_basic(self, simple_df: pd.DataFrame) -> None:
        mem = compute_memory_usage(simple_df)
        assert mem > 0
        assert isinstance(mem, int)


class TestFlattenList:
    def test_flatten(self) -> None:
        assert flatten_list([["a", "b"], ["c"]]) == ["a", "b", "c"]

    def test_empty(self) -> None:
        assert flatten_list([]) == []

    def test_single(self) -> None:
        assert flatten_list([["x"]]) == ["x"]
