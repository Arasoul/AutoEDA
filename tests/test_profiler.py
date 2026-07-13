"""Tests for autoeda.profiler — DataProfiler, classify_variables, compute_health_score."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autoeda.profiler import (
    CategoricalColumnStats,
    DataProfiler,
    DatasetProfile,
    HealthScore,
    NumericalColumnStats,
    VariableInfo,
    classify_variables,
    compute_health_score,
)

# ---------------------------------------------------------------------------
# classify_variables
# ---------------------------------------------------------------------------

class TestClassifyVariables:
    def test_numeric_detection(self, simple_df: pd.DataFrame) -> None:
        variables = classify_variables(simple_df)
        names = [v.column for v in variables]
        assert "age" in names
        age = next(v for v in variables if v.column == "age")
        assert age.semantic_type == "numeric"
        assert age.suggested_role == "feature"

    def test_categorical_detection(self) -> None:
        df = pd.DataFrame({"color": ["red", "blue", "green", "red", "blue", "green", "red", "blue", "green", "red"]})
        variables = classify_variables(df)
        cat = next(v for v in variables if v.column == "color")
        assert cat.semantic_type == "categorical"

    def test_boolean_detection(self, simple_df: pd.DataFrame) -> None:
        variables = classify_variables(simple_df)
        flag = next(v for v in variables if v.column == "is_active")
        assert flag.semantic_type == "boolean"

    def test_constant_detection(self, df_with_constant: pd.DataFrame) -> None:
        variables = classify_variables(df_with_constant)
        const = next(v for v in variables if v.column == "constant_col")
        assert const.semantic_type == "constant"
        assert const.is_constant is True
        assert const.suggested_role == "exclude"

    def test_identifier_detection(self) -> None:
        df = pd.DataFrame({"id": range(100)})
        variables = classify_variables(df)
        ident = next(v for v in variables if v.column == "id")
        assert ident.semantic_type == "identifier"
        assert ident.is_identifier is True
        assert ident.suggested_role == "index"

    def test_text_detection(self, df_with_text: pd.DataFrame) -> None:
        variables = classify_variables(df_with_text)
        txt = next(v for v in variables if v.column == "text_col")
        assert txt.semantic_type == "text"
        assert txt.suggested_role == "exclude_from_stats"

    def test_binary_detection(self) -> None:
        df = pd.DataFrame({"binary": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]})
        variables = classify_variables(df)
        b = next(v for v in variables if v.column == "binary")
        assert b.semantic_type == "binary"

    def test_datetime_detection(self, df_with_datetime: pd.DataFrame) -> None:
        variables = classify_variables(df_with_datetime)
        dt = next(v for v in variables if v.column == "date")
        assert dt.semantic_type == "datetime"
        assert dt.suggested_role == "temporal"

    def test_returns_correct_count(self, simple_df: pd.DataFrame) -> None:
        variables = classify_variables(simple_df)
        assert len(variables) == len(simple_df.columns)

    def test_unique_ratio_calculation(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        variables = classify_variables(df)
        v = variables[0]
        assert v.n_unique == 5
        assert v.unique_ratio == pytest.approx(1.0, abs=0.01)

    def test_is_identifier_requires_numeric(self) -> None:
        df = pd.DataFrame({
            "str_id": [f"item_{i}" for i in range(50)],
        })
        variables = classify_variables(df)
        v = variables[0]
        assert v.is_identifier is False  # string, not numeric


# ---------------------------------------------------------------------------
# compute_health_score
# ---------------------------------------------------------------------------

class TestComputeHealthScore:
    def test_perfect_dataset(self) -> None:
        df = pd.DataFrame({
            "a": range(200),
            "b": np.random.randn(200),
            "c": np.random.choice(["X", "Y"], 200),
        })
        variables = classify_variables(df)
        score = compute_health_score(df, variables)
        assert isinstance(score, HealthScore)
        assert 0 <= score.overall <= 100
        assert score.label in ("Excellent", "Good", "Fair", "Poor")

    def test_missing_data_penalizes(self, df_with_missing: pd.DataFrame) -> None:
        variables = classify_variables(df_with_missing)
        score = compute_health_score(df_with_missing, variables)
        assert score.completeness < 100

    def test_constant_columns_lower_consistency(self, df_with_constant: pd.DataFrame) -> None:
        variables = classify_variables(df_with_constant)
        score = compute_health_score(df_with_constant, variables)
        assert score.consistency < 100
        issues_text = " ".join(score.issues).lower()
        assert "constant" in issues_text

    def test_identifier_columns_flagged(self) -> None:
        df = pd.DataFrame({
            "id": range(50),
            "value": np.random.randn(50),
        })
        variables = classify_variables(df)
        score = compute_health_score(df, variables)
        issues_text = " ".join(score.issues).lower()
        assert "identifier" in issues_text

    def test_no_numeric_features(self) -> None:
        df = pd.DataFrame({"a": ["x", "y", "z"] * 10})
        variables = classify_variables(df)
        score = compute_health_score(df, variables)
        issues_text = " ".join(score.issues).lower()
        assert "no numeric" in issues_text

    def test_small_dataset_penalty(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        variables = classify_variables(df)
        score = compute_health_score(df, variables)
        issues_text = " ".join(score.issues).lower()
        assert "rows" in issues_text

    def test_score_weights(self) -> None:
        df = pd.DataFrame({"a": range(100), "b": range(100)})
        variables = classify_variables(df)
        score = compute_health_score(df, variables)
        expected = round(
            score.completeness * 0.3
            + score.uniqueness * 0.2
            + score.consistency * 0.2
            + score.analysis_readiness * 0.3
        )
        assert score.overall == max(0, min(100, expected))

    def test_label_excellent(self) -> None:
        score = HealthScore(
            overall=95, label="Excellent", completeness=100,
            uniqueness=100, consistency=100, analysis_readiness=80,
            issues=[], strengths=[],
        )
        assert score.label == "Excellent"


# ---------------------------------------------------------------------------
# DataProfiler
# ---------------------------------------------------------------------------

class TestDataProfiler:
    def test_basic_profile(self, simple_df: pd.DataFrame) -> None:
        profiler = DataProfiler()
        profile = profiler.profile(simple_df)
        assert isinstance(profile, DatasetProfile)
        assert profile.n_rows == 10
        assert profile.n_columns == 5
        assert profile.memory_bytes > 0
        assert profile.memory_human != ""
        assert profile.missing_total == 0
        assert profile.duplicate_rows == 0

    def test_numerical_stats(self, simple_df: pd.DataFrame) -> None:
        profile = DataProfiler().profile(simple_df)
        assert len(profile.numerical_stats) == 3
        age_stats = next(s for s in profile.numerical_stats if s.column == "age")
        assert isinstance(age_stats, NumericalColumnStats)
        assert age_stats.mean == pytest.approx(47.5, abs=0.1)
        assert age_stats.count == 10

    def test_categorical_stats(self, simple_df: pd.DataFrame) -> None:
        profile = DataProfiler().profile(simple_df)
        assert len(profile.categorical_stats) == 1
        cat_stats = profile.categorical_stats[0]
        assert isinstance(cat_stats, CategoricalColumnStats)
        assert cat_stats.column == "category"
        assert cat_stats.n_unique == 2

    def test_column_types(self, simple_df: pd.DataFrame) -> None:
        profile = DataProfiler().profile(simple_df)
        assert "age" in profile.numerical_columns
        assert "category" in profile.categorical_columns
        assert "is_active" in profile.boolean_columns

    def test_health_score_present(self, simple_df: pd.DataFrame) -> None:
        profile = DataProfiler().profile(simple_df)
        assert isinstance(profile.health_score, HealthScore)
        assert profile.health_score.overall > 0

    def test_variables_present(self, simple_df: pd.DataFrame) -> None:
        profile = DataProfiler().profile(simple_df)
        assert len(profile.variables) == 5
        assert all(isinstance(v, VariableInfo) for v in profile.variables)

    def test_missing_data_profile(self, df_with_missing: pd.DataFrame) -> None:
        profile = DataProfiler().profile(df_with_missing)
        assert profile.missing_total > 0
        assert profile.missing_pct > 0
        assert len(profile.missing_by_column) > 0

    def test_outlier_detection(self, medium_df: pd.DataFrame) -> None:
        profile = DataProfiler().profile(medium_df)
        rev_stats = next(
            (s for s in profile.numerical_stats if s.column == "revenue"), None
        )
        assert rev_stats is not None
        assert rev_stats.n_outliers > 0

    def test_dtypes(self, simple_df: pd.DataFrame) -> None:
        profile = DataProfiler().profile(simple_df)
        assert isinstance(profile.dtypes, dict)
        assert sum(profile.dtypes.values()) == 5

    def test_custom_config(self, simple_df: pd.DataFrame) -> None:
        from autoeda.config import AutoEDAConfig
        cfg = AutoEDAConfig(figure_dpi=72)
        profiler = DataProfiler(cfg)
        profile = profiler.profile(simple_df)
        assert profile.n_rows == 10

    def test_all_numerical_fields_populated(self, simple_df: pd.DataFrame) -> None:
        profile = DataProfiler().profile(simple_df)
        for s in profile.numerical_stats:
            assert s.column != ""
            assert s.count > 0
            assert not np.isnan(s.mean)
            assert not np.isnan(s.std)
            assert not np.isnan(s.min)
            assert not np.isnan(s.max)
            assert not np.isnan(s.q25)
            assert not np.isnan(s.q75)
            assert not np.isnan(s.iqr)
            assert s.n_outliers >= 0
            assert s.outlier_pct >= 0

    def test_categorical_fields_populated(self, simple_df: pd.DataFrame) -> None:
        profile = DataProfiler().profile(simple_df)
        for s in profile.categorical_stats:
            assert s.column != ""
            assert s.n_unique >= 0
            assert s.top_value != "" or s.n_unique == 0
            assert s.top_freq >= 0
