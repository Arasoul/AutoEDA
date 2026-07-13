"""Shared fixtures for AutoEDA tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def simple_df() -> pd.DataFrame:
    """Small clean DataFrame with numeric, categorical, and boolean columns."""
    np.random.seed(42)
    return pd.DataFrame({
        "age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
        "income": [30000, 45000, 60000, 80000, 95000,
                   120000, 150000, 180000, 200000, 250000],
        "score": [1.0, 2.5, 3.0, 4.5, 5.0, 6.5, 7.0, 8.5, 9.0, 10.0],
        "category": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
        "is_active": [True, False, True, False, True, False, True, False, True, False],
    })


@pytest.fixture()
def medium_df() -> pd.DataFrame:
    """Medium-sized DataFrame with missing values, outliers, and skewness."""
    np.random.seed(123)
    n = 200
    df = pd.DataFrame({
        "revenue": np.random.lognormal(10, 1, n),
        "units": np.random.poisson(50, n),
        "region": np.random.choice(["North", "South", "East", "West"], n),
        "satisfaction": np.random.choice([1, 2, 3, 4, 5], n),
        "is_premium": np.random.choice([True, False], n),
    })
    # Inject missing values
    df.loc[df.sample(frac=0.05).index, "revenue"] = np.nan
    df.loc[df.sample(frac=0.10).index, "units"] = np.nan
    # Inject outliers
    df.loc[0, "revenue"] = 1_000_000
    df.loc[1, "revenue"] = 2_000_000
    return df


@pytest.fixture()
def df_with_missing() -> pd.DataFrame:
    """DataFrame with significant missing data."""
    np.random.seed(99)
    n = 100
    df = pd.DataFrame({
        "a": np.random.randn(n),
        "b": np.random.randn(n),
        "c": np.random.choice(["X", "Y", "Z"], n),
    })
    df.loc[df.sample(frac=0.40).index, "a"] = np.nan
    df.loc[df.sample(frac=0.60).index, "b"] = np.nan
    return df


@pytest.fixture()
def df_no_numeric() -> pd.DataFrame:
    """DataFrame with only categorical columns."""
    return pd.DataFrame({
        "color": ["red", "blue", "green", "red", "blue"] * 20,
        "size": ["S", "M", "L", "XL", "M"] * 20,
    })


@pytest.fixture()
def df_with_datetime() -> pd.DataFrame:
    """DataFrame with a datetime column and numeric columns."""
    np.random.seed(77)
    n = 100
    return pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=n, freq="D"),
        "sales": np.random.poisson(100, n),
        "price": np.random.uniform(10, 100, n),
    })


@pytest.fixture()
def df_with_text() -> pd.DataFrame:
    """DataFrame with a text-like column (high unique ratio, long values)."""
    sentences = [
        f"This is a very long sentence with many words for row number {i} "
        f"and it should be detected as text content by the classifier."
        for i in range(50)
    ]
    return pd.DataFrame({
        "text_col": sentences,
        "label": ["A", "B"] * 25,
    })


@pytest.fixture()
def df_with_constant() -> pd.DataFrame:
    """DataFrame with a constant column."""
    return pd.DataFrame({
        "constant_col": [42] * 50,
        "varying": range(50),
        "category": ["X", "Y"] * 25,
    })


@pytest.fixture()
def df_uniform() -> pd.DataFrame:
    """DataFrame where all numeric columns have identical values (zero variance)."""
    return pd.DataFrame({
        "all_ones": [1.0] * 30,
        "all_zeros": [0.0] * 30,
        "normal": np.random.randn(30),
    })
