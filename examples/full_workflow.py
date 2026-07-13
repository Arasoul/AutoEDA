"""Full AutoEDA workflow — single-script demo of every pipeline stage.

Run with:
    python examples/full_workflow.py
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)


def generate_sample_data(n: int = 500) -> pd.DataFrame:
    """Create a realistic sample dataset with mixed column types."""
    np.random.seed(42)
    return pd.DataFrame({
        "customer_id": range(1, n + 1),
        "age": np.random.normal(35, 12, n).astype(int).clip(18, 80),
        "annual_income": np.random.lognormal(10.8, 0.7, n).round(2),
        "spending_score": np.random.uniform(1, 100, n).round(1),
        "years_as_customer": np.random.poisson(4, n),
        "region": np.random.choice(["North", "South", "East", "West"], n, p=[0.3, 0.25, 0.25, 0.2]),
        "membership_tier": np.random.choice(["Bronze", "Silver", "Gold", "Platinum"], n, p=[0.4, 0.3, 0.2, 0.1]),
        "is_active": np.random.choice([True, False], n, p=[0.7, 0.3]),
        "signup_date": pd.date_range("2020-01-01", periods=n, freq="3D"),
        "last_purchase_days": np.random.exponential(30, n).astype(int),
        "total_orders": np.random.poisson(8, n),
        "avg_order_value": np.random.lognormal(4, 0.6, n).round(2),
    }).pipe(_inject_issues)


def _inject_issues(df: pd.DataFrame) -> pd.DataFrame:
    """Inject realistic data quality issues for demo purposes."""
    df = df.copy()
    # Missing values
    df.loc[df.sample(frac=0.03).index, "annual_income"] = np.nan
    df.loc[df.sample(frac=0.05).index, "spending_score"] = np.nan
    df.loc[df.sample(frac=0.02).index, "region"] = np.nan
    # Outliers
    df.loc[0, "annual_income"] = 500_000
    df.loc[1, "annual_income"] = 600_000
    df.loc[2, "avg_order_value"] = 50_000
    return df


def main() -> None:
    from autoeda import AutoEDA
    from autoeda.config import AutoEDAConfig

    logger.info("Generating sample dataset...")
    df = generate_sample_data(500)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}\n")

    # Configure
    config = AutoEDAConfig(
        output_dir="reports/autoeda_demo",
        report_formats=["html", "markdown"],
        save_figures=True,
        save_reports=True,
        correlation_threshold=0.5,
    )

    # Run full pipeline
    logger.info("Running AutoEDA pipeline...")
    results = AutoEDA(config).run(df)

    # Print summary
    profile = results["profile"]
    stats = results["statistics"]
    insights = results["insights"]
    reports = results["report_paths"]

    print("=" * 60)
    print("PIPELINE RESULTS")
    print("=" * 60)

    print(f"\n--- Dataset Profile ---")
    print(f"  Shape: {profile.n_rows} rows x {profile.n_columns} columns")
    print(f"  Memory: {profile.memory_human}")
    print(f"  Missing: {profile.missing_pct}%")
    print(f"  Duplicates: {profile.duplicate_rows}")
    print(f"  Health Score: {profile.health_score.overall}/100 ({profile.health_score.label})")

    print(f"\n--- Variable Classification ---")
    for v in profile.variables:
        print(f"  {v.column:25s} -> {v.semantic_type:15s} (role={v.suggested_role})")

    print(f"\n--- Statistical Analysis ---")
    print(f"  Pearson significant pairs: {stats.pearson.n_significant}")
    print(f"  Normal features: {stats.normality.n_normal}/{len(stats.normality.tests)}")
    print(f"  Hypothesis tests: {len(stats.hypothesis_tests)}")
    print(f"  Confidence intervals: {len(stats.confidence_intervals)}")

    print(f"\n--- Correlations ---")
    for p in stats.pearson.significant_pairs[:5]:
        print(f"  {p.col_a:25s} <-> {p.col_b:25s}  r={p.coefficient:+.3f}  ({p.interpretation})")

    print(f"\n--- Insights ---")
    for ins in insights.insights[:5]:
        print(f"  [{ins.severity:8s}] {ins.title}")
    print(f"  ... ({len(insights.insights)} total)")

    print(f"\n--- Recommendations ---")
    for rec in insights.recommendations[:3]:
        print(f"  [{rec.priority:6s}] {rec.title}")
    print(f"  ... ({len(insights.recommendations)} total)")

    print(f"\n--- Visualisation Suggestions ---")
    for vr in insights.viz_recommendations:
        print(f"  {vr.plot_type:25s} [{vr.priority}] — {vr.reason[:60]}...")

    print(f"\n--- Executive Summary ---")
    print(f"  {insights.executive_summary.narrative}")

    print(f"\n--- Reports ---")
    if reports.html:
        print(f"  HTML:     {reports.html}")
    if reports.markdown:
        print(f"  Markdown: {reports.markdown}")
    if reports.pdf:
        print(f"  PDF:      {reports.pdf}")

    print(f"\n--- Figures ---")
    print(f"  Total: {results['figures'].count}")
    for category in ("distribution", "comparison", "relationship", "matrix", "quality", "timeseries"):
        figs = getattr(results["figures"], category)
        if figs:
            print(f"  {category:15s}: {len(figs)}")

    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
