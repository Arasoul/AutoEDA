# AutoEDA v1.0.0

First stable release of AutoEDA — a production-quality Python library for automated exploratory data analysis, statistical analysis, visualization, and professional reporting.

## Overview

AutoEDA automates the entire exploratory analysis workflow and produces consulting-firm-quality reports (Deloitte/PwC style) with a single function call. It focuses purely on EDA — no preprocessing, no machine learning — just fast, deterministic, traceable insights.

The library is designed for data analysts, business intelligence engineers, and data scientists who need a consistent, tested, and well-documented EDA pipeline.

## Major Features

### Dataset Profiling
- Variable classification into 9 semantic types: numeric, binary, categorical, ordinal, datetime, boolean, identifier, text, constant
- Health score with 4 weighted components: completeness (30%), uniqueness (20%), consistency (20%), analysis readiness (30%)
- Missing value, outlier, and duplicate detection

### Statistical Analysis
- Pearson, Spearman, and Kendall correlation with auto-generated interpretations
- Shapiro-Wilk, D'Agostino, and Kolmogorov-Smirnov normality tests
- Independent t-test (2 groups), one-way ANOVA (3+ groups), chi-square test of independence
- Confidence intervals for the mean with configurable confidence levels

### Visualizations (12 Types)
- Distribution: Histograms, KDE Plots
- Comparison: Boxplots, Violin Plots, Count Plots
- Relationship: Scatter Plots, Pair Plot, Bubble Charts
- Matrix: Correlation Heatmap
- Quality: Missing Values Heatmap, Outlier Visualization
- Time Series: Line Charts (datetime-indexed)

### Business Insights
- Rule-based insights with severity ratings and traceable source metrics
- Actionable recommendations with priority levels
- Visualization suggestions based on data characteristics
- Executive summary with narrative prose

### Professional Reports
- 11-section consulting-firm HTML template with KPI cards, color-coded tables, and inline base64 charts
- Markdown report for version control and sharing
- PDF report via WeasyPrint (optional dependency)

## Engineering Improvements

- **187 unit tests** with 92.51% code coverage
- **Type hints** on all public APIs with mypy enforcement
- **PEP 561 compliant** (`py.typed` marker for downstream type checkers)
- **Hardened CI/CD**: ruff lint, ruff format check, mypy, coverage floor (85%), multi-Python matrix (3.11, 3.12, 3.13)
- **Clean dependency footprint**: pandas, numpy, scipy, matplotlib, seaborn, jinja2 as runtime dependencies
- **Immutable configuration**: `AutoEDAConfig` dataclass with validated defaults
- **Consistent exception hierarchy**: all errors inherit from `AutoEDAError`
- **Production/Stable classifier** on PyPI

## Breaking Changes

No breaking changes. This is the first stable public release.

## Known Limitations

- **In-memory processing**: All operations run on Pandas DataFrames in memory. Larger-than-memory datasets are not supported.
- **Configuration via Python objects**: No YAML/CLI configuration interface; all settings are passed through `AutoEDAConfig`.
- **WeasyPrint for PDF**: PDF generation requires system-level WeasyPrint dependencies (Pango, Cairo) on Linux.

## Installation

```bash
pip install autoeda
```

For PDF report generation:

```bash
pip install autoeda[pdf]
```

Requires Python 3.11+.

## Quick Start

```python
import pandas as pd
from autoeda import AutoEDA

# Load your data
df = pd.read_csv("your_data.csv")

# Run the full EDA pipeline
results = AutoEDA().run(df)

# Access everything
results["profile"]         # DatasetProfile — shape, types, health score
results["statistics"]      # StatisticalAnalysis — correlations, tests, CIs
results["figures"]         # VisualizationResult — 12+ figure types
results["insights"]        # InsightResult — insights, recommendations, narrative
results["report_paths"]    # ReportResult — paths to HTML/MD/PDF reports
```

## Ecosystem

AutoEDA is the second component of a modular data analysis ecosystem:

| Component | Purpose | Status |
|-----------|---------|--------|
| [DataPrepToolkit](https://github.com/Arasoul/DataPrepToolkit) | Data cleaning, imputation, validation, optimization | v1.0.0 |
| **AutoEDA** | Exploratory data analysis, statistics, visualization, reporting | v1.0.0 |
| AutoAnalytics | Advanced analytics | Coming soon |
| AutoBI | Business intelligence | Coming soon |

Each library works independently — there is no runtime dependency between them.

## Future Roadmap

- **AutoAnalytics** — Automated statistical modeling and advanced analytics
- **AutoBI** — Automated business intelligence dashboard generation

## Links

- [Documentation](https://github.com/Arasoul/AutoEDA#readme)
- [Changelog](CHANGELOG.md)
- [Contributing Guide](CONTRIBUTING.md)
- [Issue Tracker](https://github.com/Arasoul/AutoEDA/issues)

## License

MIT License
