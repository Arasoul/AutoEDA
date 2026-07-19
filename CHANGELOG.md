# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2026-07-20

### Changed

- Version bumped from 0.1.0 to 1.0.0 for stable release.
- Single-source versioning via `_version.py` (PEP 561, hatchling dynamic version).
- Added `py.typed` marker for PEP 561 compliance.

### Fixed

- Fixed weak typing (`object | None`) in `InsightEngine.generate` to use `VisualizationResult | None`.

### Improved

- CI coverage threshold enforced at 85%.
- README documents ecosystem relationship, tested dataset ranges, and WeasyPrint requirements.

## [0.1.0] - 2025-07-13

### Added

- Project foundation: config, exceptions, utilities, public API stub.
- DataProfiler with variable classification and health scoring.
- Analytics module: Pearson/Spearman/Kendall correlations, normality tests, hypothesis tests, confidence intervals.
- Correlation interpretation with auto-generated text descriptions.
- Visualization module: 12 figure types (histograms, KDE, boxplots, violin, countplots, scatter, pairplot, bubble, correlation heatmap, missing heatmap, outlier viz, time series).
- InsightEngine: rule-based business insights, recommendations, visualization suggestions, executive summary with narrative.
- ReportGenerator: 11-section consulting-firm HTML template, Markdown, PDF (via weasyprint).
- 187 unit tests across 8 test modules.
- CI/CD pipeline (GitHub Actions): lint, typecheck, test on Python 3.11-3.13.
- README, examples, CONTRIBUTING, CHANGELOG, LICENSE.
