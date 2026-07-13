"""Professional report generation: HTML, PDF, Markdown.

This module performs the final stage of the EDA pipeline.  It consumes
every result produced by Phases 2-5 and renders consulting-firm-quality
reports.
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from jinja2 import Template

from autoeda._version import __version__
from autoeda.config import AutoEDAConfig
from autoeda.exceptions import ReportGenerationError

if TYPE_CHECKING:
    from pathlib import Path

    from autoeda.analytics import StatisticalAnalysis
    from autoeda.insight_engine import InsightResult
    from autoeda.profiler import DatasetProfile
    from autoeda.visualization import VisualizationResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReportResult:
    """Paths to generated report files.

    Attributes:
        html: Path to the HTML report.
        pdf: Path to the PDF report.
        markdown: Path to the Markdown report.
    """

    html: Path | None = None
    pdf: Path | None = None
    markdown: Path | None = None


# ---------------------------------------------------------------------------
# HTML template — 11-section consulting-firm design
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = Template(
    r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AutoEDA Report</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',Roboto,Helvetica,Arial,sans-serif;background:#f4f6f9;color:#333;line-height:1.6}
.container{max-width:1100px;margin:0 auto;padding:20px 30px}
.header{background:linear-gradient(135deg,#1a237e 0%,#283593 100%);color:#fff;padding:40px 30px;border-radius:10px;margin-bottom:30px}
.header h1{font-size:28px;font-weight:700;margin-bottom:6px}
.header .subtitle{font-size:14px;opacity:.85}
.kpi-row{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:28px}
.kpi{flex:1;min-width:150px;background:#fff;border-radius:8px;padding:18px 20px;box-shadow:0 1px 4px rgba(0,0,0,.08);text-align:center}
.kpi .value{font-size:26px;font-weight:700;color:#1a237e}
.kpi .label{font-size:12px;color:#666;margin-top:4px;text-transform:uppercase;letter-spacing:.5px}
.kpi .sub{font-size:11px;color:#999;margin-top:2px}
.section{background:#fff;border-radius:8px;padding:24px 28px;margin-bottom:22px;box-shadow:0 1px 4px rgba(0,0,0,.06)}
.section h2{font-size:18px;color:#1a237e;margin-bottom:14px;padding-bottom:8px;border-bottom:2px solid #e8eaf6}
.section h3{font-size:15px;color:#283593;margin:16px 0 8px}
.section p{margin:8px 0;font-size:14px}
table{width:100%;border-collapse:collapse;margin:12px 0;font-size:13px}
th{background:#e8eaf6;color:#1a237e;text-align:left;padding:10px 12px;font-weight:600}
td{padding:8px 12px;border-bottom:1px solid #eee}
tr:hover td{background:#f5f7ff}
.badge{display:inline-block;padding:3px 10px;border-radius:12px;font-size:11px;font-weight:600;text-transform:uppercase}
.badge.critical{background:#ffebee;color:#c62828}
.badge.warning{background:#fff3e0;color:#e65100}
.badge.info{background:#e3f2fd;color:#1565c0}
.badge.high{background:#ffebee;color:#c62828}
.badge.medium{background:#fff3e0;color:#e65100}
.badge.low{background:#e8f5e9;color:#2e7d32}
.insight{padding:12px 16px;border-left:4px solid #1a237e;margin:10px 0;background:#fafbff;border-radius:0 6px 6px 0}
.insight .title{font-weight:600;font-size:14px;margin-bottom:4px}
.insight .detail{font-size:13px;color:#555}
.insight .source{font-size:11px;color:#999;margin-top:4px;font-family:monospace}
.rec{padding:12px 16px;border-left:4px solid #2e7d32;margin:10px 0;background:#f6fff6;border-radius:0 6px 6px 0}
.rec .title{font-weight:600;font-size:14px;margin-bottom:4px}
.rec .detail{font-size:13px;color:#555}
.figure{text-align:center;margin:16px 0}
.figure img{max-width:100%;border-radius:6px;box-shadow:0 2px 8px rgba(0,0,0,.1)}
.figure .caption{font-size:12px;color:#888;margin-top:6px}
.health-bar{height:12px;border-radius:6px;background:#eee;margin:8px 0;overflow:hidden}
.health-bar .fill{height:100%;border-radius:6px}
.health-bar .fill.good{background:linear-gradient(90deg,#2e7d32,#66bb6a)}
.health-bar .fill.fair{background:linear-gradient(90deg,#e65100,#ffa726)}
.health-bar .fill.poor{background:linear-gradient(90deg,#c62828,#ef5350)}
.toc{columns:2;column-gap:32px;margin:12px 0}
.toc a{display:block;padding:4px 0;font-size:13px;color:#1a237e;text-decoration:none}
.toc a:hover{text-decoration:underline}
.footer{text-align:center;padding:20px;color:#aaa;font-size:12px}
</style>
</head>
<body>
<div class="container">

<!-- 1. Header -->
<div class="header">
  <h1>AutoEDA &mdash; Exploratory Data Analysis Report</h1>
  <div class="subtitle">Generated {{ timestamp }} &bull; AutoEDA v{{ version }}</div>
</div>

<!-- Table of Contents -->
<div class="section">
  <h2>Contents</h2>
  <div class="toc">
    <a href="#exec-summary">1. Executive Summary</a>
    <a href="#dataset-overview">2. Dataset Overview</a>
    <a href="#health">3. Data Health Score</a>
    <a href="#variable-summary">4. Variable Summary</a>
    <a href="#statistical-analysis">5. Statistical Analysis</a>
    <a href="#correlation-analysis">6. Correlation Analysis</a>
    <a href="#distribution-analysis">7. Distribution Analysis</a>
    <a href="#outlier-analysis">8. Outlier Analysis</a>
    <a href="#visualisations">9. Visualisations</a>
    <a href="#insights">10. Key Insights &amp; Recommendations</a>
    <a href="#appendix">11. Appendix</a>
  </div>
</div>

<!-- 2. KPI Cards -->
<div class="kpi-row">
  <div class="kpi"><div class="value">{{ profile.n_rows | int }}</div><div class="label">Rows</div></div>
  <div class="kpi"><div class="value">{{ profile.n_columns }}</div><div class="label">Columns</div></div>
  <div class="kpi"><div class="value">{{ profile.memory_human }}</div><div class="label">Memory</div></div>
  <div class="kpi"><div class="value">{{ profile.missing_pct }}%</div><div class="label">Missing</div></div>
  <div class="kpi"><div class="value">{{ exec_sum.n_numerical }}</div><div class="label">Numeric</div></div>
  <div class="kpi"><div class="value">{{ exec_sum.n_categorical }}</div><div class="label">Categorical</div></div>
  <div class="kpi">
    <div class="value">{{ profile.health_score.overall }}/100</div>
    <div class="label">Health Score</div>
    <div class="sub">{{ profile.health_score.label }}</div>
  </div>
</div>

<!-- 1. Executive Summary -->
<div class="section" id="exec-summary">
  <h2>1. Executive Summary</h2>
  {% if exec_sum.narrative %}
  <p>{{ exec_sum.narrative }}</p>
  {% endif %}
  {% for finding in exec_sum.key_findings %}
  <p style="margin:6px 0">&#8226; {{ finding }}</p>
  {% endfor %}
</div>

<!-- 2. Dataset Overview -->
<div class="section" id="dataset-overview">
  <h2>2. Dataset Overview</h2>
  <table>
    <tr><th>Metric</th><th>Value</th></tr>
    <tr><td>Shape</td><td>{{ profile.n_rows }} rows &times; {{ profile.n_columns }} columns</td></tr>
    <tr><td>Memory Usage</td><td>{{ profile.memory_human }}</td></tr>
    <tr><td>Total Missing Cells</td><td>{{ profile.missing_total }} ({{ profile.missing_pct }}%)</td></tr>
    <tr><td>Duplicate Rows</td><td>{{ profile.duplicate_rows }} ({{ profile.duplicate_pct }}%)</td></tr>
  </table>
</div>

<!-- 3. Data Health Score -->
<div class="section" id="health">
  <h2>3. Data Health Score</h2>
  <p><strong>Overall: {{ profile.health_score.overall }}/100 &mdash; {{ profile.health_score.label }}</strong></p>
  <div class="health-bar"><div class="fill {% if profile.health_score.overall >= 70 %}good{% elif profile.health_score.overall >= 40 %}fair{% else %}poor{% endif %}" style="width:{{ profile.health_score.overall }}%"></div></div>
  <table>
    <tr><th>Component</th><th>Score</th></tr>
    <tr><td>Completeness (30%)</td><td>{{ profile.health_score.completeness }}/100</td></tr>
    <tr><td>Uniqueness (20%)</td><td>{{ profile.health_score.uniqueness }}/100</td></tr>
    <tr><td>Consistency (20%)</td><td>{{ profile.health_score.consistency }}/100</td></tr>
    <tr><td>Analysis Readiness (30%)</td><td>{{ profile.health_score.analysis_readiness }}/100</td></tr>
  </table>
</div>

<!-- 4. Variable Summary -->
<div class="section" id="variable-summary">
  <h2>4. Variable Summary</h2>
  {% if profile.variables %}
  <table>
    <tr><th>Variable</th><th>Type</th><th>Dtype</th><th>Unique</th><th>Unique Ratio</th><th>Role</th></tr>
    {% for v in profile.variables %}
    <tr>
      <td><strong>{{ v.column }}</strong></td>
      <td>{{ v.semantic_type }}</td>
      <td>{{ v.dtype }}</td>
      <td>{{ v.n_unique }}</td>
      <td>{{ "%.1f" | format(v.unique_ratio * 100) }}%</td>
      <td>{{ v.suggested_role }}</td>
    </tr>
    {% endfor %}
  </table>
  {% else %}
  <p><em>Variable-level summary not available.</em></p>
  {% endif %}
</div>

<!-- 5. Statistical Analysis -->
<div class="section" id="statistical-analysis">
  <h2>5. Statistical Analysis</h2>
  {% if profile.numerical_stats %}
  <h3>Numerical Features</h3>
  <table>
    <tr><th>Feature</th><th>Mean</th><th>Median</th><th>Std</th><th>Skewness</th><th>Kurtosis</th><th>Outliers</th></tr>
    {% for s in profile.numerical_stats %}
    <tr>
      <td><strong>{{ s.column }}</strong></td>
      <td>{{ "%.2f" | format(s.mean) }}</td>
      <td>{{ "%.2f" | format(s.median) }}</td>
      <td>{{ "%.2f" | format(s.std) }}</td>
      <td>{{ "%.2f" | format(s.skewness) }}</td>
      <td>{{ "%.2f" | format(s.kurtosis) }}</td>
      <td>{{ s.n_outliers }} ({{ s.outlier_pct }}%)</td>
    </tr>
    {% endfor %}
  </table>
  {% endif %}

  {% if profile.categorical_stats %}
  <h3>Categorical Features</h3>
  <table>
    <tr><th>Feature</th><th>Unique</th><th>Top Value</th><th>Top %</th><th>Missing</th></tr>
    {% for s in profile.categorical_stats %}
    <tr>
      <td><strong>{{ s.column }}</strong></td>
      <td>{{ s.n_unique }}</td>
      <td>{{ s.top_value }}</td>
      <td>{{ s.top_pct }}%</td>
      <td>{{ s.missing_pct }}%</td>
    </tr>
    {% endfor %}
  </table>
  {% endif %}

  {% if stats.normality.tests %}
  <h3>Normality Tests</h3>
  <p>{{ stats.normality.n_normal }} of {{ stats.normality.tests | length }} features are normally distributed.</p>
  <table>
    <tr><th>Feature</th><th>Shapiro-Wilk p</th><th>D'Agostino p</th><th>KS p</th><th>Normal?</th></tr>
    {% for t in stats.normality.tests %}
    <tr>
      <td><strong>{{ t.column }}</strong></td>
      <td>{{ "%.6f" | format(t.shapiro_p) }}</td>
      <td>{{ "%.6f" | format(t.dagostino_p) }}</td>
      <td>{{ "%.6f" | format(t.ks_p) }}</td>
      <td><span class="badge {{ 'info' if t.is_normal else 'warning' }}">{{ 'Yes' if t.is_normal else 'No' }}</span></td>
    </tr>
    {% endfor %}
  </table>
  {% endif %}

  {% if stats.hypothesis_tests %}
  <h3>Hypothesis Tests</h3>
  <table>
    <tr><th>Test</th><th>Column</th><th>Grouping</th><th>Statistic</th><th>p-value</th><th>Result</th></tr>
    {% for t in stats.hypothesis_tests %}
    <tr>
      <td>{{ t.test_name | replace('_', ' ') | title }}</td>
      <td>{{ t.column }}</td>
      <td>{{ t.categorical_column }}{% if t.group_a %} ({{ t.group_a }} vs {{ t.group_b }}){% endif %}</td>
      <td>{{ "%.4f" | format(t.statistic) }}</td>
      <td>{{ "%.6f" | format(t.p_value) }}</td>
      <td><span class="badge {{ 'info' if t.significant else 'low' }}">{{ t.interpretation }}</span></td>
    </tr>
    {% endfor %}
  </table>
  {% endif %}
</div>

<!-- 6. Correlation Analysis -->
<div class="section" id="correlation-analysis">
  <h2>6. Correlation Analysis</h2>
  {% if stats.pearson.matrix %}
  <p><strong>Pearson correlations:</strong> {{ stats.pearson.n_significant }} significant pair(s).</p>
  {% if stats.pearson.significant_pairs %}
  <table>
    <tr><th>Feature A</th><th>Feature B</th><th>r</th><th>Interpretation</th><th>p-value</th></tr>
    {% for p in stats.pearson.significant_pairs %}
    <tr>
      <td>{{ p.col_a }}</td><td>{{ p.col_b }}</td>
      <td>{{ "%.4f" | format(p.coefficient) }}</td>
      <td><span class="badge {{ 'info' if p.significant else 'low' }}">{{ p.interpretation }}</span></td>
      <td>{{ "%.6f" | format(p.p_value) }}</td>
    </tr>
    {% endfor %}
  </table>
  {% endif %}
  {% endif %}
</div>

<!-- 7. Distribution Analysis -->
<div class="section" id="distribution-analysis">
  <h2>7. Distribution Analysis</h2>
  {% set skewed = [] %}
  {% for s in profile.numerical_stats %}{% if (s.skewness | abs) > skew_threshold %}{% if skewed.append(s) %}{% endif %}{% endif %}{% endfor %}
  {% if skewed | length > 0 %}
  <p><strong>{{ skewed | length }} feature(s) with heavy skewness (|skew| &gt; {{ skew_threshold }}):</strong></p>
  <table>
    <tr><th>Feature</th><th>Skewness</th><th>Kurtosis</th><th>Recommendation</th></tr>
    {% for s in skewed %}
    <tr>
      <td><strong>{{ s.column }}</strong></td>
      <td>{{ "%.2f" | format(s.skewness) }}</td>
      <td>{{ "%.2f" | format(s.kurtosis) }}</td>
      <td>{% if (s.skewness | abs) > 2 %}Consider log/Box-Cox transformation{% else %}Mild skew — monitor{% endif %}</td>
    </tr>
    {% endfor %}
  </table>
  {% else %}
  <p>All numeric distributions are approximately symmetric.</p>
  {% endif %}
</div>

<!-- 8. Outlier Analysis -->
<div class="section" id="outlier-analysis">
  <h2>8. Outlier Analysis</h2>
  {% set outlier_cols = [] %}
  {% for s in profile.numerical_stats %}{% if s.n_outliers > 0 %}{% if outlier_cols.append(s) %}{% endif %}{% endif %}{% endfor %}
  {% if outlier_cols | length > 0 %}
  <p><strong>{{ outlier_cols | length }} feature(s) contain outliers (IQR method):</strong></p>
  <table>
    <tr><th>Feature</th><th>Outliers</th><th>Outlier %</th><th>Min</th><th>Q1</th><th>Q3</th><th>Max</th></tr>
    {% for s in outlier_cols %}
    <tr>
      <td><strong>{{ s.column }}</strong></td>
      <td>{{ s.n_outliers }}</td>
      <td>{{ s.outlier_pct }}%</td>
      <td>{{ "%.2f" | format(s.min) }}</td>
      <td>{{ "%.2f" | format(s.q25) }}</td>
      <td>{{ "%.2f" | format(s.q75) }}</td>
      <td>{{ "%.2f" | format(s.max) }}</td>
    </tr>
    {% endfor %}
  </table>
  {% else %}
  <p>No significant outliers detected.</p>
  {% endif %}
</div>

<!-- 9. Visualisations -->
<div class="section" id="visualisations">
  <h2>9. Visualisations</h2>
  {% if figures.all_figures | length > 0 %}
  {% for fig in figures.all_figures %}
  <div class="figure">
    <img src="data:image/png;base64,{{ fig_b64[fig.filename] }}" alt="{{ fig.title }}">
    <div class="caption">{{ fig.title }} &mdash; {{ fig.description }}</div>
  </div>
  {% endfor %}
  {% else %}
  <p><em>No visualisations generated.</em></p>
  {% endif %}
</div>

<!-- 10. Insights & Recommendations -->
<div class="section" id="insights">
  <h2>10. Key Insights &amp; Recommendations</h2>
  {% if insights.viz_recommendations | length > 0 %}
  <h3>Suggested Visualisations</h3>
  <table>
    <tr><th>Plot Type</th><th>Priority</th><th>Why</th><th>Columns</th></tr>
    {% for vr in insights.viz_recommendations %}
    <tr>
      <td><strong>{{ vr.plot_type }}</strong></td>
      <td><span class="badge {{ vr.priority }}">{{ vr.priority }}</span></td>
      <td>{{ vr.reason }}</td>
      <td>{{ vr.columns | join(', ') }}</td>
    </tr>
    {% endfor %}
  </table>
  {% endif %}

  {% if insights.insights | length > 0 %}
  <h3>Insights</h3>
  {% for ins in insights.insights %}
  <div class="insight">
    <div class="title"><span class="badge {{ ins.severity }}">{{ ins.severity }}</span> {{ ins.title }}</div>
    <div class="detail">{{ ins.detail }}</div>
    <div class="source">Source: {{ ins.source_metric }}</div>
  </div>
  {% endfor %}
  {% endif %}

  {% if insights.recommendations | length > 0 %}
  <h3>Recommendations</h3>
  {% for rec in insights.recommendations %}
  <div class="rec">
    <div class="title"><span class="badge {{ rec.priority }}">{{ rec.priority }}</span> {{ rec.title }}</div>
    <div class="detail">{{ rec.detail }}</div>
  </div>
  {% endfor %}
  {% endif %}
</div>

<!-- 11. Appendix -->
<div class="section" id="appendix">
  <h2>11. Appendix</h2>
  <table>
    <tr><th>Metric</th><th>Value</th></tr>
    <tr><td>Numerical Features</td><td>{{ exec_sum.n_numerical }}</td></tr>
    <tr><td>Categorical Features</td><td>{{ exec_sum.n_categorical }}</td></tr>
    <tr><td>Datetime Features</td><td>{{ exec_sum.n_datetime }}</td></tr>
    <tr><td>Boolean Features</td><td>{{ exec_sum.n_boolean }}</td></tr>
    <tr><td>Strong Correlations</td><td>{{ exec_sum.n_strong_correlations }}</td></tr>
    <tr><td>Skewed Features</td><td>{{ exec_sum.n_skewed_features }}</td></tr>
    <tr><td>Features with Outliers</td><td>{{ exec_sum.n_outlier_features }}</td></tr>
    <tr><td>Significant Tests</td><td>{{ exec_sum.n_significant_tests }}</td></tr>
  </table>
</div>

<div class="footer">
  AutoEDA v{{ version }} &bull; Report generated {{ timestamp }}
</div>

</div>
</body>
</html>""",
)


# ---------------------------------------------------------------------------
# Report Generator
# ---------------------------------------------------------------------------


class ReportGenerator:
    """Generate professional reports from EDA pipeline results.

    Example::

        gen = ReportGenerator(config)
        paths = gen.generate(profile, stats, figures, insights)
    """

    def __init__(self, config: AutoEDAConfig | None = None) -> None:
        self.config = config or AutoEDAConfig()

    def generate(
        self,
        profile: DatasetProfile,
        stats: StatisticalAnalysis,
        figures: VisualizationResult,
        insights: InsightResult,
    ) -> ReportResult:
        """Generate all configured report formats.

        Args:
            profile: Profiling result from Phase 2.
            stats: Statistical analysis from Phase 3.
            figures: Visualisation result from Phase 4.
            insights: Insight result from Phase 5.

        Returns:
            A :class:`ReportResult` with paths to generated files.
        """
        if not self.config.save_reports:
            logger.info("Report saving disabled in config")
            return ReportResult()

        report_dir = self.config.output_dir
        report_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M UTC")

        fig_b64 = self._encode_figures(figures)

        ctx: dict[str, Any] = {
            "timestamp": timestamp,
            "version": __version__,
            "profile": profile,
            "stats": stats,
            "figures": figures,
            "insights": insights,
            "exec_sum": insights.executive_summary,
            "fig_b64": fig_b64,
            "skew_threshold": self.config.skewness_threshold,
        }

        html_path: Path | None = None
        pdf_path: Path | None = None
        md_path: Path | None = None

        base_name = self.config.report_filename

        if "html" in self.config.report_formats:
            html_path = self._render_html(ctx, report_dir, base_name)
        if "pdf" in self.config.report_formats:
            pdf_path = self._render_pdf(
                html_path or self._render_html(ctx, report_dir, base_name), report_dir, base_name
            )
        if "markdown" in self.config.report_formats:
            md_path = self._render_markdown(ctx, report_dir, base_name)

        result = ReportResult(html=html_path, pdf=pdf_path, markdown=md_path)
        logger.info("Reports generated: %s", result)
        return result

    # ------------------------------------------------------------------
    # HTML
    # ------------------------------------------------------------------

    def _render_html(
        self,
        ctx: dict[str, Any],
        report_dir: Path,
        base_name: str,
    ) -> Path:
        """Render the HTML report."""
        html = _HTML_TEMPLATE.render(**ctx)
        path = report_dir / f"{base_name}.html"
        path.write_text(html, encoding="utf-8")
        logger.info("HTML report saved to %s", path)
        return path

    # ------------------------------------------------------------------
    # PDF
    # ------------------------------------------------------------------

    def _render_pdf(
        self,
        html_path: Path,
        report_dir: Path,
        base_name: str,
    ) -> Path:
        """Render the PDF report from HTML via weasyprint."""
        try:
            from weasyprint import HTML as WeasyHTML
        except ImportError as exc:
            raise ReportGenerationError(
                "weasyprint is required for PDF generation. Install with: pip install weasyprint"
            ) from exc

        pdf_path = report_dir / f"{base_name}.pdf"
        try:
            WeasyHTML(filename=str(html_path)).write_pdf(str(pdf_path))
            logger.info("PDF report saved to %s", pdf_path)
        except Exception as exc:
            raise ReportGenerationError(f"PDF generation failed: {exc}") from exc
        return pdf_path

    # ------------------------------------------------------------------
    # Markdown
    # ------------------------------------------------------------------

    def _render_markdown(
        self,
        ctx: dict[str, Any],
        report_dir: Path,
        base_name: str,
    ) -> Path:
        """Render the Markdown report."""
        profile: DatasetProfile = ctx["profile"]
        stats: StatisticalAnalysis = ctx["stats"]
        figures: VisualizationResult = ctx["figures"]
        insights: InsightResult = ctx["insights"]
        exec_sum = ctx["exec_sum"]
        timestamp: str = ctx["timestamp"]

        lines: list[str] = []
        _a = lines.append

        _a("# AutoEDA Report\n")
        _a(f"*Generated: {timestamp}*\n")
        _a("---\n")

        # Executive Summary
        _a("## Executive Summary\n")
        if exec_sum.narrative:
            _a(f"{exec_sum.narrative}\n")
        for f in exec_sum.key_findings:
            _a(f"- {f}")
        _a("")

        # Dataset Overview
        _a("## Dataset Overview\n")
        _a(
            f"**Health Score:** {profile.health_score.overall}/100 ({profile.health_score.label})\n"
        )
        _a("| Metric | Value |")
        _a("|--------|-------|")
        _a(f"| Rows | {profile.n_rows} |")
        _a(f"| Columns | {profile.n_columns} |")
        _a(f"| Memory | {profile.memory_human} |")
        _a(f"| Missing | {profile.missing_pct}% |")
        _a(f"| Duplicates | {profile.duplicate_rows} ({profile.duplicate_pct}%) |")
        _a("")

        # Variable Summary
        if profile.variables:
            _a("## Variable Summary\n")
            _a("| Variable | Type | Dtype | Unique | Role |")
            _a("|----------|------|-------|--------|------|")
            for v in profile.variables:
                _a(
                    f"| {v.column} | {v.semantic_type} | {v.dtype} | {v.n_unique} | {v.suggested_role} |"
                )
            _a("")

        # Numerical
        if profile.numerical_stats:
            _a("## Numerical Features\n")
            _a("| Feature | Mean | Median | Std | Skew | Kurtosis | Outliers |")
            _a("|---------|------|--------|-----|------|----------|----------|")
            for s in profile.numerical_stats:
                _a(
                    f"| {s.column} | {s.mean:.2f} | {s.median:.2f} "
                    f"| {s.std:.2f} | {s.skewness:.2f} | {s.kurtosis:.2f} "
                    f"| {s.n_outliers} ({s.outlier_pct}%) |"
                )
            _a("")

        # Categorical
        if profile.categorical_stats:
            _a("## Categorical Features\n")
            _a("| Feature | Unique | Top Value | Top % | Missing |")
            _a("|---------|--------|-----------|-------|---------|")
            for cs in profile.categorical_stats:
                _a(
                    f"| {cs.column} | {cs.n_unique} | {cs.top_value} | {cs.top_pct}% | {cs.missing_pct}% |"
                )
            _a("")

        # Correlation
        if stats.pearson.significant_pairs:
            _a("## Correlation Analysis\n")
            _a("| Feature A | Feature B | r | Interpretation | p-value |")
            _a("|-----------|-----------|---|----------------|---------|")
            for p in stats.pearson.significant_pairs:
                _a(
                    f"| {p.col_a} | {p.col_b} | {p.coefficient:.4f} "
                    f"| {p.interpretation} | {p.p_value:.6f} |"
                )
            _a("")

        # Hypothesis tests
        if stats.hypothesis_tests:
            _a("## Statistical Tests\n")
            _a("| Test | Column | Grouping | Statistic | p-value | Interpretation |")
            _a("|------|--------|----------|-----------|---------|----------------|")
            for t in stats.hypothesis_tests:
                grp = t.categorical_column
                if t.group_a:
                    grp += f" ({t.group_a} vs {t.group_b})"
                _a(
                    f"| {t.test_name} | {t.column} | {grp} "
                    f"| {t.statistic:.4f} | {t.p_value:.6f} "
                    f"| {t.interpretation} |"
                )
            _a("")

        # Figures
        if figures.all_figures:
            _a("## Visualisations\n")
            for fig in figures.all_figures:
                _a(f"### {fig.title}\n")
                if fig.path:
                    _a(f"![{fig.title}]({fig.path.as_posix()})\n")
                _a(f"*{fig.description}*\n")

        # Viz Recommendations
        if insights.viz_recommendations:
            _a("## Suggested Visualisations\n")
            _a("| Plot Type | Priority | Reason | Columns |")
            _a("|-----------|----------|--------|---------|")
            for vr in insights.viz_recommendations:
                _a(f"| {vr.plot_type} | {vr.priority} | {vr.reason} | {', '.join(vr.columns)} |")
            _a("")

        # Insights
        if insights.insights:
            _a("## Business Insights\n")
            for ins in insights.insights:
                _a(f"**[{ins.severity.upper()}]** {ins.title}\n")
                _a(f"> {ins.detail}\n")

        # Recommendations
        if insights.recommendations:
            _a("## Recommendations\n")
            for rec in insights.recommendations:
                _a(f"**[{rec.priority.upper()}]** {rec.title}\n")
                _a(f"> {rec.detail}\n")

        _a("---\n")
        _a(f"*AutoEDA v{ctx['version']}*\n")

        md_path = report_dir / f"{base_name}.md"
        md_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Markdown report saved to %s", md_path)
        return md_path

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_figures(figures: VisualizationResult) -> dict[str, str]:
        """Encode all figure files as base64 strings."""
        encoded: dict[str, str] = {}
        for fig in figures.all_figures:
            if fig.path and fig.path.exists():
                data = fig.path.read_bytes()
                encoded[fig.filename] = base64.b64encode(data).decode("ascii")
            elif fig.path:
                logger.warning("Figure file not found: %s", fig.path)
        return encoded
