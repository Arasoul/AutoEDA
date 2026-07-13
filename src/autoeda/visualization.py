"""Visualisation: histograms, KDE, boxplots, scatterplots, heatmaps, etc.

This module performs the third stage of the EDA pipeline.  It consumes
the :class:`~autoeda.profiler.DatasetProfile` and
:class:`~autoeda.analytics.StatisticalAnalysis` produced by Phases 2-3
and generates publication-quality figures.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from autoeda.config import AutoEDAConfig
from autoeda.utils import validate_dataframe

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd

    from autoeda.analytics import StatisticalAnalysis
    from autoeda.profiler import DatasetProfile

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FigureResult:
    """Metadata for a single generated figure."""

    title: str
    description: str
    category: str
    path: Path | None
    filename: str


@dataclass
class VisualizationResult:
    """Aggregated results of all visualisation passes.

    Intentionally mutable (unfrozen) because figures are appended
    incrementally during the ``generate_all`` pass.

    Attributes:
        distribution: Histograms and KDE plots.
        comparison: Boxplots, violin plots, countplots.
        relationship: Scatterplots, pairplots, bubble charts.
        matrix: Correlation heatmaps.
        quality: Missing-value and outlier visualisations.
        timeseries: Line plots for datetime columns.
    """

    distribution: list[FigureResult] = field(default_factory=list)
    comparison: list[FigureResult] = field(default_factory=list)
    relationship: list[FigureResult] = field(default_factory=list)
    matrix: list[FigureResult] = field(default_factory=list)
    quality: list[FigureResult] = field(default_factory=list)
    timeseries: list[FigureResult] = field(default_factory=list)

    @property
    def all_figures(self) -> list[FigureResult]:
        """Return every figure across all categories."""
        return (
            self.distribution
            + self.comparison
            + self.relationship
            + self.matrix
            + self.quality
            + self.timeseries
        )

    @property
    def count(self) -> int:
        """Total number of figures generated."""
        return len(self.all_figures)


# ---------------------------------------------------------------------------
# Visualiser
# ---------------------------------------------------------------------------


class Visualization:
    """Generate publication-quality figures for an EDA pipeline.

    Example::

        viz = Visualization(config)
        result = viz.generate_all(df, profile, stats)
    """

    def __init__(self, config: AutoEDAConfig | None = None) -> None:
        self.config = config or AutoEDAConfig()
        self._fig_dir: Path | None = None
        if self.config.enable_plots:
            matplotlib.use("Agg")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_all(
        self,
        df: pd.DataFrame,
        profile: DatasetProfile,
        stats: StatisticalAnalysis,
    ) -> VisualizationResult:
        """Generate every available visualisation.

        Args:
            df: The cleaned DataFrame.
            profile: Profiling result from Phase 2.
            stats: Statistical analysis from Phase 3.

        Returns:
            A :class:`VisualizationResult` with all generated figures.
        """
        if not self.config.enable_plots:
            logger.info("Plots disabled in config — skipping visualisation")
            return VisualizationResult()

        validate_dataframe(df)
        self._fig_dir = self._ensure_output_dir()
        result = VisualizationResult()

        self._histograms(df, profile, result)
        self._kde_plots(df, profile, result)
        self._boxplots(df, profile, result)
        self._violin_plots(df, profile, result)
        self._countplots(df, profile, result)
        self._scatterplots(df, profile, result)
        self._pairplot(df, profile, result)
        self._correlation_heatmap(stats, result)
        self._bubble_charts(df, profile, result)
        self._missing_value_heatmap(df, result)
        self._outlier_visualisation(df, profile, result)
        self._timeseries_plots(df, profile, result)

        logger.info("Generated %d figures", result.count)
        return result

    # ------------------------------------------------------------------
    # Internal: helpers
    # ------------------------------------------------------------------

    def _ensure_output_dir(self) -> Path:
        """Create the figures output directory if needed."""
        fig_dir = self.config.output_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        return fig_dir

    def _save(
        self,
        fig: plt.Figure,
        name: str,
        result: VisualizationResult,
        category: str,
        title: str,
        desc: str,
    ) -> None:
        """Save a figure and append to the result list."""
        filename = f"{name}.png"
        path: Path | None = None
        if self.config.save_figures and self._fig_dir is not None:
            path = self._fig_dir / filename
            fig.savefig(path, dpi=self.config.figure_dpi, bbox_inches="tight")
        plt.close(fig)

        entry = FigureResult(
            title=title,
            description=desc,
            category=category,
            path=path,
            filename=filename,
        )
        getattr(result, category).append(entry)

    def _grid_layout(self, n: int) -> tuple[int, int]:
        """Compute nrows, ncols for up to max_grid_columns columns."""
        ncols = min(n, self.config.max_grid_columns)
        nrows = (n + ncols - 1) // ncols
        return nrows, ncols

    # ------------------------------------------------------------------
    # Distribution plots
    # ------------------------------------------------------------------

    def _histograms(
        self,
        df: pd.DataFrame,
        profile: DatasetProfile,
        result: VisualizationResult,
    ) -> None:
        """Generate histograms for numeric columns."""
        num_cols = profile.numerical_columns
        if not num_cols:
            return
        n = len(num_cols)
        nrows, ncols = self._grid_layout(n)
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(self.config.figure_width, self.config.figure_height * nrows / 2),
        )
        axes_arr = np.array(axes).flatten() if n > 1 else [axes]
        palette = sns.color_palette(self.config.color_palette, n)

        for i, col in enumerate(num_cols):
            ax = axes_arr[i]
            df[col].dropna().hist(
                ax=ax,
                bins=self.config.histogram_bins,
                color=palette[i],
                edgecolor="white",
                alpha=0.8,
            )
            ax.set_title(col, fontsize=self.config.label_fontsize, fontweight="bold")
            ax.set_ylabel("Frequency")

        for j in range(n, len(axes_arr)):
            axes_arr[j].set_visible(False)

        fig.suptitle(
            "Distribution of Numerical Features",
            fontsize=self.config.title_fontsize,
            fontweight="bold",
            y=1.02,
        )
        fig.tight_layout()
        self._save(
            fig,
            "histograms",
            result,
            "distribution",
            "Numerical Histograms",
            "Histograms showing the distribution of each numerical feature.",
        )

    def _kde_plots(
        self,
        df: pd.DataFrame,
        profile: DatasetProfile,
        result: VisualizationResult,
    ) -> None:
        """Generate KDE plots for numeric columns."""
        num_cols = profile.numerical_columns
        if not num_cols:
            return
        n = len(num_cols)
        nrows, ncols = self._grid_layout(n)
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(self.config.figure_width, self.config.figure_height * nrows / 2),
        )
        axes_arr = np.array(axes).flatten() if n > 1 else [axes]
        palette = sns.color_palette(self.config.color_palette, n)

        for i, col in enumerate(num_cols):
            ax = axes_arr[i]
            try:
                sns.kdeplot(data=df, x=col, ax=ax, fill=True, color=palette[i], alpha=0.6)
                ax.set_title(col, fontsize=self.config.label_fontsize, fontweight="bold")
            except Exception:
                ax.set_visible(False)

        for j in range(n, len(axes_arr)):
            axes_arr[j].set_visible(False)

        fig.suptitle(
            "Density Estimation (KDE)",
            fontsize=self.config.title_fontsize,
            fontweight="bold",
            y=1.02,
        )
        fig.tight_layout()
        self._save(
            fig,
            "kde_plots",
            result,
            "distribution",
            "KDE Plots",
            "Kernel density estimation for each numerical feature.",
        )

    # ------------------------------------------------------------------
    # Comparison plots
    # ------------------------------------------------------------------

    def _boxplots(
        self,
        df: pd.DataFrame,
        profile: DatasetProfile,
        result: VisualizationResult,
    ) -> None:
        """Generate boxplots for numeric columns."""
        num_cols = profile.numerical_columns
        if not num_cols:
            return
        n = len(num_cols)
        nrows, ncols = self._grid_layout(n)
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(self.config.figure_width, self.config.figure_height * nrows / 2),
        )
        axes_arr = np.array(axes).flatten() if n > 1 else [axes]
        palette = sns.color_palette(self.config.color_palette, n)

        for i, col in enumerate(num_cols):
            ax = axes_arr[i]
            sns.boxplot(data=df, y=col, ax=ax, color=palette[i])
            ax.set_title(col, fontsize=self.config.label_fontsize, fontweight="bold")

        for j in range(n, len(axes_arr)):
            axes_arr[j].set_visible(False)

        fig.suptitle(
            "Boxplots of Numerical Features",
            fontsize=self.config.title_fontsize,
            fontweight="bold",
            y=1.02,
        )
        fig.tight_layout()
        self._save(
            fig,
            "boxplots",
            result,
            "comparison",
            "Boxplots",
            "Boxplots showing spread, median, and outliers for each numerical feature.",
        )

    def _violin_plots(
        self,
        df: pd.DataFrame,
        profile: DatasetProfile,
        result: VisualizationResult,
    ) -> None:
        """Generate violin plots for numeric columns."""
        num_cols = profile.numerical_columns
        if not num_cols:
            return
        n = len(num_cols)
        nrows, ncols = self._grid_layout(n)
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(self.config.figure_width, self.config.figure_height * nrows / 2),
        )
        axes_arr = np.array(axes).flatten() if n > 1 else [axes]
        palette = sns.color_palette(self.config.color_palette, n)

        for i, col in enumerate(num_cols):
            ax = axes_arr[i]
            sns.violinplot(data=df, y=col, ax=ax, color=palette[i], inner="quartile")
            ax.set_title(col, fontsize=self.config.label_fontsize, fontweight="bold")

        for j in range(n, len(axes_arr)):
            axes_arr[j].set_visible(False)

        fig.suptitle(
            "Violin Plots", fontsize=self.config.title_fontsize, fontweight="bold", y=1.02
        )
        fig.tight_layout()
        self._save(
            fig,
            "violin_plots",
            result,
            "comparison",
            "Violin Plots",
            "Violin plots showing distribution shape and density for numerical features.",
        )

    def _countplots(
        self,
        df: pd.DataFrame,
        profile: DatasetProfile,
        result: VisualizationResult,
    ) -> None:
        """Generate countplots for categorical columns."""
        cat_cols = profile.categorical_columns
        if not cat_cols:
            return
        top_n = self.config.top_n_categories
        n = len(cat_cols)
        nrows, ncols = self._grid_layout(n)
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(self.config.figure_width, self.config.figure_height * nrows / 2),
        )
        axes_arr = np.array(axes).flatten() if n > 1 else [axes]

        for i, col in enumerate(cat_cols):
            ax = axes_arr[i]
            order = df[col].value_counts().head(top_n).index
            sns.countplot(
                data=df,
                y=col,
                ax=ax,
                order=order,
                hue=col,
                palette=self.config.color_palette,
                legend=False,
            )  # type: ignore[arg-type]
            ax.set_title(col, fontsize=self.config.label_fontsize, fontweight="bold")
            ax.set_xlabel("Count")

        for j in range(n, len(axes_arr)):
            axes_arr[j].set_visible(False)

        fig.suptitle(
            "Categorical Feature Counts",
            fontsize=self.config.title_fontsize,
            fontweight="bold",
            y=1.02,
        )
        fig.tight_layout()
        self._save(
            fig,
            "countplots",
            result,
            "comparison",
            "Countplots",
            f"Top {top_n} value counts for each categorical feature.",
        )

    # ------------------------------------------------------------------
    # Relationship plots
    # ------------------------------------------------------------------

    def _scatterplots(
        self,
        df: pd.DataFrame,
        profile: DatasetProfile,
        result: VisualizationResult,
    ) -> None:
        """Generate scatterplots for the strongest correlated numeric pairs."""
        num_cols = profile.numerical_columns
        if len(num_cols) < 2:
            return
        sub = df[num_cols].dropna()
        if len(sub) < 3:
            return
        corr = sub.corr(method="pearson")
        pairs: list[tuple[str, str, float]] = []
        for i, ca in enumerate(num_cols):
            for j, cb in enumerate(num_cols):
                if j > i:
                    pairs.append((ca, cb, abs(float(corr.loc[ca, cb]))))  # type: ignore[arg-type]
        pairs.sort(key=lambda x: x[2], reverse=True)
        top_pairs = pairs[: self.config.max_scatter_pairs]

        if not top_pairs:
            return
        n = len(top_pairs)
        ncols = min(n, 2)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(self.config.figure_width, self.config.figure_height * nrows / 2),
        )
        axes_arr = np.array(axes).flatten() if n > 1 else [axes]

        for idx, (ca, cb, _) in enumerate(top_pairs):
            ax = axes_arr[idx]
            sns.scatterplot(data=df, x=ca, y=cb, ax=ax, alpha=0.6, s=30)
            ax.set_title(f"{ca} vs {cb}", fontsize=self.config.label_fontsize, fontweight="bold")

        for j in range(n, len(axes_arr)):
            axes_arr[j].set_visible(False)

        fig.suptitle(
            "Top Correlated Scatterplots",
            fontsize=self.config.title_fontsize,
            fontweight="bold",
            y=1.02,
        )
        fig.tight_layout()
        self._save(
            fig,
            "scatterplots",
            result,
            "relationship",
            "Scatterplots",
            "Scatterplots for the most strongly correlated numeric feature pairs.",
        )

    def _pairplot(
        self,
        df: pd.DataFrame,
        profile: DatasetProfile,
        result: VisualizationResult,
    ) -> None:
        """Generate a pairplot for up to N numeric columns."""
        num_cols = profile.numerical_columns[: self.config.pairplot_max_columns]
        if len(num_cols) < 2:
            return
        try:
            g = sns.pairplot(
                df[num_cols].dropna(),
                corner=True,
                diag_kind="kde",
                plot_kws={"alpha": 0.5, "s": 20},
            )
            g.fig.suptitle(
                "Pairplot of Numerical Features",
                fontsize=self.config.title_fontsize,
                fontweight="bold",
                y=1.02,
            )
            if self.config.save_figures and self._fig_dir is not None:
                path = self._fig_dir / "pairplot.png"
                g.savefig(path, dpi=self.config.figure_dpi, bbox_inches="tight")
            plt.close(g.fig)
            result.relationship.append(
                FigureResult(
                    title="Pairplot",
                    category="relationship",
                    description=f"Pairwise scatterplots and KDE for the first {len(num_cols)} numerical features.",
                    path=self._fig_dir / "pairplot.png"
                    if self.config.save_figures and self._fig_dir
                    else None,
                    filename="pairplot.png",
                )
            )
        except Exception as exc:
            logger.warning("Pairplot failed: %s", exc)

    def _bubble_charts(
        self,
        df: pd.DataFrame,
        profile: DatasetProfile,
        result: VisualizationResult,
    ) -> None:
        """Generate bubble charts for numeric triples (x, y, size)."""
        num_cols = profile.numerical_columns
        if len(num_cols) < 3:
            return
        sub = df[num_cols].dropna()
        if len(sub) < 10:
            return
        ca, cb, cc = num_cols[0], num_cols[1], num_cols[2]
        fig, ax = plt.subplots(figsize=(self.config.figure_width, self.config.figure_height))
        sizes = sub[cc]
        sizes_norm = (sizes - sizes.min()) / (sizes.max() - sizes.min() + 1e-9) * 500 + 20
        ax.scatter(
            sub[ca], sub[cb], s=sizes_norm, alpha=0.5, c=sub[cc], cmap=self.config.color_palette
        )
        ax.set_xlabel(ca)
        ax.set_ylabel(cb)
        ax.set_title(
            f"Bubble Chart: {ca} vs {cb} (size={cc})",
            fontsize=self.config.title_fontsize,
            fontweight="bold",
        )
        cbar = fig.colorbar(ax.collections[0], ax=ax)
        cbar.set_label(cc)
        fig.tight_layout()
        self._save(
            fig,
            "bubble_chart",
            result,
            "relationship",
            "Bubble Chart",
            f"Bubble chart with {ca} on x-axis, {cb} on y-axis, and {cc} as bubble size.",
        )

    # ------------------------------------------------------------------
    # Matrix plots
    # ------------------------------------------------------------------

    def _correlation_heatmap(
        self,
        stats: StatisticalAnalysis,
        result: VisualizationResult,
    ) -> None:
        """Generate a correlation heatmap from the Pearson matrix."""
        matrix = stats.pearson.matrix
        if not matrix:
            return
        cols = list(matrix.keys())
        data = np.array([[matrix[a][b] for b in cols] for a in cols])

        fig, ax = plt.subplots(figsize=(self.config.figure_width, self.config.figure_height))
        sns.heatmap(
            data,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            center=0,
            vmin=-1,
            vmax=1,
            xticklabels=cols,
            yticklabels=cols,
            ax=ax,
            square=True,
            linewidths=0.5,
        )
        ax.set_title(
            "Pearson Correlation Heatmap", fontsize=self.config.title_fontsize, fontweight="bold"
        )
        fig.tight_layout()
        self._save(
            fig,
            "correlation_heatmap",
            result,
            "matrix",
            "Correlation Heatmap",
            "Pearson correlation matrix heatmap for all numerical features.",
        )

    # ------------------------------------------------------------------
    # Quality plots
    # ------------------------------------------------------------------

    def _missing_value_heatmap(
        self,
        df: pd.DataFrame,
        result: VisualizationResult,
    ) -> None:
        """Generate a missing-value heatmap."""
        if df.isna().sum().sum() == 0:
            return
        fig, ax = plt.subplots(figsize=(self.config.figure_width, self.config.figure_height))
        sns.heatmap(df.isna(), cbar=True, yticklabels=False, cmap="viridis", ax=ax)
        ax.set_title(
            "Missing Values Heatmap", fontsize=self.config.title_fontsize, fontweight="bold"
        )
        ax.set_xlabel("Columns")
        fig.tight_layout()
        self._save(
            fig,
            "missing_heatmap",
            result,
            "quality",
            "Missing Values Heatmap",
            "Yellow stripes indicate missing values in each column.",
        )

    def _outlier_visualisation(
        self,
        df: pd.DataFrame,
        profile: DatasetProfile,
        result: VisualizationResult,
    ) -> None:
        """Highlight outliers using a boxplot strip for numeric columns."""
        num_cols = profile.numerical_columns
        if not num_cols:
            return
        n = len(num_cols)
        nrows, ncols = self._grid_layout(n)
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(self.config.figure_width, self.config.figure_height * nrows / 2),
        )
        axes_arr = np.array(axes).flatten() if n > 1 else [axes]
        palette = sns.color_palette(self.config.color_palette, n)

        for i, col in enumerate(num_cols):
            ax = axes_arr[i]
            sns.boxplot(
                data=df,
                y=col,
                ax=ax,
                color=palette[i],
                flierprops={"marker": "o", "markersize": 4},
            )
            ax.set_title(
                f"{col} (outliers highlighted)",
                fontsize=self.config.label_fontsize,
                fontweight="bold",
            )

        for j in range(n, len(axes_arr)):
            axes_arr[j].set_visible(False)

        fig.suptitle(
            "Outlier Visualisation", fontsize=self.config.title_fontsize, fontweight="bold", y=1.02
        )
        fig.tight_layout()
        self._save(
            fig,
            "outlier_visualisation",
            result,
            "quality",
            "Outlier Visualisation",
            "Boxplots with outlier points highlighted for each numerical feature.",
        )

    # ------------------------------------------------------------------
    # Time-series plots
    # ------------------------------------------------------------------

    def _timeseries_plots(
        self,
        df: pd.DataFrame,
        profile: DatasetProfile,
        result: VisualizationResult,
    ) -> None:
        """Generate line plots for datetime-indexed numeric columns."""
        dt_cols = profile.datetime_columns
        num_cols = profile.numerical_columns
        if not dt_cols or not num_cols:
            return
        date_col = dt_cols[0]
        n = min(len(num_cols), self.config.timeseries_max_columns)
        ncols = min(n, 2)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(self.config.figure_width, self.config.figure_height * nrows / 2),
        )
        axes_arr = np.array(axes).flatten() if n > 1 else [axes]
        palette = sns.color_palette(self.config.color_palette, n)

        for i, col in enumerate(num_cols[:n]):
            ax = axes_arr[i]
            tmp = df[[date_col, col]].dropna().sort_values(date_col)
            ax.plot(tmp[date_col], tmp[col], color=palette[i], alpha=0.8, linewidth=1)
            ax.set_title(col, fontsize=self.config.label_fontsize, fontweight="bold")
            ax.tick_params(axis="x", rotation=45)

        for j in range(n, len(axes_arr)):
            axes_arr[j].set_visible(False)

        fig.suptitle(
            f"Time Series (indexed by {date_col})",
            fontsize=self.config.title_fontsize,
            fontweight="bold",
            y=1.02,
        )
        fig.tight_layout()
        self._save(
            fig,
            "timeseries",
            result,
            "timeseries",
            "Time Series Plots",
            f"Line plots of numerical features over {date_col}.",
        )
