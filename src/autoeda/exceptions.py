"""Custom exception hierarchy for AutoEDA."""

from __future__ import annotations


class AutoEDAError(Exception):
    """Base exception for all AutoEDA errors."""


class EmptyDataFrameError(AutoEDAError):
    """Raised when an empty DataFrame is provided."""

    def __init__(self, message: str = "DataFrame is empty") -> None:
        super().__init__(message)


class InvalidConfigError(AutoEDAError):
    """Raised when an invalid configuration value is provided."""


class ReportGenerationError(AutoEDAError):
    """Raised when report generation fails."""


class VisualizationError(AutoEDAError):
    """Raised when a visualization cannot be created."""


class AnalysisError(AutoEDAError):
    """Raised when a statistical analysis fails."""
