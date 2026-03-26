"""
ab-metric-eval: Statistical analysis tools for A/B experiment metric evaluation.

Provides functions for summarising experiment groups and running
hypothesis tests (with optional CUPED variance reduction) on continuous
and binary metrics.
"""

from .analysis import (
    generate_experiment_summary,
    analyze_experiment_metric,
)
from .schemas import (
    ExperimentConfig,
    MetricDefinition,
)

__all__ = [
    "generate_experiment_summary",
    "analyze_experiment_metric",
    "ExperimentConfig",
    "MetricDefinition",
]

__version__ = "0.1.0"
