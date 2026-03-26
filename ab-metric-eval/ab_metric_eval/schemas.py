"""
Data schemas for experiment evaluation workflows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ExperimentConfig:
    """Configuration for an experiment analysis."""
    
    expt_name: str
    enrollment_date_start: str
    enrollment_date_end_incl: str
    analysis_date_start: str
    analysis_date_end_incl: str
    unit_type: str
    
    def to_dict(self) -> Dict[str, str]:
        """Convert config to dictionary."""
        return {
            "expt_name": self.expt_name,
            "enrollment_date_start": self.enrollment_date_start,
            "enrollment_date_end_incl": self.enrollment_date_end_incl,
            "analysis_date_start": self.analysis_date_start,
            "analysis_date_end_incl": self.analysis_date_end_incl,
            "unit_type": self.unit_type,
        }


@dataclass
class MetricDefinition:
    """Definition of a metric to be computed in aggregation."""
    
    name: str
    sql_expression: str
    description: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "sql_expression": self.sql_expression,
            "description": self.description,
        }
