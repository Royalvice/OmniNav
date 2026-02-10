"""
OmniNav Evaluation Layer - Evaluation System

Provides abstract interfaces for task definitions and evaluation metrics,
plus inspection-specific implementations.
"""

from omninav.evaluation.base import TaskBase, MetricBase, TaskResult
from omninav.evaluation.tasks.inspection_task import InspectionTask
from omninav.evaluation.metrics.inspection_metrics import (
    CoverageRate, DetectionRate, InspectionTime, SafetyScore,
)

__all__ = [
    "TaskBase", "MetricBase", "TaskResult",
    "InspectionTask",
    "CoverageRate", "DetectionRate", "InspectionTime", "SafetyScore",
]
