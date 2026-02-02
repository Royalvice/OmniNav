"""
OmniNav Evaluation Layer - Evaluation System

Provides abstract interfaces for task definitions and evaluation metrics.
"""

from omninav.evaluation.base import TaskBase, MetricBase, TaskResult

__all__ = ["TaskBase", "MetricBase", "TaskResult"]
