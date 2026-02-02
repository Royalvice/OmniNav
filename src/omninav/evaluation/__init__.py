"""
OmniNav Evaluation Layer - 评测系统

提供任务定义和评价指标的抽象接口。
"""

from omninav.evaluation.base import TaskBase, MetricBase, TaskResult

__all__ = ["TaskBase", "MetricBase", "TaskResult"]
