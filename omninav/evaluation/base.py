"""
Evaluation Task and Metric Abstract Base Classes

Defines the interface for tasks and evaluation metrics.
Uses Observation TypedDict for standardized data input.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, TYPE_CHECKING
import numpy as np
from omegaconf import DictConfig
from omninav.core.lifecycle import LifecycleMixin, LifecycleState
from omninav.core.types import TaskResult

if TYPE_CHECKING:
    from omninav.core.types import Observation, Action

class MetricBase(ABC):
    """
    Abstract base class for evaluation metrics.

    All evaluation metrics must inherit from this class and implement abstract methods.
    Metrics are registered via METRIC_REGISTRY and instantiated from config.
    """

    # Metric name (used for registration and result dictionary keys)
    METRIC_NAME: str = ""

    def __init__(self):
        """Initialize metric."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Reset metric state.

        Called when a new task starts.
        """
        pass

    @abstractmethod
    def update(self, obs: "Observation", action: Optional["Action"] = None, **kwargs) -> None:
        """
        Update metric with current observation and action.

        Called after each simulation step.

        Args:
            obs: Current observation (Batch-First)
            action: Action taken at current step
            **kwargs: Additional data for custom metrics
        """
        pass

    @abstractmethod
    def compute(self) -> float:
        """
        Compute final metric value.

        Called when task ends.

        Returns:
            Metric value
        """
        pass


class TaskBase(ABC, LifecycleMixin):
    """
    Abstract base class for evaluation tasks.

    All evaluation tasks must inherit from this class and implement abstract methods.
    Tasks define evaluation objectives, termination conditions, and metrics.
    """

    # Task type identifier (for registration)
    TASK_TYPE: str = ""

    def __init__(self, cfg: DictConfig):
        """
        Initialize task.

        Args:
            cfg: Task configuration
        """
        self.cfg = cfg
        self.metrics: List[MetricBase] = []
        self._state = LifecycleState.CREATED

    @abstractmethod
    def reset(self) -> Dict[str, Any]:
        """
        Reset task state.

        Returns:
            task_info: Task information passed to algorithm for initialization
                - Should include "start_position", "goal_position", "waypoints", etc.
        """
        pass

    @abstractmethod
    def step(self, obs: "Observation", action: Optional["Action"] = None) -> None:
        """
        Record information at each step.

        Used for computing cumulative metrics (e.g., path length, collision count).

        Args:
            obs: Current Observation TypedDict
            action: Action taken at current step
        """
        pass

    @abstractmethod
    def is_terminated(self, obs: "Observation") -> np.ndarray:
        """
        Determine if task is terminated.

        May terminate due to success, failure, or timeout.

        Args:
            obs: Current Observation TypedDict

        Returns:
            Batch termination mask. Shape: (B,)
        """
        pass

    @abstractmethod
    def compute_result(self) -> TaskResult:
        """
        Compute final task result.

        Aggregates all metrics to compute final result.

        Returns:
            TaskResult: Contains success flag, metric values, and additional info
        """
        pass
