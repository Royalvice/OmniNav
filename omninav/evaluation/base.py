"""
Evaluation Task and Metric Abstract Base Classes

Defines the interface for tasks and evaluation metrics.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List
import numpy as np
from omegaconf import DictConfig


@dataclass
class TaskResult:
    """
    Task execution result.
    
    Attributes:
        success: Whether task succeeded
        metrics: Evaluation metric values
        info: Additional info (e.g., trajectory, step count)
    """
    success: bool
    metrics: Dict[str, float] = field(default_factory=dict)
    info: Dict[str, Any] = field(default_factory=dict)


class MetricBase(ABC):
    """
    Abstract base class for evaluation metrics.
    
    All evaluation metrics must inherit from this class and implement abstract methods.
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
    def update(self, **kwargs) -> None:
        """
        Update metric.
        
        Called after each simulation step with relevant data.
        
        Args:
            **kwargs: Data needed for update, e.g., robot_position, goal_position
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


class TaskBase(ABC):
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
    
    @abstractmethod
    def reset(self) -> Dict[str, Any]:
        """
        Reset task state.
        
        Returns:
            task_info: Task information passed to algorithm for initialization
                - Should include "start_position", "goal_position", etc.
        """
        pass
    
    @abstractmethod
    def step(self, robot_state: Any, action: np.ndarray) -> None:
        """
        Record information at each step.
        
        Used for computing cumulative metrics (e.g., path length, collision count).
        
        Args:
            robot_state: RobotState object
            action: Action taken at current step
        """
        pass
    
    @abstractmethod
    def is_terminated(self, robot_state: Any) -> bool:
        """
        Determine if task is terminated.
        
        May terminate due to success, failure, or timeout.
        
        Args:
            robot_state: RobotState object
            
        Returns:
            True if task should terminate, False otherwise
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
