"""
Algorithm Abstract Base Class

Defines the interface for pluggable algorithms.
Uses Observation TypedDict for standardized input.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, TYPE_CHECKING
import numpy as np
from omegaconf import DictConfig
from omninav.core.lifecycle import LifecycleMixin, LifecycleState

if TYPE_CHECKING:
    from omninav.core.types import Observation


class AlgorithmBase(ABC, LifecycleMixin):
    """
    Abstract base class for pluggable algorithms.

    All navigation, perception, and other algorithms must inherit from this class
    and implement abstract methods. Algorithms registered via ALGORITHM_REGISTRY
    can be selected through configuration files.

    Uses Observation TypedDict for standardized input data contract.
    """

    # Algorithm type identifier (for registration)
    ALGORITHM_TYPE: str = ""

    def __init__(self, cfg: DictConfig):
        """
        Initialize algorithm.

        Args:
            cfg: Algorithm configuration
        """
        self.cfg = cfg
        self._state = LifecycleState.CREATED

    @abstractmethod
    def reset(self, task_info: Optional[Dict[str, Any]] = None) -> None:
        """
        Reset algorithm state.

        Called when a new task starts, with optional task info for initialization.

        Args:
            task_info: Task information like start position, goal position, map, etc.
        """
        pass

    @abstractmethod
    def step(self, obs: "Observation") -> np.ndarray:
        """
        Compute action from observation.

        Args:
            obs: Observation TypedDict containing:
                - robot_state: RobotState (Batch-First)
                - sensors: dict of SensorData
                - sim_time: Current simulation time
                - goal_position: Optional target position

        Returns:
            cmd_vel: Batch-First velocity commands with shape (B, 3)
        """
        pass

    @property
    @abstractmethod
    def is_done(self) -> bool:
        """
        Whether algorithm considers task complete.

        Returns:
            True if algorithm determines task is complete, False otherwise
        """
        pass

    @property
    def info(self) -> Dict[str, Any]:
        """
        Return algorithm internal info.

        Used for debugging, visualization, or logging. Returns empty dict by default.

        Returns:
            Dictionary containing algorithm internal state
        """
        return {}
