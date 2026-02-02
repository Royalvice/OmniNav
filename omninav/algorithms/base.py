"""
Algorithm Abstract Base Class

Defines the interface for pluggable algorithms.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np
from omegaconf import DictConfig


class AlgorithmBase(ABC):
    """
    Abstract base class for pluggable algorithms.
    
    All navigation, perception, and other algorithms must inherit from this class
    and implement abstract methods. Algorithms registered via ALGORITHM_REGISTRY
    can be selected through configuration files.
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
    
    @abstractmethod
    def reset(self, task_info: Dict[str, Any]) -> None:
        """
        Reset algorithm state.
        
        Called when a new task starts, with task info for initialization.
        
        Args:
            task_info: Task information like start position, goal position, map, etc.
        """
        pass
    
    @abstractmethod
    def step(self, observation: Dict[str, Any]) -> np.ndarray:
        """
        Compute action from observation.
        
        Args:
            observation: Sensor observations + robot state
                - Sensor data (e.g., "depth_camera", "lidar_2d")
                - "robot_state": RobotState object
                - "sim_time": Current simulation time
        
        Returns:
            cmd_vel: [vx, vy, wz] velocity command
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
