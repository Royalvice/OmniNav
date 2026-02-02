"""
Locomotion Controller Abstract Base Class

Converts high-level cmd_vel commands to joint-level control.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
import numpy as np
from omegaconf import DictConfig

if TYPE_CHECKING:
    from omninav.robots.base import RobotBase


class LocomotionControllerBase(ABC):
    """
    Abstract base class for locomotion controllers.
    
    Responsible for converting high-level velocity commands (cmd_vel) to 
    specific joint control signals, making the robot move at specified
    linear and angular velocities.
    """
    
    # Controller type identifier (for registration)
    CONTROLLER_TYPE: str = ""
    
    def __init__(self, cfg: DictConfig, robot: "RobotBase"):
        """
        Initialize locomotion controller.
        
        Args:
            cfg: Controller configuration
            robot: Robot instance to control
        """
        self.cfg = cfg
        self.robot = robot
    
    @abstractmethod
    def reset(self) -> None:
        """
        Reset controller state.
        
        Called when simulation resets or new episode starts.
        """
        pass
    
    @abstractmethod
    def compute_action(self, cmd_vel: np.ndarray) -> np.ndarray:
        """
        Compute joint action from velocity command.
        
        Args:
            cmd_vel: [vx, vy, wz] target linear velocity (m/s) and angular velocity (rad/s)
        
        Returns:
            Joint position/velocity/torque targets (depends on control mode)
        """
        pass
    
    @abstractmethod
    def step(self, cmd_vel: np.ndarray) -> None:
        """
        Execute one locomotion control step.
        
        Computes and applies joint control to the robot.
        
        Args:
            cmd_vel: [vx, vy, wz] target linear velocity (m/s) and angular velocity (rad/s)
        """
        pass
