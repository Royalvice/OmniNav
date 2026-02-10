"""
Locomotion Controller Abstract Base Class

Converts high-level cmd_vel commands to joint-level control.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, TYPE_CHECKING
import numpy as np
from omegaconf import DictConfig, OmegaConf

if TYPE_CHECKING:
    from omninav.robots.base import RobotBase
    from omninav.core.types import Observation
    from omninav.sensors.base import SensorBase


class LocomotionControllerBase(ABC):
    """
    Abstract base class for locomotion controllers.
    """

    CONTROLLER_TYPE: str = ""

    def __init__(self, cfg: DictConfig, robot: "RobotBase"):
        """
        Initialize locomotion controller.
        """
        self.cfg = cfg
        self.robot = robot
        self._sensors: Dict[str, "SensorBase"] = {}

    @property
    def required_sensors(self) -> dict:
        """
        Get dictionary of required sensors for this controller.
        
        Subclasses should override this to request specific sensors.
        Returns a dict mapping sensor_name to sensor_config.
        """
        return {}

    def bind_sensors(self, sensors: Dict[str, "SensorBase"]) -> None:
        """
        Bind instantiated sensors to the controller.
        
        Args:
            sensors: Dictionary of sensor instances available on the robot.
        """
        self._sensors = sensors

    @abstractmethod
    def reset(self) -> None:
        """Reset controller state."""
        pass

    @abstractmethod
    def compute_action(self, cmd_vel: np.ndarray) -> np.ndarray:
        """Compute joint action from velocity command."""
        pass

    @abstractmethod
    def step(self, cmd_vel: np.ndarray, obs: Optional["Observation"] = None) -> None:
        """Execute one locomotion control step."""
        pass
