"""
Sensor Abstract Base Class

Defines the interface for all sensor implementations in OmniNav.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, TYPE_CHECKING
import numpy as np
from omegaconf import DictConfig

if TYPE_CHECKING:
    import genesis as gs
    from omninav.robots.base import RobotBase


@dataclass
class SensorMount:
    """
    Sensor mount configuration.
    
    Attributes:
        sensor_name: Sensor config name (corresponds to configs/sensor/*.yaml)
        link_name: Name of the link to mount on
        position: Position relative to link [x, y, z]
        orientation: Orientation relative to link as euler angles [roll, pitch, yaw]
    """
    sensor_name: str
    link_name: str
    position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    orientation: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])


class SensorBase(ABC):
    """
    Abstract base class for sensors.
    
    All sensor implementations must inherit from this class and implement
    the abstract methods. Sensors are registered via SENSOR_REGISTRY
    and can be dynamically instantiated from configuration.
    
    Lifecycle:
        1. __init__: Store configuration
        2. attach: Attach to robot link
        3. create: Create Genesis sensor object (called before scene.build)
        4. get_data: Read sensor data (called after scene.step)
    """
    
    # Sensor type identifier (for registration)
    SENSOR_TYPE: str = ""
    
    def __init__(self, cfg: DictConfig, scene: "gs.Scene", robot: "RobotBase"):
        """
        Initialize sensor.
        
        Args:
            cfg: Sensor configuration from configs/sensor/*.yaml
            scene: Genesis scene object
            robot: Robot instance this sensor is attached to
        """
        self.cfg = cfg
        self.scene = scene
        self.robot = robot
        self._gs_sensor: Optional[Any] = None
        self._is_created: bool = False
        
        # Mount configuration (set by attach())
        self._link_name: Optional[str] = None
        self._pos_offset: np.ndarray = np.zeros(3)
        self._euler_offset: np.ndarray = np.zeros(3)
    
    def attach(
        self,
        link_name: str,
        position: List[float],
        orientation: List[float]
    ) -> None:
        """
        Set sensor attachment configuration.
        
        Args:
            link_name: Name of the robot link to attach to
            position: Position offset relative to link [x, y, z]
            orientation: Euler angle offset relative to link [roll, pitch, yaw]
        """
        self._link_name = link_name
        self._pos_offset = np.array(position, dtype=np.float32)
        self._euler_offset = np.array(orientation, dtype=np.float32)
    
    @abstractmethod
    def create(self) -> None:
        """
        Create sensor in the Genesis scene.
        
        This method should call the appropriate Genesis API to add the sensor
        (e.g., scene.add_sensor for Lidar, scene.visualizer.add_camera for Camera).
        
        Must be called before scene.build().
        """
        pass
    
    @abstractmethod
    def get_data(self) -> Dict[str, np.ndarray]:
        """
        Read sensor data.
        
        Returns:
            Dictionary containing sensor data. Keys depend on sensor type:
            - Lidar: {'ranges': np.ndarray, 'points': np.ndarray}
            - Camera: {'rgb': np.ndarray, 'depth': np.ndarray}
        """
        pass
    
    @property
    def is_ready(self) -> bool:
        """
        Check if sensor is ready to provide data.
        
        Returns:
            True if sensor is created and scene is built
        """
        return self._is_created and self.scene.is_built
    
    @property
    def gs_sensor(self) -> Any:
        """Get the underlying Genesis sensor object."""
        return self._gs_sensor
    
    def _get_link(self) -> Any:
        """
        Get the Genesis link object for attachment.
        
        Returns:
            Genesis link object
        
        Raises:
            ValueError: If link not found or attach() not called
        """
        if self._link_name is None:
            raise ValueError("Sensor not attached. Call attach() first.")
        
        link = self.robot.entity.get_link(self._link_name)
        if link is None:
            raise ValueError(
                f"Link '{self._link_name}' not found on robot. "
                f"Available links: {[l.name for l in self.robot.entity.links]}"
            )
        return link
