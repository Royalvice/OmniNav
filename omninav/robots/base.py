"""
Robot and Sensor Abstract Base Classes

Defines the interface specifications for robots and sensors.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, TYPE_CHECKING
import numpy as np
from omegaconf import DictConfig

if TYPE_CHECKING:
    import genesis as gs


@dataclass
class RobotState:
    """
    Robot state data class.
    
    Attributes:
        position: [x, y, z] world coordinate position
        orientation: [qw, qx, qy, qz] quaternion orientation
        linear_velocity: [vx, vy, vz] linear velocity
        angular_velocity: [wx, wy, wz] angular velocity
        joint_positions: Joint position array
        joint_velocities: Joint velocity array
    """
    position: np.ndarray
    orientation: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    joint_positions: np.ndarray
    joint_velocities: np.ndarray


@dataclass
class SensorMount:
    """
    Sensor mount configuration.
    
    Attributes:
        sensor_name: Sensor config name (corresponds to configs/sensor/*.yaml)
        link_name: Name of the link to mount on
        position: Position relative to link [x, y, z]
        orientation: Orientation relative to link [qw, qx, qy, qz]
    """
    sensor_name: str
    link_name: str
    position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    orientation: List[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])


class SensorBase(ABC):
    """
    Abstract base class for sensors.
    
    All sensor implementations must inherit from this class and implement abstract methods.
    """
    
    # Sensor type identifier (for registration)
    SENSOR_TYPE: str = ""
    
    def __init__(self, cfg: DictConfig):
        """
        Initialize sensor.
        
        Args:
            cfg: Sensor configuration
        """
        self.cfg = cfg
        self._attached_robot: Optional["RobotBase"] = None
        self._gs_sensor: Optional[Any] = None  # Genesis sensor object
    
    @abstractmethod
    def create(self, scene: "gs.Scene") -> None:
        """
        Create sensor in the scene.
        
        Calls appropriate Genesis API based on sensor type (e.g., scene.add_camera).
        
        Args:
            scene: Genesis scene object
        """
        pass
    
    @abstractmethod
    def get_data(self) -> Any:
        """
        Get sensor data.
        
        Returns:
            Sensor data (type depends on sensor type)
        """
        pass
    
    def attach_to_robot(
        self, 
        robot: "RobotBase", 
        link_name: str,
        position: List[float],
        orientation: List[float]
    ) -> None:
        """
        Attach sensor to a specific robot link.
        
        Args:
            robot: Robot instance
            link_name: Name of the link to attach to
            position: Relative position [x, y, z]
            orientation: Relative orientation [qw, qx, qy, qz]
        """
        self._attached_robot = robot
        # Concrete implementations should call Genesis API to set sensor parent link


class RobotBase(ABC):
    """
    Abstract base class for robots.
    
    All robot implementations must inherit from this class and implement abstract methods.
    """
    
    # Robot type identifier (for registration)
    ROBOT_TYPE: str = ""
    
    def __init__(self, cfg: DictConfig, scene: "gs.Scene"):
        """
        Initialize robot.
        
        Args:
            cfg: Robot configuration
            scene: Genesis scene object
        """
        self.cfg = cfg
        self.scene = scene
        self.entity: Optional[Any] = None  # Genesis entity object
        self.sensors: Dict[str, SensorBase] = {}
        self._sensor_mounts: List[SensorMount] = []
        self._initial_pos: Optional[np.ndarray] = None
        self._initial_quat: Optional[np.ndarray] = None
    
    @abstractmethod
    def spawn(self) -> None:
        """
        Spawn robot in the scene.
        
        Loads robot model and adds to Genesis scene.
        """
        pass
    
    @abstractmethod
    def get_state(self) -> RobotState:
        """
        Get current robot state.
        
        Returns:
            RobotState: Contains position, orientation, velocity, joint state
        """
        pass
    
    @abstractmethod
    def apply_command(self, cmd_vel: np.ndarray) -> None:
        """
        Apply velocity command.
        
        High-level interface called by LocomotionController to control robot motion.
        
        Args:
            cmd_vel: [vx, vy, wz] linear velocity (m/s) + angular velocity (rad/s)
        """
        pass
    
    def mount_sensor(self, mount: SensorMount, sensor: SensorBase) -> None:
        """
        Mount sensor to robot.
        
        Args:
            mount: Mount configuration
            sensor: Sensor instance
        """
        sensor.attach_to_robot(
            self, 
            mount.link_name, 
            mount.position, 
            mount.orientation
        )
        self.sensors[mount.sensor_name] = sensor
        self._sensor_mounts.append(mount)
    
    def get_observations(self) -> Dict[str, Any]:
        """
        Get all sensor observations.
        
        Returns:
            Dictionary with sensor names as keys and sensor data as values
        """
        obs = {}
        for name, sensor in self.sensors.items():
            obs[name] = sensor.get_data()
        return obs
    
    def reset(self) -> None:
        """
        Reset robot to initial state.
        """
        if self.entity is not None and self._initial_pos is not None:
            self.entity.set_pos(self._initial_pos)
        if self.entity is not None and self._initial_quat is not None:
            self.entity.set_quat(self._initial_quat)
