"""
Robot Abstract Base Class

Defines the interface specifications for robots in OmniNav.
Sensors are now defined in omninav.sensors.base module.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, TYPE_CHECKING
import numpy as np
from omegaconf import DictConfig

if TYPE_CHECKING:
    import genesis as gs
    from omninav.sensors.base import SensorBase


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


class RobotBase(ABC):
    """
    Abstract base class for robots.

    All robot implementations must inherit from this class and implement
    abstract methods. Robots are registered via ROBOT_REGISTRY and can
    be dynamically instantiated from configuration.

    Lifecycle:
        1. __init__: Store configuration and scene reference
        2. spawn: Load robot model via scene.add_entity()
        3. mount_sensors: Create and attach sensors (before scene.build)
        4. [scene.build is called externally]
        5. get_state/get_observations: Read robot and sensor state
    """

    # Robot type identifier (for registration)
    ROBOT_TYPE: str = ""

    def __init__(self, cfg: DictConfig, scene: "gs.Scene"):
        """
        Initialize robot.

        Args:
            cfg: Robot configuration (from configs/robot/*.yaml)
            scene: Genesis scene object
        """
        self.cfg = cfg
        self.scene = scene
        self.entity: Optional[Any] = None  # Genesis entity object
        self.sensors: Dict[str, "SensorBase"] = {}
        self._initial_pos: Optional[np.ndarray] = None
        self._initial_quat: Optional[np.ndarray] = None

    @abstractmethod
    def spawn(self) -> None:
        """
        Spawn robot in the scene.

        Loads robot model (URDF/MJCF) and adds to Genesis scene via
        scene.add_entity(). After this call, self.entity.idx is available.
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

        High-level interface called by LocomotionController to control
        robot motion.

        Args:
            cmd_vel: [vx, vy, wz] linear velocity (m/s) + angular velocity (rad/s)
        """
        pass

    def post_build(self) -> None:
        """
        Called after scene.build().

        Override this method to perform robot-specific initialization that
        requires the scene to be built, such as:
        - Getting joint DOF indices
        - Setting PD control gains
        - Setting initial joint positions

        Must be called by SimulationManager after scene.build().
        """
        pass

    def mount_sensors(self, sensor_cfgs: List[DictConfig]) -> None:
        """
        Mount sensors from configuration after robot spawn.

        Creates sensors via SENSOR_REGISTRY, attaches to robot links,
        and calls sensor.create() to add to Genesis scene.

        Must be called after spawn() and before scene.build().

        Args:
            sensor_cfgs: List of sensor configurations, each containing:
                - type: Sensor type registered in SENSOR_REGISTRY
                - link_name: Robot link to attach to
                - position: [x, y, z] offset from link
                - orientation: [roll, pitch, yaw] euler angles offset
        """
        from omninav.core.registry import SENSOR_REGISTRY

        for sensor_cfg in sensor_cfgs:
            sensor_name = sensor_cfg.get("name", sensor_cfg.type)
            link_name = sensor_cfg.get("link_name", "base_link")
            position = list(sensor_cfg.get("position", [0.0, 0.0, 0.0]))
            orientation = list(sensor_cfg.get("orientation", [0.0, 0.0, 0.0]))

            # Create sensor via registry
            sensor = SENSOR_REGISTRY.build(sensor_cfg, scene=self.scene, robot=self)

            # Attach to robot link and create in scene
            sensor.attach(link_name, position, orientation)
            sensor.create()

            self.sensors[sensor_name] = sensor

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

        Resets position, orientation, and optionally joint states.
        """
        if self.entity is None:
            return

        if self._initial_pos is not None:
            self.entity.set_pos(self._initial_pos)
        if self._initial_quat is not None:
            self.entity.set_quat(self._initial_quat)
