"""
Robot Abstract Base Class

Defines the interface specifications for robots in OmniNav.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, TYPE_CHECKING
import numpy as np
from omegaconf import DictConfig

from omninav.core.types import RobotState, JointInfo, MountInfo, SensorData
from omninav.core.lifecycle import LifecycleMixin, LifecycleState

if TYPE_CHECKING:
    import genesis as gs
    from omninav.sensors.base import SensorBase


class RobotBase(LifecycleMixin, ABC):
    """
    Abstract base class for robots.
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
        self._state = LifecycleState.CREATED
        self.cfg = cfg
        self.scene = scene
        self.entity: Optional[Any] = None  # Genesis entity object
        self.sensors: Dict[str, "SensorBase"] = {}
        self._initial_pos: Optional[np.ndarray] = None
        self._initial_quat: Optional[np.ndarray] = None

    @abstractmethod
    def spawn(self) -> None:
        """Spawn robot in the scene."""
        pass

    @abstractmethod
    def get_state(self) -> RobotState:
        """Get current robot state (Batch-First)."""
        pass

    @abstractmethod
    def get_joint_info(self) -> JointInfo:
        """Get robot joint metadata (read-only descriptor)."""
        pass

    def control_joints_position(
        self, targets: np.ndarray, indices: np.ndarray
    ) -> None:
        """Apply joint position control targets."""
        self._require_state(LifecycleState.BUILT, "control_joints_position")
        self.entity.control_dofs_position(targets, indices)

    def control_joints_velocity(
        self, targets: np.ndarray, indices: np.ndarray
    ) -> None:
        """Apply joint velocity control targets."""
        self._require_state(LifecycleState.BUILT, "control_joints_velocity")
        self.entity.control_dofs_velocity(targets, indices)

    def post_build(self) -> None:
        """Called after scene.build()."""
        if self.lifecycle_state >= LifecycleState.BUILT:
            return
        self._transition_to(LifecycleState.BUILT)

    def mount_sensors(self, sensor_cfgs: List[DictConfig]) -> None:
        """
        Mount multiple sensors from configuration.
        """
        self._require_state(LifecycleState.SPAWNED, "mount_sensors")

        from omninav.core.registry import SENSOR_REGISTRY

        for sensor_cfg in sensor_cfgs:
            sensor = SENSOR_REGISTRY.build(sensor_cfg, scene=self.scene, robot=self)
            
            sensor_name = sensor_cfg.get("name", sensor.SENSOR_TYPE)
            self.mount_sensor(sensor_name, sensor)

        self._transition_to(LifecycleState.SENSORS_MOUNTED)

    def mount_sensor(self, name: str, sensor: "SensorBase") -> None:
        """
        Mount a single pre-built sensor.
        
        Args:
            name: Sensor name
            sensor: Sensor instance
        """
        # Determine attachment point from sensor config (stored in sensor)
        # However, SensorBase stores cfg. 
        # But we need to call attach() and create().
        
        # Note: attach() expects link_name, pos, orientation.
        # We need to extract these from sensor.cfg if available.
        cfg = sensor.cfg
        link_name = cfg.get("link_name", "base_link")
        
        # Convert ListConfig to list if needed
        pos = cfg.get("position", [0.0, 0.0, 0.0])
        if hasattr(pos, "tolist"): pos = pos.tolist()
        elif hasattr(pos, "__iter__"): pos = list(pos)
            
        ori = cfg.get("orientation", [0.0, 0.0, 0.0])
        if hasattr(ori, "tolist"): ori = ori.tolist()
        elif hasattr(ori, "__iter__"): ori = list(ori)

        sensor.attach(link_name, pos, ori)
        sensor.create()
        self.sensors[name] = sensor

    def get_observations(self) -> Dict[str, SensorData]:
        """Get all sensor observations."""
        obs: Dict[str, SensorData] = {}
        for name, sensor in self.sensors.items():
            obs[name] = sensor.get_data()
        return obs

    def get_mount_info(self, link_name: str, position: list, orientation: list) -> MountInfo:
        """Create a MountInfo descriptor."""
        link = self.entity.get_link(link_name) if self.entity else None
        return MountInfo(
            link_handle=link,
            position=np.array(position, dtype=np.float32),
            orientation=np.array(orientation, dtype=np.float32),
            scene_handle=self.scene,
        )

    def reset(self) -> None:
        """Reset robot to initial state."""
        self._require_state(LifecycleState.BUILT, "reset")

        if self.entity is None:
            return

        if self._initial_pos is not None:
            self.entity.set_pos(self._initial_pos)
        if self._initial_quat is not None:
            self.entity.set_quat(self._initial_quat)

        self._transition_to(LifecycleState.READY)
