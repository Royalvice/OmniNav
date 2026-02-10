"""
Sensor Abstract Base Class

Defines the interface for all sensor implementations in OmniNav.

Design principles:
1. get_data() returns SensorData TypedDict with Batch-First arrays
2. Sensor mounting uses MountInfo from core/types.py
3. Genesis API calls are confined to create() and get_data()
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, List, TYPE_CHECKING
import numpy as np
from omegaconf import DictConfig

from omninav.core.types import SensorData

if TYPE_CHECKING:
    import genesis as gs
    from omninav.robots.base import RobotBase


def _to_numpy_batch(x, target_shape_after_batch: tuple = None) -> np.ndarray:
    """
    Convert Genesis tensor to numpy with guaranteed Batch-First shape.

    Args:
        x: Genesis tensor or numpy array
        target_shape_after_batch: Expected shape after batch dim.
            If None, just ensures batch dim exists.

    Returns:
        Array with shape (B, ...)
    """
    if hasattr(x, 'cpu'):
        arr = x.cpu().numpy()
    else:
        arr = np.asarray(x)

    # If the array has no batch dim, add one
    if target_shape_after_batch is not None:
        ndim_expected = 1 + len(target_shape_after_batch)
        if arr.ndim < ndim_expected:
            arr = np.expand_dims(arr, 0)
    elif arr.ndim >= 1 and arr.ndim < 2:
        arr = np.expand_dims(arr, 0)

    return arr


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
        4. get_data: Read sensor data as SensorData TypedDict (Batch-First)
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
        (e.g., scene.add_sensor for Lidar, scene.add_camera for Camera).

        Must be called before scene.build().
        """
        pass

    @abstractmethod
    def get_data(self) -> SensorData:
        """
        Read sensor data (Batch-First).

        Returns:
            SensorData TypedDict with arrays shaped (B, ...).
            Keys depend on sensor type:
            - Lidar: 'ranges' (B, N), 'points' (B, N, 3)
            - Camera: 'rgb' (B, H, W, 3), 'depth' (B, H, W)
            - Raycaster: 'depth' (B, H, W)
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
