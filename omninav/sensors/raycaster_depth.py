"""
Raycaster-based Depth Sensor (Depth Camera via Raycasting)

Returns SensorData TypedDict with Batch-First arrays.
"""

from typing import TYPE_CHECKING
import numpy as np
from omegaconf import DictConfig

from omninav.sensors.base import SensorBase, _to_numpy_batch
from omninav.core.types import SensorData
from omninav.core.registry import SENSOR_REGISTRY

if TYPE_CHECKING:
    import genesis as gs
    from omninav.robots.base import RobotBase

@SENSOR_REGISTRY.register("raycaster_depth")
class RaycasterDepthSensor(SensorBase):
    """
    Raycaster-based Depth Sensor using Genesis DepthCamera.
    Outputs a depth image using raycasting (not rasterization).
    Returns Batch-First SensorData.
    """

    SENSOR_TYPE = "raycaster_depth"

    def __init__(self, cfg: DictConfig, scene: "gs.Scene", robot: "RobotBase"):
        super().__init__(cfg, scene, robot)
        self._width = cfg.get("width", 320)
        self._height = cfg.get("height", 240)
        self._min_range = cfg.get("min_range", 0.0)
        self._max_range = cfg.get("max_range", 10.0)
        self._draw_debug = cfg.get("draw_debug", False)

    def create(self) -> None:
        import genesis as gs
        link = self._get_link()

        # Create Depth Camera Sensor using Raycasting pattern
        self._gs_sensor = self.scene.add_sensor(
            gs.sensors.DepthCamera(
                pattern=gs.sensors.DepthCameraPattern(
                    res=(self._width, self._height),
                ),
                entity_idx=self.robot.entity.idx,
                link_idx_local=link.idx_local,
                pos_offset=tuple(self._pos_offset.tolist()),
                euler_offset=tuple(self._euler_offset.tolist()),
                min_range=self._min_range,
                max_range=self._max_range,
                draw_debug=self._draw_debug,
                return_world_frame=True,
            )
        )

        self._is_created = True

    def get_data(self) -> SensorData:
        """
        Read depth data (Batch-First).

        Returns:
            SensorData with 'depth': (B, H, W) float32
        """
        if not self.is_ready:
            return SensorData(
                depth=np.zeros((1, self._height, self._width), dtype=np.float32)
            )

        # Genesis DepthCameraSensor provides read_image()
        depth = self._gs_sensor.read_image()
        depth_arr = _to_numpy_batch(depth, target_shape_after_batch=(self._height, self._width))

        return SensorData(depth=depth_arr.astype(np.float32))
