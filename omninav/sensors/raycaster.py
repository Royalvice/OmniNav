"""
Generic Raycaster Sensor

Wraps Genesis Raycaster with GridPattern.
Returns SensorData with 'ranges' (B, N) and optionally 'points' (B, N, 3).
"""

from typing import TYPE_CHECKING, Tuple, Optional
import numpy as np
from omegaconf import DictConfig

from omninav.sensors.base import SensorBase, _to_numpy_batch
from omninav.core.types import SensorData
from omninav.core.registry import SENSOR_REGISTRY

if TYPE_CHECKING:
    import genesis as gs
    from omninav.robots.base import RobotBase


@SENSOR_REGISTRY.register("raycaster")
class RaycasterSensor(SensorBase):
    """
    Generic Raycaster using Genesis GridPattern.
    
    Config:
        resolution: float (grid spacing)
        size: [w, h] (grid dimensions)
        direction: [x, y, z] (ray direction)
        return_points: bool (whether to return 3D contact points)
    
    Returns:
        SensorData:
            ranges: (B, N) distance to hit
            points: (B, N, 3) contact points (optional)
    """

    SENSOR_TYPE = "raycaster"

    def __init__(self, cfg: DictConfig, scene: "gs.Scene", robot: "RobotBase"):
        super().__init__(cfg, scene, robot)
        self._resolution = cfg.get("resolution", 0.1)
        self._size = tuple(cfg.get("size", [0.0, 0.0]))
        self._direction = tuple(cfg.get("direction", (0.0, 0.0, -1.0)))
        self._return_points = cfg.get("return_points", False)
        self._draw_debug = cfg.get("draw_debug", False)

    def create(self) -> None:
        import genesis as gs
        link = self._get_link()

        # Create Raycaster with GridPattern
        self._gs_sensor = self.scene.add_sensor(
            gs.sensors.Raycaster(
                pattern=gs.sensors.GridPattern(
                    resolution=self._resolution,
                    size=self._size,
                    direction=self._direction,
                ),
                entity_idx=self.robot.entity.idx,
                link_idx_local=link.idx_local,
                pos_offset=tuple(self._pos_offset.tolist()),
                euler_offset=tuple(self._euler_offset.tolist()),
                draw_debug=self._draw_debug,
                return_world_frame=True,
            )
        )

        self._is_created = True

    def get_data(self) -> SensorData:
        """
        Read raycaster data.

        Returns:
            SensorData:
                ranges: (B, N) float32
                points: (B, N, 3) float32 (if configured)
        """
        if not self.is_ready:
            return SensorData(ranges=np.zeros((1, 0), dtype=np.float32))

        # Read directly from genesis sensor
        # .read() returns named tuple/dataclass with .distances (N,) and .points (N, 3)
        raw = self._gs_sensor.read()
        
        # Process ranges
        dists = getattr(raw, "distances", np.array([]))
        ranges = _to_numpy_batch(dists, target_shape_after_batch=(-1,))
        
        data = SensorData(ranges=ranges.astype(np.float32))

        if self._return_points:
            pts = getattr(raw, "points", np.zeros((0, 3)))
            points = _to_numpy_batch(pts, target_shape_after_batch=(-1, 3))
            data["points"] = points.astype(np.float32)

        return data
