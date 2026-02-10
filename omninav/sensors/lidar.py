"""
2D Lidar Sensor Implementation

Provides a 2D laser scanner using Genesis Lidar with SphericalPattern.
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


@SENSOR_REGISTRY.register("lidar_2d")
class Lidar2DSensor(SensorBase):
    """
    2D Lidar sensor using Genesis Lidar with SphericalPattern.

    Simulates a planar laser scanner by configuring the vertical FOV to be
    very small (< 1 degree). Outputs range data suitable for 2D SLAM and
    obstacle avoidance.

    Config example (configs/sensor/lidar_2d.yaml):
        type: lidar_2d
        horizontal_fov: 360.0
        num_rays: 720
        min_range: 0.1
        max_range: 30.0
        vertical_fov: 0.5  # Small value for 2D
    """

    SENSOR_TYPE = "lidar_2d"

    def __init__(self, cfg: DictConfig, scene: "gs.Scene", robot: "RobotBase"):
        """
        Initialize 2D Lidar sensor.

        Args:
            cfg: Sensor configuration
            scene: Genesis scene
            robot: Robot instance
        """
        super().__init__(cfg, scene, robot)

        # Lidar parameters from config
        self._horizontal_fov = cfg.get("horizontal_fov", 360.0)
        self._vertical_fov = cfg.get("vertical_fov", 0.5)  # Nearly flat for 2D
        self._num_rays = cfg.get("num_rays", 720)
        self._min_range = cfg.get("min_range", 0.1)
        self._max_range = cfg.get("max_range", 30.0)
        self._draw_debug = cfg.get("draw_debug", False)

    def create(self) -> None:
        """
        Create 2D Lidar in Genesis scene.

        Uses gs.sensors.Lidar with SphericalPattern configured for 2D scanning.
        Attaches to robot entity via entity_idx and link_idx_local.
        """
        import genesis as gs

        link = self._get_link()

        # Configure spherical pattern for 2D scanning
        # fov: (horizontal, vertical), n_points: (horizontal, vertical)
        pattern = gs.sensors.SphericalPattern(
            fov=(self._horizontal_fov, self._vertical_fov),
            n_points=(self._num_rays, 1),  # Single row for 2D
        )

        # Create Lidar sensor attached to robot link
        # Following Genesis sensors.md documentation pattern
        self._gs_sensor = self.scene.add_sensor(
            gs.sensors.Lidar(
                pattern=pattern,
                entity_idx=self.robot.entity.idx,
                link_idx_local=link.idx_local,
                pos_offset=tuple(self._pos_offset.tolist()),
                euler_offset=tuple(self._euler_offset.tolist()),
                min_range=self._min_range,
                max_range=self._max_range,
                return_world_frame=False,
                draw_debug=self._draw_debug,
            )
        )

        self._is_created = True

    def get_data(self) -> SensorData:
        """
        Read 2D Lidar data (Batch-First).

        Returns:
            SensorData with:
            - 'ranges': (B, N) range values
            - 'points': (B, N, 3) hit points in sensor frame
        """
        if not self.is_ready:
            return SensorData(
                ranges=np.zeros((1, self._num_rays), dtype=np.float32),
                points=np.zeros((1, self._num_rays, 3), dtype=np.float32),
            )

        # Read from Genesis sensor - returns NamedTuple with points, distances
        data = self._gs_sensor.read()

        # Extract hit positions → Batch-First (B, N, 3)
        hit_pos = _to_numpy_batch(data.points, target_shape_after_batch=(self._num_rays, 3))

        # Handle intermediate squeeze: (B, N, 1, 3) → (B, N, 3)
        if hit_pos.ndim == 4 and hit_pos.shape[2] == 1:
            hit_pos = hit_pos.squeeze(2)

        # Extract ranges → Batch-First (B, N)
        ranges = _to_numpy_batch(data.distances, target_shape_after_batch=(self._num_rays,))

        # Handle intermediate squeeze: (B, N, 1) → (B, N)
        if ranges.ndim == 3 and ranges.shape[2] == 1:
            ranges = ranges.squeeze(2)

        return SensorData(
            ranges=ranges.astype(np.float32),
            points=hit_pos.astype(np.float32),
        )

    @property
    def angle_min(self) -> float:
        """Minimum scan angle in radians."""
        return -np.deg2rad(self._horizontal_fov / 2)

    @property
    def angle_max(self) -> float:
        """Maximum scan angle in radians."""
        return np.deg2rad(self._horizontal_fov / 2)

    @property
    def angle_increment(self) -> float:
        """Angle increment between rays in radians."""
        return np.deg2rad(self._horizontal_fov) / self._num_rays
