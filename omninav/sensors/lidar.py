"""
2D Lidar Sensor Implementation

Provides a 2D laser scanner using Genesis Lidar with flat spherical pattern.
"""

from typing import Dict, Any, TYPE_CHECKING
import numpy as np
from omegaconf import DictConfig

from omninav.sensors.base import SensorBase
from omninav.core.registry import SENSOR_REGISTRY

if TYPE_CHECKING:
    import genesis as gs
    from omninav.robots.base import RobotBase


@SENSOR_REGISTRY.register("lidar_2d")
class Lidar2DSensor(SensorBase):
    """
    2D Lidar sensor using Genesis Lidar with flat spherical pattern.
    
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
        """
        import genesis as gs
        
        link = self._get_link()
        
        # Configure spherical pattern for 2D scanning
        # fov: (horizontal, vertical), n_points: (horizontal, vertical)
        pattern = gs.sensors.SphericalPattern(
            fov=(self._horizontal_fov, self._vertical_fov),
            n_points=(self._num_rays, 1),  # Single row for 2D
        )
        
        # Create Lidar sensor
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
    
    def get_data(self) -> Dict[str, np.ndarray]:
        """
        Read 2D Lidar data.
        
        Returns:
            Dictionary with:
            - 'ranges': 1D array of range values [num_rays]
            - 'points': 2D array of hit points in sensor frame [num_rays, 3]
            - 'intensities': 1D array of intensity values (if available)
        """
        if not self.is_ready:
            return {
                "ranges": np.zeros(self._num_rays, dtype=np.float32),
                "points": np.zeros((self._num_rays, 3), dtype=np.float32),
            }
        
        # Read from Genesis sensor
        data = self._gs_sensor.read()
        
        # Extract hit points and compute ranges
        # Genesis Lidar returns hit_pos in shape (n_envs, n_rays, 3) or (n_rays, 3)
        hit_pos = data.hit_pos
        if hit_pos.ndim == 3:
            hit_pos = hit_pos[0]  # Take first environment
        
        # Flatten from (num_rays, 1, 3) to (num_rays, 3) if needed
        if hit_pos.ndim == 3 and hit_pos.shape[1] == 1:
            hit_pos = hit_pos.squeeze(1)
        
        # Compute ranges from hit positions
        ranges = np.linalg.norm(hit_pos, axis=-1).astype(np.float32)
        
        # Replace invalid hits (beyond max range) with max_range
        ranges = np.clip(ranges, self._min_range, self._max_range)
        
        return {
            "ranges": ranges.flatten(),
            "points": hit_pos.reshape(-1, 3).astype(np.float32),
        }
    
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
