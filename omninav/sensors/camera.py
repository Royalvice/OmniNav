"""
Camera Sensor Implementation

Provides RGB and depth camera using Genesis RasterizerCameraOptions.
"""

from typing import Dict, List, TYPE_CHECKING
import numpy as np
from omegaconf import DictConfig

from omninav.sensors.base import SensorBase
from omninav.core.registry import SENSOR_REGISTRY

if TYPE_CHECKING:
    import genesis as gs
    from omninav.robots.base import RobotBase


@SENSOR_REGISTRY.register("camera")
class CameraSensor(SensorBase):
    """
    RGB-D Camera sensor using Genesis RasterizerCameraOptions.

    Can provide RGB and depth images. Uses Rasterizer backend for fast
    real-time rendering (suitable for RL/navigation).

    Config example (configs/sensor/camera.yaml):
        type: camera
        width: 640
        height: 480
        fov: 60.0
        near: 0.1
        far: 100.0
        camera_types: ["rgb", "depth"]
    """

    SENSOR_TYPE = "camera"

    def __init__(self, cfg: DictConfig, scene: "gs.Scene", robot: "RobotBase"):
        """
        Initialize camera sensor.

        Args:
            cfg: Sensor configuration
            scene: Genesis scene
            robot: Robot instance
        """
        super().__init__(cfg, scene, robot)

        # Camera parameters from config
        self._width = cfg.get("width", 640)
        self._height = cfg.get("height", 480)
        self._fov = cfg.get("fov", 60.0)
        self._near = cfg.get("near", 0.1)
        self._far = cfg.get("far", 100.0)
        self._camera_types: List[str] = list(cfg.get("camera_types", ["rgb"]))

    def create(self) -> None:
        """
        Create camera in Genesis scene.

        Uses gs.sensors.RasterizerCameraOptions for fast rasterized rendering.
        Attaches to robot entity via entity_idx and link_idx_local.
        """
        import genesis as gs

        link = self._get_link()

        # Compute lookat direction from euler offset
        # Default: camera looks forward (+X in robot frame)
        # This creates a simple forward-facing camera
        lookat_local = np.array([1.0, 0.0, 0.0])  # Look forward

        # Create camera sensor using RasterizerCameraOptions
        # Following Genesis camera_sensors.md documentation
        self._gs_sensor = self.scene.add_sensor(
            gs.sensors.RasterizerCameraOptions(
                res=(self._width, self._height),
                pos=tuple(self._pos_offset.tolist()),
                lookat=tuple((self._pos_offset + lookat_local).tolist()),
                up=(0.0, 0.0, 1.0),
                fov=self._fov,
                near=self._near,
                far=self._far,
                entity_idx=self.robot.entity.idx,
                link_idx_local=link.idx_local,
            )
        )

        self._is_created = True

    def get_data(self) -> Dict[str, np.ndarray]:
        """
        Read camera data.

        Returns:
            Dictionary with enabled outputs:
            - 'rgb': RGB image array [H, W, 3] uint8
            - 'depth': Depth image array [H, W] float32
        """
        if not self.is_ready:
            result = {}
            if "rgb" in self._camera_types:
                result["rgb"] = np.zeros(
                    (self._height, self._width, 3), dtype=np.uint8
                )
            if "depth" in self._camera_types:
                result["depth"] = np.zeros(
                    (self._height, self._width), dtype=np.float32
                )
            return result

        # Read from Genesis camera sensor
        data = self._gs_sensor.read()
        result = {}

        # Process RGB data
        if "rgb" in self._camera_types:
            rgb = data.rgb
            if hasattr(rgb, "cpu"):
                rgb = rgb.cpu().numpy()
            else:
                rgb = np.array(rgb)

            # Handle batch dimension: (n_envs, H, W, 3) -> (H, W, 3)
            if rgb.ndim == 4:
                rgb = rgb[0]

            result["rgb"] = np.asarray(rgb, dtype=np.uint8)

        # Process depth data
        if "depth" in self._camera_types:
            depth = data.depth
            if hasattr(depth, "cpu"):
                depth = depth.cpu().numpy()
            else:
                depth = np.array(depth)

            # Handle batch dimension: (n_envs, H, W) -> (H, W)
            if depth.ndim == 3:
                depth = depth[0]

            result["depth"] = np.asarray(depth, dtype=np.float32)

        return result

    @property
    def resolution(self) -> tuple:
        """Get camera resolution (width, height)."""
        return (self._width, self._height)
