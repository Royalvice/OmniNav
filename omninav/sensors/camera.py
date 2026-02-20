"""
Camera Sensor Implementation

Provides RGB and depth camera using Genesis RasterizerCameraOptions.
Returns SensorData TypedDict with Batch-First arrays.
"""

from typing import List, TYPE_CHECKING
import numpy as np
from omegaconf import DictConfig

from omninav.sensors.base import SensorBase, _to_numpy_batch
from omninav.core.types import SensorData
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
        self._update_every_n_steps = max(1, int(cfg.get("update_every_n_steps", 1)))
        self._cached_data: SensorData = {}
        self._last_render_step: int = -1

    def create(self) -> None:
        """
        Create camera in Genesis scene.
        """
        import genesis as gs

        link = self._get_link()
        # Create a neutral camera first and then attach it with mount offset.
        # This follows Genesis attached-camera usage and avoids ambiguous world pose.
        self._gs_sensor = self.scene.add_camera(
            res=(self._width, self._height),
            pos=(0.0, 0.0, 0.0),
            lookat=(1.0, 0.0, 0.0),
            up=(0.0, 0.0, 1.0),
            fov=self._fov,
            near=self._near,
            far=self._far,
            GUI=False,
        )

        offset_T = gs.utils.geom.trans_quat_to_T(
            self._pos_offset,
            gs.utils.geom.euler_to_quat(self._euler_offset)
        )
        self._gs_sensor.attach(link, offset_T)

        self._is_created = True

    def get_data(self) -> SensorData:
        """
        Read camera data (Batch-First).

        Returns:
            SensorData with enabled outputs:
            - 'rgb': (B, H, W, 3) uint8
            - 'depth': (B, H, W) float32
        """
        if not self.is_ready:
            result: SensorData = {}
            if "rgb" in self._camera_types:
                result["rgb"] = np.zeros(
                    (1, self._height, self._width, 3), dtype=np.uint8
                )
            if "depth" in self._camera_types:
                result["depth"] = np.zeros(
                    (1, self._height, self._width), dtype=np.float32
                )
            return result

        scene_step = int(getattr(self.scene, "t", 0))
        if (
            self._cached_data
            and self._update_every_n_steps > 1
            and scene_step > self._last_render_step
            and (scene_step % self._update_every_n_steps) != 0
        ):
            return self._cached_data

        # Render and read from Genesis camera object
        # Camera.render() returns (rgb, depth, segmentation, normal)
        rgb, depth, _, _ = self._gs_sensor.render(
            rgb=("rgb" in self._camera_types),
            depth=("depth" in self._camera_types),
            segmentation=False,
            normal=False,
        )
        
        result: SensorData = {}
        if rgb is not None:
            rgb_arr = _to_numpy_batch(rgb, target_shape_after_batch=(self._height, self._width, 3))
            result["rgb"] = rgb_arr.astype(np.uint8)

        if depth is not None:
            depth_arr = _to_numpy_batch(depth, target_shape_after_batch=(self._height, self._width))
            result["depth"] = depth_arr.astype(np.float32)

        self._cached_data = result
        self._last_render_step = scene_step
        return result

    @property
    def resolution(self) -> tuple:
        """Get camera resolution (width, height)."""
        return (self._width, self._height)
