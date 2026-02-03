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
        
        Refactored to use scene.add_camera() which shares the visualizer's context.
        This is more stable on Windows systems than the standalone Rasterizer sensor.
        """
        import genesis as gs

        link = self._get_link()

        # Compute camera pose from offset
        # Note: add_camera uses [pos, lookat, up]. 
        # We assume lookat is forward looking relative to camera pos.
        lookat_local = np.array([1.0, 0.0, 0.0])
        
        # We add the camera to the scene visualizer
        self._gs_sensor = self.scene.add_camera(
            res=(self._width, self._height),
            pos=tuple(self._pos_offset.tolist()),
            lookat=tuple((self._pos_offset + lookat_local).tolist()),
            up=(0.0, 0.0, 1.0),
            fov=self._fov,
            near=self._near,
            far=self._far,
            GUI=False,
        )
        
        # Attach the camera to the robot link
        # In Genesis, Camera.attach(link, offset_T)
        offset_T = gs.utils.geom.trans_quat_to_T(
            self._pos_offset, 
            gs.utils.geom.euler_to_quat(self._euler_offset)
        )
        self._gs_sensor.attach(link, offset_T)

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

        # Render and read from Genesis camera object
        # Camera.render() returns (rgb, depth, segmentation, normal)
        rgb, depth, _, _ = self._gs_sensor.render(
            rgb=("rgb" in self._camera_types),
            depth=("depth" in self._camera_types),
            segmentation=False,
            normal=False,
        )
        
        result = {}
        if rgb is not None:
            # Genesis might return torch tensor or numpy
            if hasattr(rgb, "cpu"):
                rgb = rgb.cpu().numpy()
            result["rgb"] = np.asarray(rgb, dtype=np.uint8)

        if depth is not None:
            if hasattr(depth, "cpu"):
                depth = depth.cpu().numpy()
            result["depth"] = np.asarray(depth, dtype=np.float32)

        return result

    @property
    def resolution(self) -> tuple:
        """Get camera resolution (width, height)."""
        return (self._width, self._height)
