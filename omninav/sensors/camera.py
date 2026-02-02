"""
Camera Sensor Implementation

Provides RGB and depth camera using Genesis Camera.
"""

from typing import Dict, Any, List, TYPE_CHECKING
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
    RGB-D Camera sensor using Genesis Camera.
    
    Can provide RGB, depth, segmentation, and normal images.
    
    Config example (configs/sensor/camera_rgb.yaml):
        type: camera
        width: 640
        height: 480
        fov: 60.0
        near: 0.1
        far: 100.0
        camera_types: ["rgb", "depth"]  # Which outputs to enable
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
        
        Uses gs.vis.camera.Camera attached to robot link.
        """
        import genesis as gs
        from genesis.utils.geom import euler_to_quat
        
        link = self._get_link()
        
        # Create camera object
        # Note: Genesis Camera is created via visualizer
        self._gs_sensor = gs.vis.camera.Camera(
            visualizer=self.scene.visualizer,
            model="pinhole",
            res=(self._width, self._height),
            fov=self._fov,
            near=self._near,
            far=self._far,
            up=(0.0, 0.0, 1.0),
            GUI=False,
            spp=1,  # Samples per pixel (1 for fast rendering)
            denoise=False,
        )
        
        # Build transform matrix for attachment
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = gs.utils.geom.quat_to_R(
            euler_to_quat(self._euler_offset)
        )
        T[:3, 3] = self._pos_offset
        
        # Attach camera to link
        self._gs_sensor.attach(link, T)
        
        # Register camera with visualizer
        self.scene._visualizer._cameras.append(self._gs_sensor)
        
        self._is_created = True
    
    def get_data(self) -> Dict[str, np.ndarray]:
        """
        Read camera data.
        
        Returns:
            Dictionary with enabled outputs:
            - 'rgb': RGB image array [H, W, 3] uint8
            - 'depth': Depth image array [H, W] float32
            - 'segmentation': Segmentation mask [H, W] int16
            - 'normal': Normal map [H, W, 3] float32
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
        
        # Move camera to attachment point
        self._gs_sensor.move_to_attach()
        
        # Render and collect requested outputs
        result = {}
        
        if "rgb" in self._camera_types:
            rendered = self._gs_sensor.render(rgb=True)
            rgb = rendered[0]  # First element is RGB
            if hasattr(rgb, "cpu"):
                rgb = rgb.cpu().numpy()
            result["rgb"] = np.asarray(rgb, dtype=np.uint8)
        
        if "depth" in self._camera_types:
            rendered = self._gs_sensor.render(depth=True)
            depth = rendered[1] if len(rendered) > 1 else rendered[0]
            if hasattr(depth, "cpu"):
                depth = depth.cpu().numpy()
            result["depth"] = np.asarray(depth, dtype=np.float32)
        
        if "segmentation" in self._camera_types:
            rendered = self._gs_sensor.render(segmentation=True)
            seg = rendered[2] if len(rendered) > 2 else rendered[0]
            if hasattr(seg, "cpu"):
                seg = seg.cpu().numpy()
            result["segmentation"] = np.asarray(seg, dtype=np.int16)
        
        if "normal" in self._camera_types:
            rendered = self._gs_sensor.render(normal=True)
            normal = rendered[3] if len(rendered) > 3 else rendered[0]
            if hasattr(normal, "cpu"):
                normal = normal.cpu().numpy()
            result["normal"] = np.asarray(normal, dtype=np.float32)
        
        return result
    
    @property
    def resolution(self) -> tuple:
        """Get camera resolution (width, height)."""
        return (self._width, self._height)
