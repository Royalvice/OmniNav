"""ROS2 data adaptation helpers for OmniNav."""

from __future__ import annotations

from typing import Optional

import numpy as np

from omninav.core.types import validate_batch_shape


class Ros2Adapter:
    """Utility conversion layer between ROS message payloads and Batch-First arrays."""

    @staticmethod
    def cmd_vel_from_twist(msg) -> np.ndarray:
        """Extract ``[vx, vy, wz]`` from a ``geometry_msgs/Twist`` message."""
        return np.array([msg.linear.x, msg.linear.y, msg.angular.z], dtype=np.float32)

    @staticmethod
    def normalize_cmd_vel_batch(cmd_vel: np.ndarray, name: str = "cmd_vel") -> np.ndarray:
        """Normalize velocity command to shape ``(B, 3)``."""
        cmd_vel = np.asarray(cmd_vel, dtype=np.float32)
        if cmd_vel.ndim == 1:
            if cmd_vel.shape[0] != 3:
                raise ValueError(f"{name}: expected 3 elements for 1D cmd_vel, got {cmd_vel.shape}")
            cmd_vel = cmd_vel.reshape(1, 3)
        validate_batch_shape(cmd_vel, f"{name}.cmd_vel", (3,))
        return cmd_vel

    @staticmethod
    def first_batch(cmd_vel: np.ndarray) -> np.ndarray:
        """Return first row as ``(3,)`` for single-message publishing paths."""
        batch = Ros2Adapter.normalize_cmd_vel_batch(cmd_vel)
        return batch[0]

    @staticmethod
    def pick_scan_data(sensors: dict) -> Optional[dict]:
        """Best-effort extraction of 2D lidar-like data from sensor dict."""
        for sensor_data in sensors.values():
            if isinstance(sensor_data, dict) and "ranges" in sensor_data:
                return sensor_data
        return None

    @staticmethod
    def enrich_scan_data(scan_data: dict) -> dict:
        """Fill LaserScan metadata when sensor payload omits ROS fields."""
        ranges = np.asarray(scan_data.get("ranges", np.array([], dtype=np.float32)), dtype=np.float32)
        if ranges.ndim == 2:
            ranges = ranges[0]
        if ranges.ndim != 1:
            ranges = ranges.reshape(-1)

        num_rays = int(ranges.shape[0])
        angle_min = float(scan_data.get("angle_min", -np.pi))
        angle_max = float(scan_data.get("angle_max", np.pi))

        if num_rays > 1:
            default_increment = (angle_max - angle_min) / float(num_rays)
        else:
            default_increment = 0.0

        return {
            "ranges": ranges,
            "angle_min": angle_min,
            "angle_max": angle_max,
            "angle_increment": float(scan_data.get("angle_increment", default_increment)),
            "time_increment": float(scan_data.get("time_increment", 0.0)),
            "scan_time": float(scan_data.get("scan_time", 0.0)),
            "range_min": float(scan_data.get("range_min", 0.1)),
            "range_max": float(scan_data.get("range_max", 30.0)),
        }

    @staticmethod
    def pick_camera_data(sensors: dict) -> Optional[dict]:
        """Best-effort extraction of camera data with rgb/depth keys."""
        for sensor_data in sensors.values():
            if not isinstance(sensor_data, dict):
                continue
            if ("rgb" in sensor_data) or ("depth" in sensor_data):
                return sensor_data
        return None

    @staticmethod
    def normalize_rgb_image(image: np.ndarray) -> np.ndarray:
        """Normalize RGB image to shape ``(H, W, 3)`` uint8."""
        arr = np.asarray(image)
        if arr.ndim == 4:
            arr = arr[0]
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"rgb image must be (H, W, 3), got {arr.shape}")
        return arr.astype(np.uint8, copy=False)

    @staticmethod
    def normalize_depth_image(image: np.ndarray) -> np.ndarray:
        """Normalize depth image to shape ``(H, W)`` float32."""
        arr = np.asarray(image)
        if arr.ndim == 3:
            arr = arr[0]
        if arr.ndim != 2:
            raise ValueError(f"depth image must be (H, W), got {arr.shape}")
        return arr.astype(np.float32, copy=False)
