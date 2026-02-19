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
