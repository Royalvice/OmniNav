"""
Tests for ROS2 Bridge (Mock-based).

Note: These tests don't require actual ROS2 installation.
They use mocks to verify the bridge logic.
"""

import numpy as np
from omegaconf import OmegaConf
from unittest.mock import MagicMock


class TestRos2BridgeDisabled:
    """Test ROS2 bridge when disabled."""
    
    def test_bridge_disabled_by_default(self):
        """Test bridge is disabled when enabled=false."""
        from omninav.interfaces.ros2.bridge import ROS2Bridge
        
        cfg = OmegaConf.create({
            "enabled": False,
        })
        
        mock_sim = MagicMock()
        bridge = ROS2Bridge(cfg, mock_sim)
        
        assert not bridge.enabled
        assert bridge._node is None
    
    def test_spin_once_noop_when_disabled(self):
        """Test spin_once does nothing when disabled."""
        from omninav.interfaces.ros2.bridge import ROS2Bridge
        
        cfg = OmegaConf.create({"enabled": False})
        mock_sim = MagicMock()
        bridge = ROS2Bridge(cfg, mock_sim)
        
        # Should not raise
        bridge.spin_once()
    
    def test_get_cmd_vel_returns_none_when_disabled(self):
        """Test get_cmd_vel returns None when disabled."""
        from omninav.interfaces.ros2.bridge import ROS2Bridge
        
        cfg = OmegaConf.create({"enabled": False})
        mock_sim = MagicMock()
        bridge = ROS2Bridge(cfg, mock_sim)
        
        assert bridge.get_external_cmd_vel() is None

    def test_python_source_ignores_external_cmd(self):
        """Control source python should not expose external command."""
        from omninav.interfaces.ros2.bridge import ROS2Bridge

        cfg = OmegaConf.create({"enabled": False, "control_source": "python"})
        bridge = ROS2Bridge(cfg, MagicMock())
        bridge._cmd_vel = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        assert bridge.get_external_cmd_vel() is None


class TestRos2BridgeLogic:
    """Test ROS2 bridge internal logic (with mocked rclpy)."""
    
    def test_cmd_vel_callback_stores_velocity(self):
        """Test cmd_vel callback stores velocity correctly."""
        from omninav.interfaces.ros2.bridge import ROS2Bridge
        
        cfg = OmegaConf.create({"enabled": False, "control_source": "ros2"})
        mock_sim = MagicMock()
        bridge = ROS2Bridge(cfg, mock_sim)
        
        # Create mock Twist message
        mock_msg = MagicMock()
        mock_msg.linear.x = 1.0
        mock_msg.linear.y = 0.5
        mock_msg.angular.z = 0.2
        
        bridge._cmd_vel_callback(mock_msg)
        
        cmd_vel = bridge.get_external_cmd_vel()
        assert cmd_vel is not None
        assert np.isclose(cmd_vel[0], 1.0)
        assert np.isclose(cmd_vel[1], 0.5)
        assert np.isclose(cmd_vel[2], 0.2)

    def test_cmd_timeout_returns_zero_velocity(self):
        """Stale external command should decay to zero for safety."""
        from omninav.interfaces.ros2.bridge import ROS2Bridge

        cfg = OmegaConf.create({"enabled": False, "control_source": "ros2", "cmd_vel_timeout_sec": 0.1})
        bridge = ROS2Bridge(cfg, MagicMock())
        bridge._cmd_vel = np.array([0.8, 0.1, 0.0], dtype=np.float32)
        bridge._last_cmd_vel_ts = 0.0

        # Force stale value by setting timeout check against very old timestamp.
        bridge._cmd_vel_fresh = lambda: False
        cmd_vel = bridge.get_external_cmd_vel()
        assert np.allclose(cmd_vel, np.zeros(3, dtype=np.float32))

    def test_sensor_mount_resolution_by_type(self):
        """Static TF mount lookup should not depend on hard-coded sensor names."""
        from omninav.interfaces.ros2.bridge import ROS2Bridge

        class _DummySensor:
            SENSOR_TYPE = "camera"

            def __init__(self, cfg):
                self.cfg = cfg

        cfg = OmegaConf.create({"enabled": False})
        bridge = ROS2Bridge(cfg, MagicMock())
        bridge._robot = MagicMock()
        bridge._robot.sensors = {
            "depth_camera": _DummySensor(
                {
                    "type": "camera",
                    "position": [0.45, 0.0, 0.2],
                    "orientation": [90.0, 0.0, -90.0],
                }
            )
        }

        mount = bridge._get_sensor_mount("camera", preferred_names=("front_camera", "depth_camera"))
        assert mount is not None
        assert mount[0] == [0.45, 0.0, 0.2]
        assert mount[1] == [90.0, 0.0, -90.0]


class TestRos2Adapter:
    """Pure data adapter tests (no ROS runtime required)."""

    def test_enrich_scan_data_fills_angle_increment(self):
        from omninav.interfaces.ros2.adapter import Ros2Adapter

        scan_data = {"ranges": np.linspace(0.1, 3.0, 720, dtype=np.float32)}
        enriched = Ros2Adapter.enrich_scan_data(scan_data)

        assert enriched["ranges"].shape == (720,)
        assert np.isclose(enriched["angle_min"], -np.pi)
        assert np.isclose(enriched["angle_max"], np.pi)
        assert enriched["angle_increment"] > 0.0

    def test_pick_camera_data_prefers_sensor_with_rgb_or_depth(self):
        from omninav.interfaces.ros2.adapter import Ros2Adapter

        sensors = {
            "lidar_2d": {"ranges": np.zeros((1, 360), dtype=np.float32)},
            "depth_camera": {"rgb": np.zeros((1, 8, 8, 3), dtype=np.uint8)},
        }
        data = Ros2Adapter.pick_camera_data(sensors)
        assert data is not None
        assert "rgb" in data

    def test_normalize_depth_image_accepts_batched_input(self):
        from omninav.interfaces.ros2.adapter import Ros2Adapter

        depth = np.zeros((1, 16, 32), dtype=np.float32)
        normalized = Ros2Adapter.normalize_depth_image(depth)
        assert normalized.shape == (16, 32)
