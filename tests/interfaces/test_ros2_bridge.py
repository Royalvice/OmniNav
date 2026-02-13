"""
Tests for ROS2 Bridge (Mock-based).

Note: These tests don't require actual ROS2 installation.
They use mocks to verify the bridge logic.
"""

import pytest
import numpy as np
from omegaconf import OmegaConf
from unittest.mock import MagicMock, patch


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


class TestRos2BridgeLogic:
    """Test ROS2 bridge internal logic (with mocked rclpy)."""
    
    def test_cmd_vel_callback_stores_velocity(self):
        """Test cmd_vel callback stores velocity correctly."""
        from omninav.interfaces.ros2.bridge import ROS2Bridge
        
        cfg = OmegaConf.create({"enabled": False})
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
