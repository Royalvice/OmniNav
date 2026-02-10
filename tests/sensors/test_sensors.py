"""
Tests for Sensor implementations.
"""

import pytest
import numpy as np
from omegaconf import OmegaConf
from unittest.mock import MagicMock, patch


class TestSensorBase:
    """Test suite for SensorBase."""
    
    def test_attach_sets_properties(self, mock_scene, mock_robot):
        """Test that attach sets position and orientation offsets."""
        from omninav.sensors.base import SensorBase
        
        # Create a concrete subclass for testing
        class TestSensor(SensorBase):
            def create(self):
                pass
            def get_data(self):
                return {}
        
        cfg = OmegaConf.create({"type": "test"})
        sensor = TestSensor(cfg, mock_scene, mock_robot)
        
        sensor.attach("base", [1.0, 2.0, 3.0], [0.1, 0.2, 0.3])
        
        assert sensor._link_name == "base"
        assert np.allclose(sensor._pos_offset, [1.0, 2.0, 3.0])
        assert np.allclose(sensor._euler_offset, [0.1, 0.2, 0.3])
    
    def test_is_ready_before_create(self, mock_scene, mock_robot):
        """Test that is_ready is False before create is called."""
        from omninav.sensors.base import SensorBase
        
        class TestSensor(SensorBase):
            def create(self):
                self._is_created = True
            def get_data(self):
                return {}
        
        cfg = OmegaConf.create({"type": "test"})
        sensor = TestSensor(cfg, mock_scene, mock_robot)
        
        assert not sensor.is_ready


class TestLidar2DSensor:
    """Test suite for Lidar2DSensor."""
    
    def test_sensor_registered(self):
        """Test that Lidar2DSensor is registered in SENSOR_REGISTRY."""
        from omninav.core.registry import SENSOR_REGISTRY
        
        assert "lidar_2d" in SENSOR_REGISTRY
    
    def test_angle_properties(self, mock_scene, mock_robot):
        """Test angle_min, angle_max, and angle_increment properties."""
        from omninav.sensors.lidar import Lidar2DSensor
        
        cfg = OmegaConf.create({
            "type": "lidar_2d",
            "horizontal_fov": 360.0,
            "num_rays": 720,
        })
        
        sensor = Lidar2DSensor(cfg, mock_scene, mock_robot)
        
        assert np.isclose(sensor.angle_min, -np.pi)
        assert np.isclose(sensor.angle_max, np.pi)
        assert np.isclose(sensor.angle_increment, 2 * np.pi / 720)
    
    def test_get_data_shape_before_ready(self, mock_scene, mock_robot):
        """Test get_data returns zeros before sensor is ready."""
        from omninav.sensors.lidar import Lidar2DSensor
        
        cfg = OmegaConf.create({
            "type": "lidar_2d",
            "num_rays": 360,
        })
        
        sensor = Lidar2DSensor(cfg, mock_scene, mock_robot)
        # Don't call create() - sensor not ready
        
        data = sensor.get_data()
        
        assert "ranges" in data
        assert data["ranges"].shape == (1, 360)
        assert np.all(data["ranges"] == 0)


class TestCameraSensor:
    """Test suite for CameraSensor."""
    
    def test_sensor_registered(self):
        """Test that CameraSensor is registered in SENSOR_REGISTRY."""
        from omninav.core.registry import SENSOR_REGISTRY
        
        assert "camera" in SENSOR_REGISTRY
    
    def test_resolution_property(self, mock_scene, mock_robot):
        """Test resolution property returns correct values."""
        from omninav.sensors.camera import CameraSensor
        
        cfg = OmegaConf.create({
            "type": "camera",
            "width": 640,
            "height": 480,
        })
        
        sensor = CameraSensor(cfg, mock_scene, mock_robot)
        
        assert sensor.resolution == (640, 480)
    
    def test_get_data_shape_before_ready(self, mock_scene, mock_robot):
        """Test get_data returns zeros before sensor is ready."""
        from omninav.sensors.camera import CameraSensor
        
        cfg = OmegaConf.create({
            "type": "camera",
            "width": 320,
            "height": 240,
            "camera_types": ["rgb", "depth"],
        })
        
        sensor = CameraSensor(cfg, mock_scene, mock_robot)
        # Don't call create() - sensor not ready
        
        data = sensor.get_data()
        
        assert "rgb" in data
        assert data["rgb"].shape == (1, 240, 320, 3)
        assert "depth" in data
        assert data["depth"].shape == (1, 240, 320)
