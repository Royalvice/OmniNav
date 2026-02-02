"""Pytest configuration file"""

import pytest
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# =============================================================================
# Mock Genesis Objects
# =============================================================================

import numpy as np
from omegaconf import OmegaConf
from unittest.mock import MagicMock


class MockLink:
    """Mock Genesis link object."""
    
    def __init__(self, name: str, idx_local: int = 0):
        self.name = name
        self.idx_local = idx_local


class MockEntity:
    """Mock Genesis entity (robot) object."""
    
    def __init__(self, joint_names=None):
        self.idx = 0
        self.joint_names = joint_names or [
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        ]
        self.links = [
            MockLink("base", 0),
            MockLink("FL_foot", 1),
            MockLink("FR_foot", 2),
            MockLink("RL_foot", 3),
            MockLink("RR_foot", 4),
        ]
        self._dof_positions = np.zeros(12, dtype=np.float32)
        self._dof_velocities = np.zeros(12, dtype=np.float32)
    
    def get_link(self, name: str):
        for link in self.links:
            if link.name == name:
                return link
        return None
    
    def get_dofs_position(self):
        return MagicMock(cpu=lambda: MagicMock(numpy=lambda: self._dof_positions))
    
    def control_dofs_position(self, positions, dof_indices=None):
        if dof_indices is not None:
            self._dof_positions[dof_indices] = positions
        else:
            self._dof_positions[:len(positions)] = positions
    
    def control_dofs_velocity(self, velocities, dof_indices=None):
        if dof_indices is not None:
            self._dof_velocities[dof_indices] = velocities
        else:
            self._dof_velocities[:len(velocities)] = velocities


class MockScene:
    """Mock Genesis scene object."""
    
    def __init__(self):
        self.is_built = True
        self.visualizer = MagicMock()
        self._visualizer = MagicMock()
        self._visualizer._cameras = []
        self._sensors = []
    
    def add_sensor(self, sensor_options):
        mock_sensor = MagicMock()
        mock_sensor._is_built = True
        mock_sensor.read = MagicMock(return_value=MagicMock(
            hit_pos=np.zeros((720, 3), dtype=np.float32)
        ))
        self._sensors.append(mock_sensor)
        return mock_sensor


class MockRobot:
    """Mock robot for testing."""
    
    def __init__(self):
        self.entity = MockEntity()
        self.sensors = {}
        self.cfg = OmegaConf.create({})


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_scene():
    """Provide a mock Genesis scene."""
    return MockScene()


@pytest.fixture
def mock_robot():
    """Provide a mock robot."""
    return MockRobot()


@pytest.fixture
def mock_entity():
    """Provide a mock Genesis entity."""
    return MockEntity()


@pytest.fixture
def sample_wheel_cfg():
    """Provide sample wheel controller configuration."""
    return OmegaConf.create({
        "type": "wheel_controller",
        "wheel_radius": 0.05,
        "wheel_base": 0.4,
        "track_width": 0.3,
        "max_wheel_speed": 20.0,
        "wheel_joints": [
            "FL_wheel_joint",
            "FR_wheel_joint",
            "RL_wheel_joint",
            "RR_wheel_joint",
        ],
    })
