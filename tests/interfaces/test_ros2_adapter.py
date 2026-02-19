"""Tests for ROS2 adapter helpers."""

import numpy as np
import pytest

from omninav.interfaces.ros2.adapter import Ros2Adapter


def test_normalize_cmd_vel_from_1d():
    out = Ros2Adapter.normalize_cmd_vel_batch(np.array([1.0, 0.0, 0.2], dtype=np.float32))
    assert out.shape == (1, 3)


def test_normalize_cmd_vel_from_batch():
    out = Ros2Adapter.normalize_cmd_vel_batch(np.zeros((4, 3), dtype=np.float32))
    assert out.shape == (4, 3)


def test_normalize_cmd_vel_rejects_invalid_shape():
    with pytest.raises(ValueError):
        Ros2Adapter.normalize_cmd_vel_batch(np.array([1.0, 0.0], dtype=np.float32), "bad")


def test_pick_scan_data_finds_ranges_entry():
    sensors = {
        "camera": {"rgb": np.zeros((32, 32, 3), dtype=np.uint8)},
        "lidar": {"ranges": np.ones((1, 8), dtype=np.float32)},
    }
    out = Ros2Adapter.pick_scan_data(sensors)
    assert out is sensors["lidar"]


def test_pick_scan_data_returns_none_when_absent():
    sensors = {"camera": {"rgb": np.zeros((8, 8, 3), dtype=np.uint8)}}
    assert Ros2Adapter.pick_scan_data(sensors) is None
