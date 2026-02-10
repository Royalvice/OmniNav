"""
Tests for core data type contracts.
"""

import pytest
import numpy as np

from omninav.core.types import (
    RobotState,
    SensorData,
    Observation,
    Action,
    JointInfo,
    MountInfo,
    TaskResult,
    validate_batch_shape,
    make_batch,
)


class TestRobotState:
    """Tests for RobotState TypedDict."""

    def test_create_valid_state(self):
        """Test creating a properly shaped RobotState."""
        state: RobotState = {
            "position": np.zeros((1, 3)),
            "orientation": np.array([[1.0, 0.0, 0.0, 0.0]]),
            "linear_velocity": np.zeros((1, 3)),
            "angular_velocity": np.zeros((1, 3)),
            "joint_positions": np.zeros((1, 12)),
            "joint_velocities": np.zeros((1, 12)),
        }
        assert state["position"].shape == (1, 3)
        assert state["orientation"].shape == (1, 4)

    def test_batch_dimension(self):
        """Test RobotState with multiple envs."""
        B = 4
        state: RobotState = {
            "position": np.zeros((B, 3)),
            "orientation": np.zeros((B, 4)),
            "linear_velocity": np.zeros((B, 3)),
            "angular_velocity": np.zeros((B, 3)),
            "joint_positions": np.zeros((B, 12)),
            "joint_velocities": np.zeros((B, 12)),
        }
        assert state["position"].shape[0] == B
        assert state["joint_positions"].shape == (B, 12)


class TestSensorData:
    """Tests for SensorData TypedDict (partial keys)."""

    def test_lidar_data(self):
        """Test SensorData with lidar ranges."""
        data: SensorData = {"ranges": np.zeros((1, 720))}
        assert data["ranges"].shape == (1, 720)

    def test_camera_data(self):
        """Test SensorData with RGB + depth."""
        data: SensorData = {
            "rgb": np.zeros((1, 480, 640, 3), dtype=np.uint8),
            "depth": np.zeros((1, 480, 640), dtype=np.float32),
        }
        assert data["rgb"].shape == (1, 480, 640, 3)
        assert data["depth"].dtype == np.float32

    def test_raycaster_data(self):
        """Test SensorData with hit positions."""
        data: SensorData = {
            "hit_positions": np.zeros((1, 100, 3)),
        }
        assert data["hit_positions"].shape == (1, 100, 3)


class TestObservation:
    """Tests for Observation TypedDict."""

    def test_minimal_observation(self):
        """Test observation with only required fields."""
        obs: Observation = {
            "robot_state": {
                "position": np.zeros((1, 3)),
                "orientation": np.array([[1.0, 0.0, 0.0, 0.0]]),
                "linear_velocity": np.zeros((1, 3)),
                "angular_velocity": np.zeros((1, 3)),
                "joint_positions": np.zeros((1, 12)),
                "joint_velocities": np.zeros((1, 12)),
            },
            "sim_time": 0.0,
            "sensors": {},
        }
        assert obs["sim_time"] == 0.0
        assert len(obs["sensors"]) == 0

    def test_observation_with_sensors(self):
        """Test observation including sensor data."""
        obs: Observation = {
            "robot_state": {
                "position": np.zeros((1, 3)),
                "orientation": np.array([[1.0, 0.0, 0.0, 0.0]]),
                "linear_velocity": np.zeros((1, 3)),
                "angular_velocity": np.zeros((1, 3)),
                "joint_positions": np.zeros((1, 12)),
                "joint_velocities": np.zeros((1, 12)),
            },
            "sim_time": 1.5,
            "sensors": {
                "front_lidar": {"ranges": np.zeros((1, 720))},
                "front_camera": {
                    "rgb": np.zeros((1, 480, 640, 3), dtype=np.uint8),
                    "depth": np.zeros((1, 480, 640), dtype=np.float32),
                },
            },
        }
        assert len(obs["sensors"]) == 2
        assert "front_lidar" in obs["sensors"]

    def test_multi_robot_observation(self):
        """Test observation with multi-robot indexing."""
        B = 6  # 2 envs × 3 robots
        obs: Observation = {
            "robot_state": {
                "position": np.zeros((B, 3)),
                "orientation": np.zeros((B, 4)),
                "linear_velocity": np.zeros((B, 3)),
                "angular_velocity": np.zeros((B, 3)),
                "joint_positions": np.zeros((B, 12)),
                "joint_velocities": np.zeros((B, 12)),
            },
            "sim_time": 0.0,
            "sensors": {},
            "robot_ids": np.array([0, 1, 2, 0, 1, 2]),
            "env_ids": np.array([0, 0, 0, 1, 1, 1]),
        }
        assert obs["robot_ids"].shape == (B,)
        assert obs["env_ids"].shape == (B,)


class TestAction:
    """Tests for Action TypedDict."""

    def test_action_creation(self):
        """Test creating an action."""
        action: Action = {"cmd_vel": np.array([[0.5, 0.0, 0.1]])}
        assert action["cmd_vel"].shape == (1, 3)

    def test_batch_action(self):
        """Test batch action."""
        B = 4
        action: Action = {"cmd_vel": np.zeros((B, 3))}
        assert action["cmd_vel"].shape == (B, 3)


class TestJointInfo:
    """Tests for JointInfo frozen dataclass."""

    def test_joint_info_creation(self):
        """Test creating a JointInfo descriptor."""
        info = JointInfo(
            names=("hip", "knee", "ankle"),
            dof_indices=np.array([0, 1, 2]),
            num_joints=3,
            position_limits_lower=np.array([-1.0, -2.0, -1.5]),
            position_limits_upper=np.array([1.0, 2.0, 1.5]),
            velocity_limits=np.array([10.0, 10.0, 10.0]),
        )
        assert info.num_joints == 3
        assert len(info.names) == 3

    def test_joint_info_is_frozen(self):
        """Test that JointInfo is immutable."""
        info = JointInfo(
            names=("hip",),
            dof_indices=np.array([0]),
            num_joints=1,
            position_limits_lower=np.array([-1.0]),
            position_limits_upper=np.array([1.0]),
            velocity_limits=np.array([10.0]),
        )
        with pytest.raises(AttributeError):
            info.num_joints = 5  # type: ignore


class TestMountInfo:
    """Tests for MountInfo dataclass."""

    def test_mount_info_creation(self):
        """Test creating mount info."""
        mount = MountInfo(
            link_handle=object(),
            position=np.array([0.3, 0.0, 0.1]),
            orientation=np.zeros(3),
            scene_handle=object(),
        )
        assert mount.position.shape == (3,)


class TestTaskResult:
    """Tests for TaskResult dataclass."""

    def test_task_result(self):
        """Test creating a task result."""
        result = TaskResult(
            success=True,
            episode_length=500,
            elapsed_time=50.0,
            metrics={"spl": 0.85, "coverage": 0.92},
            info={"final_position": [1.0, 2.0, 0.0]},
        )
        assert result.success is True
        assert result.metrics["spl"] == 0.85


class TestValidateBatchShape:
    """Tests for validate_batch_shape helper."""

    def test_valid_shape(self):
        """Test that valid shapes pass."""
        arr = np.zeros((4, 3))
        validate_batch_shape(arr, "position", (3,))  # Should not raise

    def test_invalid_trailing(self):
        """Test that wrong trailing shape raises."""
        arr = np.zeros((4, 5))
        with pytest.raises(ValueError, match="expected trailing shape"):
            validate_batch_shape(arr, "position", (3,))

    def test_too_few_dims(self):
        """Test that missing batch dim raises."""
        arr = np.zeros(3)
        with pytest.raises(ValueError, match="expected at least"):
            validate_batch_shape(arr, "position", (3,))

    def test_matrix_shape(self):
        """Test validation with matrix trailing shape."""
        arr = np.zeros((2, 480, 640))
        validate_batch_shape(arr, "depth", (480, 640))


class TestMakeBatch:
    """Tests for make_batch helper."""

    def test_already_batched(self):
        """Test that already-batched arrays pass through."""
        arr = np.zeros((4, 3))
        result = make_batch(arr)
        assert result.shape == (4, 3)

    def test_scalar_gets_batch(self):
        """Test that scalar gets batch wrapping."""
        arr = np.float64(1.5)
        result = make_batch(arr)
        assert result.ndim >= 1
