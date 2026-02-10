"""
Tests for Robot layer refactoring.

Tests lifecycle enforcement, Batch-First state, and joint info.
Uses mock objects (no Genesis dependency).
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, PropertyMock
from omegaconf import OmegaConf

from omninav.robots.base import RobotBase
from omninav.core.types import RobotState, JointInfo
from omninav.core.lifecycle import LifecycleState


# =============================================================================
# Concrete Test Robot (no Genesis dependency)
# =============================================================================

class MockRobotImpl(RobotBase):
    """Concrete robot implementation for testing."""

    ROBOT_TYPE = "mock_robot"
    JOINT_NAMES = ["joint_a", "joint_b", "joint_c"]

    def __init__(self, cfg=None, scene=None):
        if cfg is None:
            cfg = OmegaConf.create({"type": "mock_robot"})
        super().__init__(cfg, scene)
        self._joint_info = None

    def spawn(self) -> None:
        self._require_state(LifecycleState.CREATED, "spawn")
        # Simulate entity creation
        self.entity = MagicMock()
        self.entity.get_pos.return_value = np.array([1.0, 2.0, 0.4])
        self.entity.get_quat.return_value = np.array([1.0, 0.0, 0.0, 0.0])
        self.entity.get_vel.return_value = np.array([0.0, 0.0, 0.0])
        self.entity.get_ang.return_value = np.array([0.0, 0.0, 0.0])
        self.entity.get_dofs_position.return_value = np.zeros(3)
        self.entity.get_dofs_velocity.return_value = np.zeros(3)
        self.entity.get_link.return_value = MagicMock()
        self._initial_pos = np.array([0.0, 0.0, 0.4])
        self._initial_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self._transition_to(LifecycleState.SPAWNED)

    def get_state(self) -> RobotState:
        self._require_state(LifecycleState.BUILT, "get_state")
        pos = self.entity.get_pos()
        quat = self.entity.get_quat()
        vel = self.entity.get_vel()
        ang = self.entity.get_ang()
        jp = self.entity.get_dofs_position()
        jv = self.entity.get_dofs_velocity()
        # Ensure Batch-First
        def batch(x):
            a = np.asarray(x)
            return a.reshape(1, -1) if a.ndim == 1 else a
        return RobotState(
            position=batch(pos),
            orientation=batch(quat),
            linear_velocity=batch(vel),
            angular_velocity=batch(ang),
            joint_positions=batch(jp),
            joint_velocities=batch(jv),
        )

    def get_joint_info(self) -> JointInfo:
        self._require_state(LifecycleState.BUILT, "get_joint_info")
        if self._joint_info is None:
            self._joint_info = JointInfo(
                names=tuple(self.JOINT_NAMES),
                dof_indices=np.array([0, 1, 2], dtype=np.int32),
                num_joints=3,
                position_limits_lower=np.array([-3.14, -3.14, -3.14]),
                position_limits_upper=np.array([3.14, 3.14, 3.14]),
                velocity_limits=np.array([10.0, 10.0, 10.0]),
            )
        return self._joint_info

    def post_build(self) -> None:
        # Initialize joints
        self._joint_info = None  # Will be lazily created
        super().post_build()


# =============================================================================
# Tests
# =============================================================================

class TestRobotLifecycle:
    """Test lifecycle state enforcement."""

    def test_initial_state(self):
        robot = MockRobotImpl()
        assert robot.lifecycle_state == LifecycleState.CREATED

    def test_spawn_transitions(self):
        robot = MockRobotImpl()
        robot.spawn()
        assert robot.lifecycle_state == LifecycleState.SPAWNED

    def test_full_lifecycle(self):
        robot = MockRobotImpl()
        robot.spawn()
        robot.mount_sensors([])  # empty sensors
        robot.post_build()
        robot.reset()
        assert robot.lifecycle_state == LifecycleState.READY

    def test_get_state_before_build_fails(self):
        robot = MockRobotImpl()
        robot.spawn()
        with pytest.raises(RuntimeError, match="BUILT is required"):
            robot.get_state()

    def test_reset_before_build_fails(self):
        robot = MockRobotImpl()
        robot.spawn()
        with pytest.raises(RuntimeError, match="BUILT is required"):
            robot.reset()

    def test_spawn_twice_fails(self):
        robot = MockRobotImpl()
        robot.spawn()
        with pytest.raises(RuntimeError, match="must go forward"):
            robot.spawn()


class TestRobotStateBatchFirst:
    """Test Batch-First state output."""

    def _ready_robot(self):
        robot = MockRobotImpl()
        robot.spawn()
        robot.mount_sensors([])
        robot.post_build()
        robot.reset()
        return robot

    def test_state_has_batch_dim(self):
        robot = self._ready_robot()
        state = robot.get_state()
        assert state["position"].shape == (1, 3)
        assert state["orientation"].shape == (1, 4)
        assert state["linear_velocity"].shape == (1, 3)
        assert state["angular_velocity"].shape == (1, 3)
        assert state["joint_positions"].shape == (1, 3)
        assert state["joint_velocities"].shape == (1, 3)

    def test_state_is_typed_dict(self):
        robot = self._ready_robot()
        state = robot.get_state()
        assert isinstance(state, dict)
        assert "position" in state
        assert "orientation" in state


class TestJointInfo:
    """Test JointInfo descriptor."""

    def _built_robot(self):
        robot = MockRobotImpl()
        robot.spawn()
        robot.mount_sensors([])
        robot.post_build()
        return robot

    def test_joint_info_available_after_build(self):
        robot = self._built_robot()
        info = robot.get_joint_info()
        assert info.num_joints == 3
        assert len(info.names) == 3
        assert info.dof_indices.shape == (3,)

    def test_joint_info_before_build_fails(self):
        robot = MockRobotImpl()
        robot.spawn()
        with pytest.raises(RuntimeError, match="BUILT is required"):
            robot.get_joint_info()

    def test_joint_info_is_frozen(self):
        robot = self._built_robot()
        info = robot.get_joint_info()
        with pytest.raises(AttributeError):
            info.num_joints = 99


class TestControlJoints:
    """Test hardware abstraction control methods."""

    def _built_robot(self):
        robot = MockRobotImpl()
        robot.spawn()
        robot.mount_sensors([])
        robot.post_build()
        return robot

    def test_control_position_delegates(self):
        robot = self._built_robot()
        targets = np.array([0.1, 0.2, 0.3])
        indices = np.array([0, 1, 2])
        robot.control_joints_position(targets, indices)
        robot.entity.control_dofs_position.assert_called_once_with(targets, indices)

    def test_control_velocity_delegates(self):
        robot = self._built_robot()
        targets = np.array([1.0, 2.0, 3.0])
        indices = np.array([0, 1, 2])
        robot.control_joints_velocity(targets, indices)
        robot.entity.control_dofs_velocity.assert_called_once_with(targets, indices)

    def test_control_before_build_fails(self):
        robot = MockRobotImpl()
        robot.spawn()
        with pytest.raises(RuntimeError, match="BUILT is required"):
            robot.control_joints_position(np.zeros(3), np.array([0, 1, 2]))


class TestMountInfo:
    """Test MountInfo creation."""

    def test_get_mount_info(self):
        robot = MockRobotImpl()
        robot.spawn()
        mount = robot.get_mount_info("base_link", [0.3, 0.0, 0.1], [0, 0, 0])
        assert mount.position.shape == (3,)
        assert mount.orientation.shape == (3,)
        assert mount.link_handle is not None
