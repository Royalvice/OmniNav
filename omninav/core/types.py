"""
Core Data Type Contracts for OmniNav

All cross-layer data exchange uses these standardized types.
Every array field follows Batch-First convention: (B, ...) where B = num_envs × num_robots.

Design principles:
1. TypedDict for structured, type-checked data contracts
2. Batch-First: even single-env has shape (1, ...)
3. Centralized: all public types live here to avoid circular imports
"""

from __future__ import annotations

from typing import TypedDict, Optional, Any
from dataclasses import dataclass, field

import numpy as np


# =============================================================================
# Robot State
# =============================================================================

class RobotState(TypedDict):
    """
    Robot state snapshot. All arrays are Batch-First: (B, ...).

    B = num_envs × num_robots (flattened batch dimension).
    Single robot in single env: B = 1.
    """

    position: np.ndarray
    """World-frame position. Shape: (B, 3)"""

    orientation: np.ndarray
    """Quaternion in wxyz format. Shape: (B, 4)"""

    linear_velocity: np.ndarray
    """World-frame linear velocity. Shape: (B, 3)"""

    angular_velocity: np.ndarray
    """World-frame angular velocity. Shape: (B, 3)"""

    joint_positions: np.ndarray
    """Joint positions in radians. Shape: (B, num_joints)"""

    joint_velocities: np.ndarray
    """Joint velocities in rad/s. Shape: (B, num_joints)"""


# =============================================================================
# Sensor Data
# =============================================================================

class SensorData(TypedDict, total=False):
    """
    Sensor data dict. Keys depend on sensor type.

    All fields are optional (total=False) because different sensors
    produce different data types.
    """

    ranges: np.ndarray
    """1D distance measurements (e.g. lidar). Shape: (B, N)"""

    points: np.ndarray
    """3D point cloud. Shape: (B, N, 3)"""

    rgb: np.ndarray
    """RGB image. Shape: (B, H, W, 3), dtype uint8"""

    depth: np.ndarray
    """Depth map. Shape: (B, H, W), dtype float32, unit: meters"""

    hit_positions: np.ndarray
    """Raycaster hit positions. Shape: (B, N_rays, 3)"""

    hit_normals: np.ndarray
    """Raycaster hit normals. Shape: (B, N_rays, 3)"""


# =============================================================================
# Observation
# =============================================================================

class Observation(TypedDict, total=False):
    """
    Standardized observation passed between layers.

    All fields are Batch-First. Fields are optional (total=False)
    because different configurations produce different observation sets.
    """

    robot_state: RobotState
    """Current robot state."""

    sim_time: float
    """Current simulation time in seconds."""

    sensors: dict[str, SensorData]
    """Sensor data keyed by sensor name."""

    # Multi-robot indexing
    robot_ids: np.ndarray
    """Robot ID for each batch element. Shape: (B,)"""

    env_ids: np.ndarray
    """Environment ID for each batch element. Shape: (B,)"""

    # Task context (populated by TaskBase)
    goal_position: Optional[np.ndarray]
    """Goal position. Shape: (B, 3)"""

    goal_positions: Optional[np.ndarray]
    """Multiple goal positions (waypoints). Shape: (B, N_goals, 3)"""

    current_waypoint_index: Optional[np.ndarray]
    """Current target waypoint index. Shape: (B,)"""

    language_instruction: Optional[list[str]]
    """Natural language instructions. Length: B"""


# =============================================================================
# Action
# =============================================================================

class Action(TypedDict):
    """
    Standardized action produced by Algorithm layer, consumed by Locomotion.
    """

    cmd_vel: np.ndarray
    """Velocity command [vx, vy, wz]. Shape: (B, 3)"""


# =============================================================================
# Joint Info (Robot descriptor, read-only)
# =============================================================================

@dataclass(frozen=True)
class JointInfo:
    """
    Read-only descriptor of robot joint configuration.

    Frozen dataclass because joint info does not change at runtime.
    """

    names: tuple[str, ...]
    """Joint names, ordered by DOF index."""

    dof_indices: np.ndarray
    """DOF indices in the physics engine. Shape: (num_joints,)"""

    num_joints: int
    """Total number of controllable joints."""

    position_limits_lower: np.ndarray
    """Lower position limits. Shape: (num_joints,)"""

    position_limits_upper: np.ndarray
    """Upper position limits. Shape: (num_joints,)"""

    velocity_limits: np.ndarray
    """Maximum joint velocities. Shape: (num_joints,)"""


# =============================================================================
# Mount Info (Sensor mounting descriptor)
# =============================================================================

@dataclass
class MountInfo:
    """
    Engine-agnostic description of where to mount a sensor.

    Created by the Robot layer, consumed by Sensor.create().
    Hides the physics engine details from the Sensor layer.
    """

    link_handle: object
    """Physics engine link object (opaque to Sensor)."""

    position: np.ndarray
    """Mounting position offset relative to link. Shape: (3,)"""

    orientation: np.ndarray
    """Mounting orientation as euler angles [roll, pitch, yaw]. Shape: (3,)"""

    scene_handle: object
    """Physics engine scene object (opaque to Sensor)."""


# =============================================================================
# Task Result
# =============================================================================

@dataclass
class TaskResult:
    """
    Result of a completed evaluation task episode.
    """

    success: bool
    """Whether the task was completed successfully."""

    episode_length: int = 0
    """Number of steps in the episode."""

    elapsed_time: float = 0.0
    """Wall-clock time of the episode in seconds."""

    metrics: dict[str, float] = field(default_factory=dict)
    """Computed metric values (e.g. {'spl': 0.85, 'coverage': 0.92})."""

    info: dict[str, Any] = field(default_factory=dict)
    """Additional task-specific information."""


# =============================================================================
# Validation Helpers
# =============================================================================

def validate_batch_shape(array: np.ndarray, name: str, expected_trailing: tuple[int, ...]) -> None:
    """
    Validate that an array has Batch-First shape (B, *expected_trailing).

    Args:
        array: Array to validate
        name: Name for error messages
        expected_trailing: Expected shape after batch dimension

    Raises:
        ValueError: If shape does not match
    """
    if array.ndim < 1 + len(expected_trailing):
        raise ValueError(
            f"{name}: expected at least {1 + len(expected_trailing)} dimensions "
            f"(B, {', '.join(str(d) for d in expected_trailing)}), got shape {array.shape}"
        )
    actual_trailing = array.shape[1:]
    if actual_trailing != expected_trailing:
        raise ValueError(
            f"{name}: expected trailing shape {expected_trailing}, "
            f"got {actual_trailing} (full shape: {array.shape})"
        )


def make_batch(array: np.ndarray) -> np.ndarray:
    """
    Ensure array has a batch dimension. If missing, adds (1, ...) prefix.

    Args:
        array: Input array, possibly without batch dim

    Returns:
        Array with guaranteed batch dimension
    """
    if array.ndim == 0:
        return array.reshape(1)
    return array if array.ndim >= 2 or (array.ndim == 1 and array.shape[0] > 0) else np.expand_dims(array, 0)
