"""
Local Planner Base + DWA Implementation

LocalPlannerBase provides the interface for obstacle-avoidance planners.
DWAPlanner implements the Dynamic Window Approach algorithm.
"""

from abc import abstractmethod
from typing import Dict, Any, Optional, TYPE_CHECKING
import numpy as np
from omegaconf import DictConfig

from omninav.algorithms.base import AlgorithmBase
from omninav.core.registry import ALGORITHM_REGISTRY

if TYPE_CHECKING:
    from omninav.core.types import Observation


class LocalPlannerBase(AlgorithmBase):
    """
    Base class for local obstacle-avoidance planners.

    Local planners navigate toward a target position while avoiding nearby
    obstacles using sensor data (typically lidar ranges).
    """

    @abstractmethod
    def navigate_to(self, obs: "Observation", target: np.ndarray) -> np.ndarray:
        """
        Compute cmd_vel to navigate toward a target position.

        Args:
            obs: Current observation
            target: Target position in world frame (3,)

        Returns:
            cmd_vel: [vx, vy, wz] velocity command
        """
        pass


@ALGORITHM_REGISTRY.register("dwa_planner")
class DWAPlanner(LocalPlannerBase):
    """
    Dynamic Window Approach local planner.

    Searches over a velocity space (vx, wz) to find the trajectory that:
    1. Avoids obstacles (clearance cost)
    2. Progresses toward the goal (heading cost)
    3. Maintains desired speed (velocity cost)
    """

    ALGORITHM_TYPE = "dwa_planner"

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        # DWA parameters
        self._max_speed = cfg.get("max_speed", 0.5)
        self._max_yaw_rate = cfg.get("max_yaw_rate", 1.0)
        self._speed_resolution = cfg.get("speed_resolution", 0.05)
        self._yaw_rate_resolution = cfg.get("yaw_rate_resolution", 0.1)
        self._predict_time = cfg.get("predict_time", 2.0)
        self._dt = cfg.get("dt", 0.1)

        # Cost weights
        self._heading_weight = cfg.get("heading_weight", 1.0)
        self._clearance_weight = cfg.get("clearance_weight", 1.0)
        self._velocity_weight = cfg.get("velocity_weight", 0.5)

        # Obstacle margin (meters)
        self._obstacle_margin = cfg.get("obstacle_margin", 0.3)
        self._goal_tolerance = cfg.get("goal_tolerance", 0.5)

        # State
        self._target: Optional[np.ndarray] = None
        self._is_done = False

    def reset(self, task_info: Optional[Dict[str, Any]] = None) -> None:
        """Reset planner state."""
        self._target = None
        self._is_done = False
        if task_info is not None and "goal_position" in task_info:
            self._target = np.array(task_info["goal_position"], dtype=np.float32)

    def step(self, obs: "Observation") -> np.ndarray:
        """
        Compute cmd_vel using DWA.

        Uses lidar ranges from obs['sensors'] for obstacle avoidance
        and goal_position from obs for heading.
        """
        # Extract goal
        target = self._target
        if "goal_position" in obs and obs["goal_position"] is not None:
            gp = np.asarray(obs["goal_position"])
            if gp.ndim == 2:
                gp = gp[0]  # debatch
            target = gp[:3]

        if target is None:
            return np.zeros((1, 3), dtype=np.float32)

        # Extract robot state
        robot_state = obs.get("robot_state", {})
        pos = np.asarray(robot_state.get("position", np.zeros((1, 3))))
        if pos.ndim == 2:
            pos = pos[0]

        # Check if at goal
        dist_to_goal = np.linalg.norm(target[:2] - pos[:2])
        if dist_to_goal < self._goal_tolerance:
            self._is_done = True
            return np.zeros((1, 3), dtype=np.float32)

        # Extract lidar ranges for obstacle check
        sensors = obs.get("sensors", {})
        ranges = None
        for sensor_data in sensors.values():
            if "ranges" in sensor_data:
                r = np.asarray(sensor_data["ranges"])
                if r.ndim == 2:
                    r = r[0]  # debatch
                ranges = r
                break

        # Goal heading
        goal_angle = np.arctan2(target[1] - pos[1], target[0] - pos[0])

        # Extract robot yaw from orientation
        quat = np.asarray(robot_state.get("orientation", np.array([[1, 0, 0, 0]])))
        if quat.ndim == 2:
            quat = quat[0]
        robot_yaw = self._quat_to_yaw(quat)

        # Simple DWA search
        best_cmd = np.zeros(3, dtype=np.float32)
        best_cost = -float('inf')

        for vx in np.arange(0.0, self._max_speed + 1e-6, self._speed_resolution):
            for wz in np.arange(-self._max_yaw_rate, self._max_yaw_rate + 1e-6, self._yaw_rate_resolution):
                # Predict trajectory
                pred_yaw = robot_yaw + wz * self._predict_time

                # Heading cost: how aligned is the trajectory with goal?
                heading_error = abs(self._normalize_angle(goal_angle - pred_yaw))
                heading_cost = np.pi - heading_error

                # Velocity cost: prefer faster speeds
                velocity_cost = vx

                # Clearance cost: check obstacle proximity
                clearance_cost = 0.0
                if ranges is not None and len(ranges) > 0:
                    min_range = np.min(ranges[ranges > 0.01]) if np.any(ranges > 0.01) else float('inf')
                    if min_range < self._obstacle_margin:
                        clearance_cost = -100.0  # Strongly penalize
                    else:
                        clearance_cost = min(min_range, 3.0)

                total_cost = (
                    self._heading_weight * heading_cost +
                    self._clearance_weight * clearance_cost +
                    self._velocity_weight * velocity_cost
                )

                if total_cost > best_cost:
                    best_cost = total_cost
                    best_cmd = np.array([vx, 0.0, wz], dtype=np.float32)

        return best_cmd.reshape(1, 3)

    def navigate_to(self, obs: "Observation", target: np.ndarray) -> np.ndarray:
        """Navigate to target using DWA."""
        obs_copy = dict(obs)
        obs_copy["goal_position"] = target
        return self.step(obs_copy)

    @property
    def is_done(self) -> bool:
        return self._is_done

    @property
    def info(self) -> Dict[str, Any]:
        return {"target": self._target, "is_done": self._is_done}

    @staticmethod
    def _quat_to_yaw(quat: np.ndarray) -> float:
        """Extract yaw from quaternion (wxyz)."""
        w, x, y, z = quat
        return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
