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
        self._front_sector_width = np.deg2rad(float(cfg.get("front_sector_width_deg", 40.0)))
        self._lookahead_gain = float(cfg.get("lookahead_gain", 0.6))
        self._enable_path_tracking = bool(cfg.get("enable_path_tracking", True))
        self._path_weight = float(cfg.get("path_weight", 1.1))
        self._path_lookahead_points = int(cfg.get("path_lookahead_points", 5))
        self._near_goal_distance = float(cfg.get("near_goal_distance", 0.4))
        self._near_goal_kp_lin = float(cfg.get("near_goal_kp_lin", 0.8))
        self._near_goal_kp_yaw = float(cfg.get("near_goal_kp_yaw", 1.6))
        self._near_goal_max_speed = float(cfg.get("near_goal_max_speed", 0.25))

        # State
        self._target: Optional[np.ndarray] = None
        self._is_done = False
        self._last_info: Dict[str, Any] = {
            "target": None,
            "is_done": False,
            "selected_vx": 0.0,
            "selected_wz": 0.0,
            "min_front_range": float("inf"),
            "blocked_front": False,
            "near_goal_mode": False,
            "path_points": 0,
            "dist_to_path": float("inf"),
        }

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

        # Extract lidar ranges + angular metadata for obstacle check
        sensors = obs.get("sensors", {})
        ranges = None
        angle_min = -np.pi
        angle_increment = 2.0 * np.pi / 360.0
        for sensor_data in sensors.values():
            if "ranges" not in sensor_data:
                continue
            r = np.asarray(sensor_data["ranges"], dtype=np.float32)
            if r.ndim == 2:
                r = r[0]  # debatch
            ranges = r
            if "angle_min" in sensor_data:
                angle_min = float(sensor_data["angle_min"])
            if "angle_increment" in sensor_data:
                angle_increment = float(sensor_data["angle_increment"])
            elif "angle_max" in sensor_data and r.size > 1:
                angle_max = float(sensor_data["angle_max"])
                angle_increment = (angle_max - angle_min) / float(r.size - 1)
            break

        # Goal heading
        goal_angle = np.arctan2(target[1] - pos[1], target[0] - pos[0])

        # Extract robot yaw from orientation
        quat = np.asarray(robot_state.get("orientation", np.array([[1, 0, 0, 0]])))
        if quat.ndim == 2:
            quat = quat[0]
        robot_yaw = self._quat_to_yaw(quat)

        path_points = None
        dist_to_path = float("inf")
        if self._enable_path_tracking and "path_points" in obs and obs["path_points"] is not None:
            p = np.asarray(obs["path_points"], dtype=np.float32)
            if p.ndim == 3:
                p = p[0]
            if p.ndim == 2 and p.shape[0] > 0:
                path_points = p[:, :3]
                dist_to_path = self._distance_to_polyline(pos[:2], path_points[:, :2])
                goal_angle = self._goal_heading_from_path(pos[:2], path_points[:, :2], fallback=goal_angle)

        near_goal_mode = bool(dist_to_goal < self._near_goal_distance)
        if near_goal_mode:
            yaw_err = self._normalize_angle(goal_angle - robot_yaw)
            vx_cmd = float(np.clip(self._near_goal_kp_lin * dist_to_goal, 0.0, self._near_goal_max_speed))
            wz_cmd = float(np.clip(self._near_goal_kp_yaw * yaw_err, -self._max_yaw_rate, self._max_yaw_rate))
            cmd = np.array([vx_cmd, 0.0, wz_cmd], dtype=np.float32).reshape(1, 3)
            self._last_info = {
                "target": None if target is None else np.asarray(target, dtype=np.float32),
                "is_done": self._is_done,
                "selected_vx": float(vx_cmd),
                "selected_wz": float(wz_cmd),
                "min_front_range": float("inf"),
                "blocked_front": False,
                "near_goal_mode": True,
                "path_points": int(path_points.shape[0]) if path_points is not None else 0,
                "dist_to_path": float(dist_to_path),
            }
            return cmd

        # Simple DWA search
        best_cmd = np.array([0.0, 0.0, np.clip(goal_angle - robot_yaw, -self._max_yaw_rate, self._max_yaw_rate)], dtype=np.float32)
        best_cost = -float('inf')
        best_front_clearance = float("inf")
        blocked_front = False

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
                front_clearance = float("inf")
                if ranges is not None and len(ranges) > 0:
                    front_clearance = self._sector_clearance(
                        ranges=ranges,
                        angle_min=angle_min,
                        angle_increment=angle_increment,
                        center_angle=self._normalize_angle(pred_yaw - robot_yaw),
                        width=self._front_sector_width,
                    )
                    safety_dist = self._obstacle_margin + max(vx, 0.0) * self._predict_time * self._lookahead_gain
                    if front_clearance < safety_dist:
                        if vx > 1e-4:
                            # Candidate predicted to collide in the heading sector.
                            continue
                        blocked_front = True
                        clearance_cost = -5.0 + front_clearance
                    else:
                        clearance_cost = min(front_clearance, 3.0)

                total_cost = (
                    self._heading_weight * heading_cost +
                    self._clearance_weight * clearance_cost +
                    self._velocity_weight * velocity_cost
                )

                if total_cost > best_cost:
                    best_cost = total_cost
                    best_cmd = np.array([vx, 0.0, wz], dtype=np.float32)
                    best_front_clearance = front_clearance

        # If all forward candidates were infeasible, rotate in place toward goal.
        if best_cost == -float("inf"):
            yaw_err = self._normalize_angle(goal_angle - robot_yaw)
            best_cmd = np.array([0.0, 0.0, np.clip(yaw_err, -self._max_yaw_rate, self._max_yaw_rate)], dtype=np.float32)
            best_front_clearance = self._sector_clearance(
                ranges=ranges,
                angle_min=angle_min,
                angle_increment=angle_increment,
                center_angle=0.0,
                width=self._front_sector_width,
            ) if ranges is not None and len(ranges) > 0 else float("inf")
            blocked_front = True

        self._last_info = {
            "target": None if target is None else np.asarray(target, dtype=np.float32),
            "is_done": self._is_done,
            "selected_vx": float(best_cmd[0]),
            "selected_wz": float(best_cmd[2]),
            "min_front_range": float(best_front_clearance),
            "blocked_front": bool(blocked_front),
            "near_goal_mode": False,
            "path_points": int(path_points.shape[0]) if path_points is not None else 0,
            "dist_to_path": float(dist_to_path),
        }

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
        return dict(self._last_info)

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

    def _sector_clearance(
        self,
        ranges: Optional[np.ndarray],
        angle_min: float,
        angle_increment: float,
        center_angle: float,
        width: float,
    ) -> float:
        """Compute minimum valid range inside an angular sector."""
        if ranges is None or ranges.size == 0:
            return float("inf")
        idx = np.arange(ranges.size, dtype=np.float32)
        angles = angle_min + idx * angle_increment
        rel = np.abs((angles - center_angle + np.pi) % (2.0 * np.pi) - np.pi)
        mask = rel <= (0.5 * width)
        if not np.any(mask):
            return float("inf")
        valid = np.asarray(ranges[mask], dtype=np.float32)
        valid = valid[np.isfinite(valid) & (valid > 1e-3)]
        if valid.size == 0:
            return float("inf")
        return float(np.min(valid))

    @staticmethod
    def _distance_to_polyline(point_xy: np.ndarray, path_xy: np.ndarray) -> float:
        if path_xy.shape[0] == 0:
            return float("inf")
        d = np.linalg.norm(path_xy - point_xy.reshape(1, 2), axis=1)
        return float(np.min(d))

    def _goal_heading_from_path(self, point_xy: np.ndarray, path_xy: np.ndarray, fallback: float) -> float:
        if path_xy.shape[0] < 2:
            return fallback
        d = np.linalg.norm(path_xy - point_xy.reshape(1, 2), axis=1)
        idx = int(np.argmin(d))
        j = min(path_xy.shape[0] - 1, idx + max(1, self._path_lookahead_points))
        look = path_xy[j]
        return float(np.arctan2(look[1] - point_xy[1], look[0] - point_xy[0]))
