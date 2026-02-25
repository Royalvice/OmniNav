"""
Waypoint Task — atomic navigation task with ordered waypoint completion.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional, TYPE_CHECKING
import numpy as np
from omegaconf import DictConfig

from omninav.evaluation.base import TaskBase
from omninav.core.types import TaskResult
from omninav.core.lifecycle import LifecycleState
from omninav.core.registry import TASK_REGISTRY

if TYPE_CHECKING:
    from omninav.core.types import Observation, Action


@TASK_REGISTRY.register("waypoint")
class WaypointTask(TaskBase):
    """Ordered waypoint navigation task."""

    TASK_TYPE = "waypoint"

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self._waypoints: List[np.ndarray] = [
            np.asarray(wp, dtype=np.float32)
            for wp in cfg.get("waypoints", [])
        ]
        self._time_budget = float(cfg.get("time_budget", 120.0))
        self._success_requirement = float(cfg.get("success_requirement", 1.0))
        self._waypoint_tolerance = float(cfg.get("waypoint_tolerance", 0.5))
        self._order_policy = str(cfg.get("order_policy", "strict"))

        self._step_count = 0
        self._start_time = 0.0
        self._current_time = 0.0
        self._trajectory: List[np.ndarray] = []
        self._visited: List[bool] = []
        self._success = False

    def reset(self) -> Dict[str, Any]:
        self._step_count = 0
        self._start_time = 0.0
        self._current_time = 0.0
        self._trajectory = []
        self._visited = [False] * len(self._waypoints)
        self._success = False

        for metric in self.metrics:
            metric.reset()

        self._task_spec = {
            "task_type": self.TASK_TYPE,
            "goal_set": [wp.tolist() for wp in self._waypoints],
            "order_policy": self._order_policy,
            "waypoint_tolerance": self._waypoint_tolerance,
            "time_budget": self._time_budget,
        }
        self._transition_to(LifecycleState.READY)
        return self._task_spec

    def step(self, obs: "Observation", action: Optional["Action"] = None) -> None:
        self._step_count += 1
        self._current_time = float(obs.get("sim_time", 0.0))
        pos = np.asarray(obs.get("robot_state", {}).get("position", np.zeros((1, 3))), dtype=np.float32)
        if pos.ndim == 2:
            pos = pos[0]
        self._trajectory.append(pos.copy())

        for metric in self.metrics:
            metric.update(obs, action)

    def is_terminated(self, obs: "Observation") -> np.ndarray:
        terminated = False
        elapsed = float(obs.get("sim_time", 0.0)) - self._start_time
        if elapsed >= self._time_budget:
            terminated = True

        robot_state = obs.get("robot_state", {})
        pos = np.asarray(robot_state.get("position", np.zeros((1, 3))), dtype=np.float32)
        if pos.ndim == 2:
            p = pos[0]
            for i, wp in enumerate(self._waypoints):
                if not self._visited[i]:
                    if np.linalg.norm(wp[:2] - p[:2]) <= self._waypoint_tolerance:
                        self._visited[i] = True
                        break  # strict sequential visitation by task semantics

        coverage = self.coverage
        if coverage >= self._success_requirement:
            self._success = True
            terminated = True

        batch_size = int(pos.shape[0]) if pos.ndim >= 2 else 1
        return np.full((batch_size,), terminated, dtype=bool)

    def compute_result(self) -> TaskResult:
        coverage = self.coverage
        path_length = 0.0
        for i in range(1, len(self._trajectory)):
            path_length += float(np.linalg.norm(self._trajectory[i] - self._trajectory[i - 1]))
        elapsed = self._current_time - self._start_time

        metrics = {
            "waypoint_coverage": coverage,
            "path_length": path_length,
            "elapsed_time": elapsed,
            "step_count": float(self._step_count),
            "waypoints_total": float(len(self._waypoints)),
            "waypoints_visited": float(sum(self._visited)),
        }
        for metric in self.metrics:
            metrics[metric.METRIC_NAME] = metric.compute()

        return TaskResult(
            success=bool(coverage >= self._success_requirement),
            episode_length=self._step_count,
            elapsed_time=elapsed,
            metrics=metrics,
            info={
                "visited": list(self._visited),
                "order_policy": self._order_policy,
                "trajectory_tail": [p.tolist() for p in self._trajectory[-10:]],
            },
        )

    @property
    def coverage(self) -> float:
        if not self._waypoints:
            return 1.0
        return float(sum(self._visited)) / float(len(self._waypoints))
