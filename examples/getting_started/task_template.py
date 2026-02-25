"""
Template for defining a minimal inspection task.

Copy this file into omninav/evaluation/tasks/ and register it if needed.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, TYPE_CHECKING
import numpy as np
from omegaconf import DictConfig

from omninav.evaluation.base import TaskBase
from omninav.core.types import TaskResult
from omninav.core.registry import TASK_REGISTRY

if TYPE_CHECKING:
    from omninav.core.types import Observation, Action


@TASK_REGISTRY.register("my_min_inspection_template")
class MyMinimalInspectionTaskTemplate(TaskBase):
    """
    Minimal inspection task definition.

    This task:
    - provides a list of checkpoints to the planner
    - tracks simple coverage proxy
    - terminates on coverage threshold or timeout
    """

    TASK_TYPE = "my_min_inspection_template"

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self._waypoints = [np.asarray(w, dtype=np.float32) for w in cfg.get("waypoints", [[1, 0, 0], [2, 0, 0]])]
        self._time_budget = float(cfg.get("time_budget", 60.0))
        self._coverage_requirement = float(cfg.get("coverage_requirement", 1.0))
        self._tol = float(cfg.get("waypoint_tolerance", 0.4))
        self._visited: list[bool] = []
        self._step_count = 0
        self._current_time = 0.0
        self._start_time = 0.0

    def reset(self) -> Dict[str, Any]:
        self._visited = [False] * len(self._waypoints)
        self._step_count = 0
        self._current_time = 0.0
        self._start_time = 0.0
        self._task_spec = {
            "task_type": "inspection",
            "goal_set": [w.tolist() for w in self._waypoints],
            "waypoint_tolerance": self._tol,
            "time_budget": self._time_budget,
            "coverage_requirement": self._coverage_requirement,
            "scan_at_goal": True,
            "scan_duration": 1.5,
            "scan_angular_velocity": 0.8,
        }
        return self._task_spec

    def step(self, obs: "Observation", action: Optional["Action"] = None) -> None:
        _ = action
        self._step_count += 1
        self._current_time = float(obs.get("sim_time", 0.0))
        pos = np.asarray(obs.get("robot_state", {}).get("position", np.zeros((1, 3))), dtype=np.float32)
        if pos.ndim == 2:
            p = pos[0]
            for i, wp in enumerate(self._waypoints):
                if not self._visited[i] and np.linalg.norm(wp[:2] - p[:2]) <= self._tol:
                    self._visited[i] = True

    def is_terminated(self, obs: "Observation") -> np.ndarray:
        elapsed = float(obs.get("sim_time", 0.0)) - self._start_time
        done = elapsed >= self._time_budget or self.coverage >= self._coverage_requirement
        pos = np.asarray(obs.get("robot_state", {}).get("position", np.zeros((1, 3))), dtype=np.float32)
        b = int(pos.shape[0]) if pos.ndim >= 2 else 1
        return np.full((b,), bool(done), dtype=bool)

    def compute_result(self) -> TaskResult:
        return TaskResult(
            success=bool(self.coverage >= self._coverage_requirement),
            episode_length=self._step_count,
            elapsed_time=self._current_time - self._start_time,
            metrics={
                "coverage_rate": self.coverage,
                "waypoints_visited": float(sum(self._visited)),
                "waypoints_total": float(len(self._visited)),
            },
            info={"visited": list(self._visited)},
        )

    @property
    def coverage(self) -> float:
        if not self._visited:
            return 1.0
        return float(sum(self._visited)) / float(len(self._visited))
