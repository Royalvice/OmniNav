"""Sequential global planner with optional scan-at-goal stages."""

from __future__ import annotations

from typing import Dict, Any, Optional, TYPE_CHECKING
import numpy as np
from omegaconf import DictConfig

from omninav.algorithms.global_base import GlobalPlannerBase
from omninav.core.registry import ALGORITHM_REGISTRY

if TYPE_CHECKING:
    from omninav.core.types import Observation


class GlobalState:
    IDLE = "idle"
    NAVIGATING = "navigating"
    SCANNING = "scanning"
    COMPLETE = "complete"


@ALGORITHM_REGISTRY.register("global_sequential")
class SequentialGlobalPlanner(GlobalPlannerBase):
    """Global planner that visits goals in a deterministic sequence."""

    ALGORITHM_TYPE = "global_sequential"

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self._waypoint_tolerance = float(cfg.get("waypoint_tolerance", 0.5))
        self._scan_at_goal = bool(cfg.get("scan_at_goal", True))
        self._scan_duration = float(cfg.get("scan_duration", 0.0))
        self._scan_angular_velocity = float(cfg.get("scan_angular_velocity", 1.0))
        self._state = GlobalState.IDLE
        self._goals = np.zeros((0, 3), dtype=np.float32)
        self._idx = 0
        self._scan_start_time: Optional[float] = None
        self._current_goal: Optional[np.ndarray] = None
        self._done = False
        self._last_dist = float("inf")

    def reset(self, task_info: Optional[Dict[str, Any]] = None) -> None:
        spec = task_info or {}
        goals = spec.get("goal_set", spec.get("waypoints", []))
        arr = np.asarray(goals, dtype=np.float32)
        if arr.size == 0:
            self._goals = np.zeros((0, 3), dtype=np.float32)
            self._current_goal = None
            self._state = GlobalState.COMPLETE
            self._done = True
            return
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        self._goals = arr[:, :3]
        self._idx = 0
        self._current_goal = self._goals[0].reshape(1, 3)
        self._done = False
        self._state = GlobalState.NAVIGATING
        self._scan_start_time = None

        self._waypoint_tolerance = float(spec.get("waypoint_tolerance", self._waypoint_tolerance))
        self._scan_at_goal = bool(spec.get("scan_at_goal", self._scan_at_goal))
        self._scan_duration = float(spec.get("scan_duration", self._scan_duration))
        self._scan_angular_velocity = float(spec.get("scan_angular_velocity", self._scan_angular_velocity))

    def step(self, obs: "Observation") -> np.ndarray:
        if self._done or self._current_goal is None:
            self._state = GlobalState.COMPLETE
            return np.zeros((1, 3), dtype=np.float32)

        pos = np.asarray(obs.get("robot_state", {}).get("position", np.zeros((1, 3))), dtype=np.float32)
        if pos.ndim == 1:
            pos = pos.reshape(1, 3)
        sim_time = float(obs.get("sim_time", 0.0))
        dist = float(np.linalg.norm(self._current_goal[0, :2] - pos[0, :2]))
        self._last_dist = dist

        if self._state == GlobalState.SCANNING:
            if self._scan_start_time is None:
                self._scan_start_time = sim_time
            if sim_time - self._scan_start_time >= self._scan_duration:
                self._advance()
            return np.zeros((pos.shape[0], 3), dtype=np.float32)

        if dist <= self._waypoint_tolerance:
            if self._scan_at_goal and self._scan_duration > 0.0:
                self._state = GlobalState.SCANNING
                self._scan_start_time = sim_time
            else:
                self._advance()
        return np.zeros((pos.shape[0], 3), dtype=np.float32)

    def _advance(self) -> None:
        self._idx += 1
        if self._idx >= self._goals.shape[0]:
            self._done = True
            self._state = GlobalState.COMPLETE
            self._current_goal = None
            return
        self._current_goal = self._goals[self._idx].reshape(1, 3)
        self._state = GlobalState.NAVIGATING
        self._scan_start_time = None

    def current_goal(self) -> Optional[np.ndarray]:
        return None if self._current_goal is None else self._current_goal.copy()

    def command_override(self) -> Optional[np.ndarray]:
        if self._state == GlobalState.SCANNING:
            return np.array([[0.0, 0.0, self._scan_angular_velocity]], dtype=np.float32)
        return None

    @property
    def is_done(self) -> bool:
        return self._done

    @property
    def info(self) -> Dict[str, Any]:
        return {
            "state": self._state,
            "current_goal_index": self._idx,
            "num_goals": int(self._goals.shape[0]),
            "distance_to_goal": self._last_dist,
            "is_done": self._done,
        }
