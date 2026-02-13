"""
Inspection Task — Evaluation task for inspection missions.

Defines waypoints, inspection zones, time budgets, and coverage requirements.
"""

from typing import Dict, Any, List, Optional, TYPE_CHECKING
import numpy as np
from omegaconf import DictConfig

from omninav.evaluation.base import TaskBase, MetricBase
from omninav.core.types import TaskResult
from omninav.core.lifecycle import LifecycleState
from omninav.core.registry import TASK_REGISTRY

if TYPE_CHECKING:
    from omninav.core.types import Observation, Action


@TASK_REGISTRY.register("inspection")
class InspectionTask(TaskBase):
    """
    Inspection evaluation task.

    Evaluates a robot's ability to visit designated waypoints within a
    time budget while maintaining coverage and detecting anomalies.

    Config example (configs/task/inspection.yaml):
        type: inspection
        waypoints: [[1,0,0], [3,2,0], [5,0,0]]
        time_budget: 120.0
        coverage_requirement: 0.9
        waypoint_tolerance: 0.5
    """

    TASK_TYPE = "inspection"

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        # Task parameters
        self._waypoints: List[np.ndarray] = [
            np.array(wp, dtype=np.float32)
            for wp in cfg.get("waypoints", [])
        ]
        self._time_budget = cfg.get("time_budget", 120.0)
        self._coverage_requirement = cfg.get("coverage_requirement", 0.9)
        self._waypoint_tolerance = cfg.get("waypoint_tolerance", 0.5)

        # State
        self._visited: List[bool] = []
        self._step_count: int = 0
        self._trajectory: List[np.ndarray] = []
        self._start_time: float = 0.0
        self._current_time: float = 0.0
        self._collision_count: int = 0
        self._success: bool = False

    def reset(self) -> Dict[str, Any]:
        """Reset task and return task info with waypoints."""
        self._visited = [False] * len(self._waypoints)
        self._step_count = 0
        self._trajectory = []
        self._start_time = 0.0
        self._current_time = 0.0
        self._collision_count = 0
        self._success = False

        # Reset all metrics
        for metric in self.metrics:
            metric.reset()

        self._transition_to(LifecycleState.READY)

        return {
            "waypoints": [wp.tolist() for wp in self._waypoints],
            "time_budget": self._time_budget,
            "coverage_requirement": self._coverage_requirement,
        }

    def step(self, obs: "Observation", action: Optional["Action"] = None) -> None:
        """Record observation and check waypoint visits."""
        self._step_count += 1
        self._current_time = obs.get("sim_time", 0.0)

        # Extract robot position
        robot_state = obs.get("robot_state", {})
        pos = np.asarray(robot_state.get("position", np.zeros((1, 3))))
        if pos.ndim == 2:
            pos = pos[0]

        self._trajectory.append(pos.copy())

        # Check waypoint visits
        for i, wp in enumerate(self._waypoints):
            if not self._visited[i]:
                dist = np.linalg.norm(wp[:2] - pos[:2])
                if dist < self._waypoint_tolerance:
                    self._visited[i] = True

        # Update metrics
        for metric in self.metrics:
            metric.update(obs, action)

    def is_terminated(self, obs: "Observation") -> np.ndarray:
        """Check if task is terminated (all visited or timeout)."""
        terminated = False
        # Check time budget
        elapsed = obs.get("sim_time", 0.0) - self._start_time
        if elapsed >= self._time_budget:
            terminated = True

        # Check if all waypoints visited
        if not terminated and all(self._visited):
            self._success = True
            terminated = True

        robot_state = obs.get("robot_state", {})
        pos = np.asarray(robot_state.get("position", np.zeros((1, 3))))
        batch_size = int(pos.shape[0]) if pos.ndim >= 2 else 1
        return np.full((batch_size,), terminated, dtype=bool)

    def compute_result(self) -> TaskResult:
        """Compute inspection task result."""
        # Coverage rate
        visited_count = sum(self._visited)
        total_count = max(len(self._waypoints), 1)
        coverage = visited_count / total_count

        # Path length
        path_length = 0.0
        for i in range(1, len(self._trajectory)):
            path_length += np.linalg.norm(self._trajectory[i] - self._trajectory[i - 1])

        # Elapsed time
        elapsed = self._current_time - self._start_time

        # Success based on coverage requirement
        success = coverage >= self._coverage_requirement

        # Compute all registered metrics
        metric_values = {}
        for metric in self.metrics:
            metric_values[metric.METRIC_NAME] = metric.compute()

        # Add built-in metrics
        metric_values.update({
            "coverage_rate": coverage,
            "path_length": path_length,
            "elapsed_time": elapsed,
            "step_count": float(self._step_count),
            "waypoints_visited": float(visited_count),
            "collision_count": float(self._collision_count),
        })

        return TaskResult(
            success=success,
            episode_length=self._step_count,
            elapsed_time=elapsed,
            metrics=metric_values,
            info={
                "trajectory": [p.tolist() for p in self._trajectory[-10:]],  # last 10 points
                "visited": self._visited.copy(),
            },
        )

    @property
    def coverage(self) -> float:
        """Current coverage rate (0.0 - 1.0)."""
        if not self._waypoints:
            return 1.0
        return sum(self._visited) / len(self._waypoints)

    @property
    def remaining_waypoints(self) -> int:
        """Number of unvisited waypoints."""
        return sum(1 for v in self._visited if not v)
