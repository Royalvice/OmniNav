"""
Inspection Planner — Built-in inspection route planner.

Features:
- Waypoint ordering (greedy nearest-neighbor or TSP 2-opt)
- Point scan at each waypoint (rotation in place)
- Dynamic waypoint insertion during task
"""

from typing import Dict, Any, List, Optional, TYPE_CHECKING
import numpy as np
from omegaconf import DictConfig

from omninav.algorithms.base import AlgorithmBase
from omninav.core.registry import ALGORITHM_REGISTRY

if TYPE_CHECKING:
    from omninav.core.types import Observation


class InspectionState:
    """Internal state for the inspection planner."""
    NAVIGATING = "navigating"
    SCANNING = "scanning"
    COMPLETE = "complete"


@ALGORITHM_REGISTRY.register("inspection_planner")
class InspectionPlanner(AlgorithmBase):
    """
    Built-in inspection route planner.

    Manages a list of inspection waypoints and orchestrates:
    1. Navigation to each waypoint in optimized order
    2. Point scan (360° rotation) at each waypoint
    3. Transition to next waypoint until all visited

    Config example:
        type: inspection_planner
        planning_strategy: greedy  # or "tsp_2opt"
        scan_at_waypoint: true
        scan_duration: 3.0
        waypoint_tolerance: 0.5
        scan_angular_velocity: 1.0
    """

    ALGORITHM_TYPE = "inspection_planner"

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        # Planning parameters
        self._strategy = cfg.get("planning_strategy", "greedy")
        self._scan_at_waypoint = cfg.get("scan_at_waypoint", True)
        self._scan_duration = cfg.get("scan_duration", 3.0)
        self._waypoint_tolerance = cfg.get("waypoint_tolerance", 0.5)
        self._scan_angular_velocity = cfg.get("scan_angular_velocity", 1.0)

        # State
        self._waypoints: List[np.ndarray] = []
        self._visit_order: List[int] = []
        self._current_wp_idx: int = 0
        self._state: str = InspectionState.COMPLETE
        self._scan_elapsed: float = 0.0
        self._scan_start_time: Optional[float] = None
        self._current_waypoint: Optional[np.ndarray] = None

    @property
    def current_waypoint(self) -> Optional[np.ndarray]:
        """Get current target waypoint (used by pipeline's local planner)."""
        return self._current_waypoint

    @property
    def state(self) -> str:
        """Get current inspection state."""
        return self._state

    @property
    def progress(self) -> float:
        """Get inspection progress (0.0 to 1.0)."""
        if not self._visit_order:
            return 1.0
        return self._current_wp_idx / len(self._visit_order)

    def reset(self, task_info: Optional[Dict[str, Any]] = None) -> None:
        """
        Reset planner with new waypoints.

        Args:
            task_info: Must contain 'waypoints' key with list of (x, y, z) positions.
        """
        self._waypoints = []
        self._visit_order = []
        self._current_wp_idx = 0
        self._state = InspectionState.COMPLETE
        self._scan_elapsed = 0.0
        self._scan_start_time = None
        self._current_waypoint = None

        if task_info and "waypoints" in task_info:
            wps = task_info["waypoints"]
            self._waypoints = [np.array(wp, dtype=np.float32) for wp in wps]
            self._visit_order = self._plan_order(self._waypoints)
            if self._visit_order:
                self._state = InspectionState.NAVIGATING
                self._current_waypoint = self._waypoints[self._visit_order[0]]

    def add_waypoint(self, position: np.ndarray) -> None:
        """
        Dynamically insert a new waypoint.

        Args:
            position: (3,) world position of new waypoint
        """
        idx = len(self._waypoints)
        self._waypoints.append(np.array(position, dtype=np.float32))
        # Insert at end of remaining visit order
        self._visit_order.append(idx)

    def step(self, obs: "Observation") -> np.ndarray:
        """
        Execute one inspection planner step.

        Returns cmd_vel:
        - NAVIGATING: zeros (expects local planner to handle via pipeline)
        - SCANNING: pure rotation [0, 0, wz]
        - COMPLETE: zeros

        The pipeline reads self.current_waypoint and feeds it to local planner.
        """
        if self._state == InspectionState.COMPLETE:
            return np.zeros(3, dtype=np.float32)

        # Get robot position
        robot_state = obs.get("robot_state", {})
        pos = np.asarray(robot_state.get("position", np.zeros((1, 3))))
        if pos.ndim == 2:
            pos = pos[0]
        sim_time = obs.get("sim_time", 0.0)

        if self._state == InspectionState.NAVIGATING:
            if self._current_waypoint is None:
                self._advance_to_next()
                return np.zeros(3, dtype=np.float32)

            # Check if reached current waypoint
            dist = np.linalg.norm(self._current_waypoint[:2] - pos[:2])
            if dist < self._waypoint_tolerance:
                if self._scan_at_waypoint:
                    self._state = InspectionState.SCANNING
                    self._scan_start_time = sim_time
                    self._scan_elapsed = 0.0
                else:
                    self._advance_to_next()

            # Return zeros — local planner handles navigation via current_waypoint
            return np.zeros(3, dtype=np.float32)

        elif self._state == InspectionState.SCANNING:
            # Rotate in place for scan_duration
            if self._scan_start_time is not None:
                self._scan_elapsed = sim_time - self._scan_start_time

            if self._scan_elapsed >= self._scan_duration:
                self._advance_to_next()
                return np.zeros(3, dtype=np.float32)

            # Pure rotation
            return np.array([0.0, 0.0, self._scan_angular_velocity], dtype=np.float32)

        return np.zeros(3, dtype=np.float32)

    def _advance_to_next(self) -> None:
        """Advance to next waypoint in visit order."""
        self._current_wp_idx += 1
        if self._current_wp_idx >= len(self._visit_order):
            self._state = InspectionState.COMPLETE
            self._current_waypoint = None
        else:
            wp_idx = self._visit_order[self._current_wp_idx]
            self._current_waypoint = self._waypoints[wp_idx]
            self._state = InspectionState.NAVIGATING

    def _plan_order(self, waypoints: List[np.ndarray]) -> List[int]:
        """
        Plan visit order for waypoints.

        Args:
            waypoints: List of waypoint positions

        Returns:
            List of indices in visit order
        """
        if len(waypoints) <= 1:
            return list(range(len(waypoints)))

        if self._strategy == "greedy":
            return self._greedy_order(waypoints)
        elif self._strategy == "tsp_2opt":
            return self._tsp_2opt_order(waypoints)
        else:
            return list(range(len(waypoints)))

    @staticmethod
    def _greedy_order(waypoints: List[np.ndarray]) -> List[int]:
        """Greedy nearest-neighbor ordering."""
        n = len(waypoints)
        visited = [False] * n
        order = [0]
        visited[0] = True

        for _ in range(n - 1):
            current = waypoints[order[-1]]
            best_idx = -1
            best_dist = float('inf')
            for j in range(n):
                if not visited[j]:
                    d = np.linalg.norm(waypoints[j] - current)
                    if d < best_dist:
                        best_dist = d
                        best_idx = j
            visited[best_idx] = True
            order.append(best_idx)

        return order

    @staticmethod
    def _tsp_2opt_order(waypoints: List[np.ndarray]) -> List[int]:
        """TSP 2-opt improvement over greedy."""
        # Start with greedy order
        order = InspectionPlanner._greedy_order(waypoints)

        def route_distance(route):
            d = 0.0
            for i in range(len(route) - 1):
                d += np.linalg.norm(waypoints[route[i]] - waypoints[route[i + 1]])
            return d

        improved = True
        while improved:
            improved = False
            for i in range(1, len(order) - 1):
                for j in range(i + 1, len(order)):
                    new_order = order[:i] + order[i:j + 1][::-1] + order[j + 1:]
                    if route_distance(new_order) < route_distance(order):
                        order = new_order
                        improved = True

        return order

    @property
    def is_done(self) -> bool:
        return self._state == InspectionState.COMPLETE

    @property
    def info(self) -> Dict[str, Any]:
        return {
            "state": self._state,
            "current_wp_idx": self._current_wp_idx,
            "total_waypoints": len(self._waypoints),
            "progress": self.progress,
            "current_waypoint": self._current_waypoint.tolist() if self._current_waypoint is not None else None,
        }
