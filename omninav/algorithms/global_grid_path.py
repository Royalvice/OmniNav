"""Grid-based global planner with occupancy-map A* routing."""

from __future__ import annotations

from heapq import heappush, heappop
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from omegaconf import DictConfig

from omninav.algorithms.global_base import GlobalPlannerBase
from omninav.core.registry import ALGORITHM_REGISTRY


@ALGORITHM_REGISTRY.register("global_grid_path")
class GridPathGlobalPlanner(GlobalPlannerBase):
    """Plan and follow waypoint routes on occupancy grids."""

    ALGORITHM_TYPE = "global_grid_path"

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self._waypoint_tolerance = float(cfg.get("waypoint_tolerance", 0.25))
        self._resample_step = float(cfg.get("resample_step", 0.2))
        self._allow_diagonal = bool(cfg.get("allow_diagonal", True))
        self._goals = np.zeros((0, 3), dtype=np.float32)
        self._goal_idx = 0
        self._current_goal: Optional[np.ndarray] = None
        self._current_path = np.zeros((0, 3), dtype=np.float32)
        self._done = False
        self._map_service = None
        self._floor_id = "floor_0"
        self._last_dist = float("inf")
        self._replans = 0

    def reset(self, task_info: Optional[Dict[str, Any]] = None) -> None:
        spec = task_info or {}
        arr = np.asarray(spec.get("goal_set", spec.get("waypoints", [])), dtype=np.float32)
        if arr.size == 0:
            self._goals = np.zeros((0, 3), dtype=np.float32)
            self._goal_idx = 0
            self._current_goal = None
            self._current_path = np.zeros((0, 3), dtype=np.float32)
            self._done = True
            return
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        self._goals = arr[:, :3]
        self._goal_idx = 0
        self._current_goal = self._goals[0].reshape(1, 3)
        self._current_path = np.zeros((0, 3), dtype=np.float32)
        self._done = False
        self._replans = 0
        self._waypoint_tolerance = float(spec.get("waypoint_tolerance", self._waypoint_tolerance))
        self._map_service = spec.get("map_service", None)
        self._floor_id = "floor_0"

    def step(self, obs: Dict[str, Any]) -> np.ndarray:
        if self._done or self._current_goal is None:
            return np.zeros((1, 3), dtype=np.float32)

        pos = np.asarray(obs.get("robot_state", {}).get("position", np.zeros((1, 3))), dtype=np.float32)
        if pos.ndim == 2:
            pos = pos[0]
        self._last_dist = float(np.linalg.norm(self._current_goal[0, :2] - pos[:2]))

        if self._map_service is not None:
            self._floor_id = self._map_service.find_floor_by_xy(pos[:2], preferred_floor=self._floor_id)
            if self._current_path.shape[0] < 2:
                self._current_path = self._plan_path(pos[:2], self._current_goal[0, :2], self._floor_id)
                self._replans += 1

        if self._last_dist <= self._waypoint_tolerance:
            self._advance_goal()
            if not self._done and self._map_service is not None:
                self._current_path = self._plan_path(pos[:2], self._current_goal[0, :2], self._floor_id)
                self._replans += 1
        return np.zeros((1, 3), dtype=np.float32)

    def _advance_goal(self) -> None:
        self._goal_idx += 1
        if self._goal_idx >= self._goals.shape[0]:
            self._done = True
            self._current_goal = None
            self._current_path = np.zeros((0, 3), dtype=np.float32)
            return
        self._current_goal = self._goals[self._goal_idx].reshape(1, 3)

    def _plan_path(self, start_xy: np.ndarray, goal_xy: np.ndarray, floor_id: str) -> np.ndarray:
        if self._map_service is None:
            return np.array([[start_xy[0], start_xy[1], 0.0], [goal_xy[0], goal_xy[1], 0.0]], dtype=np.float32)

        occ = self._map_service.get_map(floor_id)
        sx, sy = self._map_service.world_to_grid(floor_id, start_xy)
        gx, gy = self._map_service.world_to_grid(floor_id, goal_xy)

        path_cells = self._astar(occ.grid, (sx, sy), (gx, gy), self._allow_diagonal)
        if not path_cells:
            return np.array([[start_xy[0], start_xy[1], 0.0], [goal_xy[0], goal_xy[1], 0.0]], dtype=np.float32)

        pts = []
        for cx, cy in path_cells:
            wxy = self._map_service.grid_to_world(floor_id, cx, cy)
            pts.append([float(wxy[0]), float(wxy[1]), 0.0])
        return self._resample(np.asarray(pts, dtype=np.float32), self._resample_step)

    @staticmethod
    def _astar(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int], diagonal: bool) -> List[Tuple[int, int]]:
        w = int(grid.shape[1])
        h = int(grid.shape[0])
        if not (0 <= start[0] < w and 0 <= start[1] < h and 0 <= goal[0] < w and 0 <= goal[1] < h):
            return []

        neigh = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if diagonal:
            neigh.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])

        def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
            return float(np.hypot(a[0] - b[0], a[1] - b[1]))

        frontier: list[tuple[float, Tuple[int, int]]] = []
        heappush(frontier, (0.0, start))
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_cost: Dict[Tuple[int, int], float] = {start: 0.0}

        while frontier:
            _, current = heappop(frontier)
            if current == goal:
                break
            for dx, dy in neigh:
                nx = current[0] + dx
                ny = current[1] + dy
                if nx < 0 or nx >= w or ny < 0 or ny >= h:
                    continue
                if int(grid[ny, nx]) > 0:
                    continue
                step = float(np.hypot(dx, dy))
                ng = g_cost[current] + step
                nxt = (nx, ny)
                if nxt not in g_cost or ng < g_cost[nxt]:
                    g_cost[nxt] = ng
                    came_from[nxt] = current
                    f = ng + heuristic(nxt, goal)
                    heappush(frontier, (f, nxt))

        if goal not in came_from and goal != start:
            return []

        out = [goal]
        cur = goal
        while cur != start:
            cur = came_from[cur]
            out.append(cur)
        out.reverse()
        return out

    @staticmethod
    def _resample(path_xyz: np.ndarray, step: float) -> np.ndarray:
        if path_xyz.shape[0] <= 2 or step <= 1e-6:
            return path_xyz
        out = [path_xyz[0]]
        accum = 0.0
        for i in range(1, path_xyz.shape[0]):
            seg = path_xyz[i] - path_xyz[i - 1]
            d = float(np.linalg.norm(seg[:2]))
            accum += d
            if accum >= step or i == path_xyz.shape[0] - 1:
                out.append(path_xyz[i])
                accum = 0.0
        return np.asarray(out, dtype=np.float32)

    def current_goal(self) -> Optional[np.ndarray]:
        return None if self._current_goal is None else self._current_goal.copy()

    def current_path(self) -> Optional[np.ndarray]:
        if self._current_path.size == 0:
            return None
        return self._current_path.copy().reshape(1, self._current_path.shape[0], 3)

    @property
    def is_done(self) -> bool:
        return self._done

    @property
    def info(self) -> Dict[str, Any]:
        return {
            "current_goal_index": int(self._goal_idx),
            "num_goals": int(self._goals.shape[0]),
            "distance_to_goal": float(self._last_dist),
            "current_floor_id": self._floor_id,
            "path_points": int(self._current_path.shape[0]),
            "replans": int(self._replans),
            "is_done": bool(self._done),
        }

