"""Route-optimized global planner built on sequential planner."""

from __future__ import annotations

from typing import Dict, Any, Optional, List
import numpy as np
from omegaconf import DictConfig

from omninav.algorithms.global_sequential import SequentialGlobalPlanner
from omninav.core.registry import ALGORITHM_REGISTRY


@ALGORITHM_REGISTRY.register("global_route_opt")
class RouteOptimizedGlobalPlanner(SequentialGlobalPlanner):
    """Global planner with greedy/TSP-like reordering."""

    ALGORITHM_TYPE = "global_route_opt"

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self._strategy = str(cfg.get("strategy", "tsp_2opt"))

    def reset(self, task_info: Optional[Dict[str, Any]] = None) -> None:
        spec = dict(task_info or {})
        goals = np.asarray(spec.get("goal_set", spec.get("waypoints", [])), dtype=np.float32)
        if goals.size != 0:
            if goals.ndim == 1:
                goals = goals.reshape(1, -1)
            goals = goals[:, :3]
            order = self._plan_order(goals)
            spec["goal_set"] = goals[order].tolist()
        super().reset(spec)

    def _plan_order(self, goals: np.ndarray) -> List[int]:
        if goals.shape[0] <= 1:
            return list(range(goals.shape[0]))
        if self._strategy == "greedy":
            return self._greedy_order(goals)
        return self._tsp_2opt(goals)

    @staticmethod
    def _greedy_order(goals: np.ndarray) -> List[int]:
        n = goals.shape[0]
        visited = [False] * n
        order = [0]
        visited[0] = True
        for _ in range(n - 1):
            cur = goals[order[-1]]
            best = None
            best_dist = float("inf")
            for j in range(n):
                if visited[j]:
                    continue
                d = float(np.linalg.norm(goals[j, :2] - cur[:2]))
                if d < best_dist:
                    best = j
                    best_dist = d
            visited[int(best)] = True
            order.append(int(best))
        return order

    @classmethod
    def _tsp_2opt(cls, goals: np.ndarray) -> List[int]:
        order = cls._greedy_order(goals)

        def route_distance(route: List[int]) -> float:
            d = 0.0
            for i in range(len(route) - 1):
                d += float(np.linalg.norm(goals[route[i], :2] - goals[route[i + 1], :2]))
            return d

        improved = True
        while improved:
            improved = False
            base = route_distance(order)
            for i in range(1, len(order) - 1):
                for j in range(i + 1, len(order)):
                    cand = order[:i] + order[i : j + 1][::-1] + order[j + 1 :]
                    cd = route_distance(cand)
                    if cd + 1e-8 < base:
                        order = cand
                        base = cd
                        improved = True
        return order
