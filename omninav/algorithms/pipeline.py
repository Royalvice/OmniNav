"""
Algorithm Pipeline — Compose GlobalPlanner + LocalPlanner into a pipeline.

Usage:
    pipeline = AlgorithmPipeline(cfg)
    cmd_vel = pipeline.step(obs)  # GlobalPlanner → waypoint → LocalPlanner → cmd_vel
"""

from typing import Dict, Any, Optional, TYPE_CHECKING
import numpy as np
from omegaconf import DictConfig

from omninav.algorithms.base import AlgorithmBase
from omninav.core.registry import ALGORITHM_REGISTRY

if TYPE_CHECKING:
    from omninav.core.types import Observation


@ALGORITHM_REGISTRY.register("algorithm_pipeline")
class AlgorithmPipeline(AlgorithmBase):
    """
    Compose a global planner and a local planner into a unified pipeline.

    The global planner provides waypoints/subgoals, and the local planner
    converts them to cmd_vel while avoiding obstacles.
    """

    ALGORITHM_TYPE = "algorithm_pipeline"

    def __init__(self, cfg: DictConfig):
        """
        Initialize pipeline.

        Args:
            cfg: Pipeline configuration with 'global_planner' and 'local_planner' sub-configs
        """
        super().__init__(cfg)

        # Build sub-planners via registry
        self._global_planner: AlgorithmBase = ALGORITHM_REGISTRY.build(cfg.global_planner)
        self._local_planner: AlgorithmBase = ALGORITHM_REGISTRY.build(cfg.local_planner)

    @property
    def global_planner(self) -> AlgorithmBase:
        """Access the global planner."""
        return self._global_planner

    @property
    def local_planner(self) -> AlgorithmBase:
        """Access the local planner."""
        return self._local_planner

    def reset(self, task_info: Optional[Dict[str, Any]] = None) -> None:
        """Reset both planners."""
        self._global_planner.reset(task_info)
        self._local_planner.reset(task_info)

    def step(self, obs: "Observation") -> np.ndarray:
        """
        Execute one pipeline step:
        1. Global planner produces a waypoint/subgoal
        2. Local planner navigates toward it while avoiding obstacles

        Args:
            obs: Observation TypedDict

        Returns:
            cmd_vel: [vx, vy, wz] velocity command
        """
        # Global planner step — may return cmd_vel directly or update internal waypoint
        _ = self._global_planner.step(obs)

        # If the global planner provides a waypoint (non-zero output),
        # inject it as goal_position for the local planner
        local_obs = dict(obs)  # shallow copy
        if hasattr(self._global_planner, 'current_waypoint') and self._global_planner.current_waypoint is not None:
            waypoint = np.asarray(self._global_planner.current_waypoint, dtype=np.float32)
            if waypoint.ndim == 1:
                waypoint = waypoint.reshape(1, -1)
            local_obs['goal_position'] = waypoint

        # Local planner produces the final cmd_vel
        cmd_vel = self._local_planner.step(local_obs)
        return cmd_vel

    @property
    def is_done(self) -> bool:
        """Pipeline is done when the global planner signals completion."""
        return self._global_planner.is_done

    @property
    def info(self) -> Dict[str, Any]:
        """Merge info from both planners."""
        return {
            "global_planner": self._global_planner.info,
            "local_planner": self._local_planner.info,
        }
