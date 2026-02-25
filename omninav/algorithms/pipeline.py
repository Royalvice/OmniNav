"""Algorithm pipeline that composes global and local planners."""

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
        self._task_spec: Dict[str, Any] = {}

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
        self._task_spec = task_info or {}
        self._global_planner.reset(self._task_spec)
        self._local_planner.reset(self._task_spec)

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
        _ = self._global_planner.step(obs)
        local_obs = dict(obs)  # shallow copy
        if hasattr(self._global_planner, "current_goal"):
            goal = self._global_planner.current_goal()
            if goal is not None:
                waypoint = np.asarray(goal, dtype=np.float32)
                if waypoint.ndim == 1:
                    waypoint = waypoint.reshape(1, -1)
                local_obs["goal_position"] = waypoint
        if hasattr(self._global_planner, "current_path"):
            path = self._global_planner.current_path()
            if path is not None:
                path_points = np.asarray(path, dtype=np.float32)
                if path_points.ndim == 2:
                    path_points = path_points.reshape(1, path_points.shape[0], path_points.shape[1])
                local_obs["path_points"] = path_points
                local_obs["local_path_points"] = path_points

        if hasattr(self._global_planner, "command_override"):
            override = self._global_planner.command_override()
            if override is not None:
                ov = np.asarray(override, dtype=np.float32)
                if ov.ndim == 1:
                    ov = ov.reshape(1, -1)
                return ov

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
