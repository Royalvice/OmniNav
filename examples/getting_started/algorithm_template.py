"""
Template APIs for adding new planning algorithms.

This file intentionally contains two templates:
1. Global planner template
2. Local planner template

Copy into omninav/algorithms/ and register with ALGORITHM_REGISTRY.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, TYPE_CHECKING
import numpy as np
from omegaconf import DictConfig

from omninav.algorithms.global_base import GlobalPlannerBase
from omninav.algorithms.local_planner import LocalPlannerBase
from omninav.core.registry import ALGORITHM_REGISTRY
from omninav.core.types import validate_batch_shape

if TYPE_CHECKING:
    from omninav.core.types import Observation


@ALGORITHM_REGISTRY.register("my_global_planner_template")
class MyGlobalPlannerTemplate(GlobalPlannerBase):
    """Minimal global planner template."""

    ALGORITHM_TYPE = "my_global_planner_template"

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self._goal = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        self._done = False

    def reset(self, task_info: Optional[Dict[str, Any]] = None) -> None:
        _ = task_info
        self._done = False

    def step(self, obs: "Observation") -> np.ndarray:
        pos = np.asarray(obs["robot_state"]["position"], dtype=np.float32)
        validate_batch_shape(pos, "MyGlobalPlannerTemplate.position", (3,))
        return np.zeros((pos.shape[0], 3), dtype=np.float32)

    def current_goal(self) -> Optional[np.ndarray]:
        return self._goal.copy()

    @property
    def is_done(self) -> bool:
        return self._done


@ALGORITHM_REGISTRY.register("my_local_planner_template")
class MyLocalPlannerTemplate(LocalPlannerBase):
    """Minimal local planner template."""

    ALGORITHM_TYPE = "my_local_planner_template"

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self._done = False

    def reset(self, task_info: Optional[Dict[str, Any]] = None) -> None:
        _ = task_info
        self._done = False

    def navigate_to(self, obs: "Observation", target: np.ndarray) -> np.ndarray:
        _ = (obs, target)
        return np.zeros((1, 3), dtype=np.float32)

    def step(self, obs: "Observation") -> np.ndarray:
        position = np.asarray(obs["robot_state"]["position"], dtype=np.float32)
        validate_batch_shape(position, "MyLocalPlannerTemplate.position", (3,))
        # Replace with real obstacle-avoidance control logic.
        return np.zeros((position.shape[0], 3), dtype=np.float32)

    @property
    def is_done(self) -> bool:
        return self._done
