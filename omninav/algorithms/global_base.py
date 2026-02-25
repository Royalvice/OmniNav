"""Base interface for global planners."""

from __future__ import annotations

from abc import abstractmethod
from typing import Dict, Any, Optional, TYPE_CHECKING
import numpy as np

from omninav.algorithms.base import AlgorithmBase

if TYPE_CHECKING:
    from omninav.core.types import Observation


class GlobalPlannerBase(AlgorithmBase):
    """Global planner produces subgoals for local planner."""

    @abstractmethod
    def current_goal(self) -> Optional[np.ndarray]:
        """Return current goal position in shape (B, 3) or None."""
        pass

    def command_override(self) -> Optional[np.ndarray]:
        """
        Optional direct command override (B, 3), e.g., scan-in-place behavior.
        """
        return None

    @property
    def info(self) -> Dict[str, Any]:
        return {}
