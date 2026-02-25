"""Base interfaces for map importers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any


class MapImporterBase(ABC):
    """Convert source scene assets into a normalized map specification."""

    @abstractmethod
    def load(self, scene_cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Load and normalize map-ready scene description."""
        raise NotImplementedError

