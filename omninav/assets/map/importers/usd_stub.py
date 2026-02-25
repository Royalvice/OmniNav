"""Placeholder USD map importer.

The real implementation should follow Genesis USD loading semantics.
"""

from __future__ import annotations

from typing import Dict, Any

from omninav.assets.map.importers.base import MapImporterBase


class USDMapImporterStub(MapImporterBase):
    """Interface-compatible placeholder for future USD map import."""

    def load(self, scene_cfg: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError(
            "USD map import is not implemented yet. "
            "Use SceneYamlMapImporter for now and follow Genesis USD examples "
            "when implementing this importer."
        )

