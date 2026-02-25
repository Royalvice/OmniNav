"""Map importer for OmniNav scene YAML configs."""

from __future__ import annotations

from typing import Dict, Any, List

from omninav.assets.map.importers.base import MapImporterBase


class SceneYamlMapImporter(MapImporterBase):
    """Normalize `configs/scene/*.yaml` into a map-oriented spec."""

    def load(self, scene_cfg: Dict[str, Any]) -> Dict[str, Any]:
        floors = scene_cfg.get("floors", None)
        if not floors:
            floors = [
                {
                    "id": "floor_0",
                    "name": str(scene_cfg.get("name", "floor_0")),
                    "obstacles": list(scene_cfg.get("obstacles", [])),
                    "spawn": scene_cfg.get("spawn", {}),
                }
            ]

        normalized_floors: List[Dict[str, Any]] = []
        for i, floor in enumerate(floors):
            floor_id = str(floor.get("id", f"floor_{i}"))
            normalized_floors.append(
                {
                    "id": floor_id,
                    "name": str(floor.get("name", floor_id)),
                    "origin": list(floor.get("origin", scene_cfg.get("navigation", {}).get("origin", [-8.0, -8.0]))),
                    "extent": list(floor.get("extent", scene_cfg.get("navigation", {}).get("extent", [16.0, 16.0]))),
                    "resolution": float(floor.get("resolution", scene_cfg.get("navigation", {}).get("resolution", 0.1))),
                    "obstacles": list(floor.get("obstacles", scene_cfg.get("obstacles", []))),
                    "spawn": floor.get("spawn", scene_cfg.get("spawn", {})),
                }
            )

        return {
            "name": str(scene_cfg.get("name", "scene")),
            "floors": normalized_floors,
            "connectors": list(scene_cfg.get("connectors", [])),
            "navigation": dict(scene_cfg.get("navigation", {})),
        }

