"""Facade to construct runtime MapService from scene config."""

from __future__ import annotations

from typing import Any
from omegaconf import DictConfig, OmegaConf

from omninav.assets.map.importers.scene_yaml import SceneYamlMapImporter
from omninav.assets.map.builders.occupancy_builder import OccupancyBuilder
from omninav.core.map.service import MapService


def build_map_service_from_scene_cfg(scene_cfg: DictConfig | dict[str, Any]) -> MapService:
    """Build a runtime map service from scene config."""
    if isinstance(scene_cfg, DictConfig):
        scene_dict = OmegaConf.to_container(scene_cfg, resolve=True)
    else:
        scene_dict = dict(scene_cfg)
    importer = SceneYamlMapImporter()
    spec = importer.load(scene_dict)
    nav_cfg = spec.get("navigation", {})
    builder = OccupancyBuilder(inflation_radius=float(nav_cfg.get("inflation_radius", 0.15)))
    map_set, connectors = builder.build(spec)
    return MapService(map_set=map_set, connectors=connectors)

