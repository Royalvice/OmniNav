"""Tests for map build facade and runtime service."""

from __future__ import annotations

import numpy as np
from omegaconf import OmegaConf

from omninav.core.map import build_map_service_from_scene_cfg


def test_build_map_service_from_scene_cfg_single_floor():
    scene_cfg = OmegaConf.create(
        {
            "name": "test_scene",
            "navigation": {"resolution": 0.2, "origin": [-2.0, -2.0], "extent": [4.0, 4.0]},
            "obstacles": [
                {"type": "box", "position": [0.0, 0.0, 0.5], "size": [1.0, 1.0, 1.0]},
            ],
        }
    )
    ms = build_map_service_from_scene_cfg(scene_cfg)
    m = ms.get_default_map()
    assert m.grid.ndim == 2
    assert m.grid.shape[0] > 0 and m.grid.shape[1] > 0
    gx, gy = ms.world_to_grid(m.floor_id, np.array([0.0, 0.0], dtype=np.float32))
    assert int(m.grid[gy, gx]) == 1


def test_map_service_floor_lookup():
    scene_cfg = OmegaConf.create(
        {
            "floors": [
                {
                    "id": "floor_0",
                    "origin": [-1.0, -1.0],
                    "extent": [2.0, 2.0],
                    "resolution": 0.1,
                    "obstacles": [],
                },
                {
                    "id": "floor_1",
                    "origin": [9.0, 9.0],
                    "extent": [2.0, 2.0],
                    "resolution": 0.1,
                    "obstacles": [],
                },
            ]
        }
    )
    ms = build_map_service_from_scene_cfg(scene_cfg)
    assert ms.find_floor_by_xy(np.array([0.0, 0.0], dtype=np.float32)) == "floor_0"
    assert ms.find_floor_by_xy(np.array([10.0, 10.0], dtype=np.float32)) == "floor_1"

