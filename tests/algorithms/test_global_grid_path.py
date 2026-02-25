"""Tests for grid-based global planner."""

from __future__ import annotations

import numpy as np
from omegaconf import OmegaConf

from omninav.algorithms.global_grid_path import GridPathGlobalPlanner
from omninav.core.map import build_map_service_from_scene_cfg
from omninav.core.types import Observation, RobotState


def _obs_at(x: float, y: float) -> Observation:
    return Observation(
        robot_state=RobotState(
            position=np.array([[x, y, 0.0]], dtype=np.float32),
            orientation=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            linear_velocity=np.zeros((1, 3), dtype=np.float32),
            angular_velocity=np.zeros((1, 3), dtype=np.float32),
            joint_positions=np.zeros((1, 4), dtype=np.float32),
            joint_velocities=np.zeros((1, 4), dtype=np.float32),
        ),
        sim_time=0.0,
        sensors={},
    )


def test_global_grid_path_basic_path():
    scene_cfg = OmegaConf.create(
        {
            "navigation": {"resolution": 0.2, "origin": [-2.0, -2.0], "extent": [4.0, 4.0], "inflation_radius": 0.0},
            "obstacles": [{"type": "box", "position": [0.0, 0.0, 0.5], "size": [0.4, 0.4, 1.0]}],
        }
    )
    ms = build_map_service_from_scene_cfg(scene_cfg)
    planner = GridPathGlobalPlanner(OmegaConf.create({"type": "global_grid_path", "waypoint_tolerance": 0.2}))
    planner.reset({"goal_set": [[1.5, 1.5, 0.0]], "map_service": ms})
    planner.step(_obs_at(-1.5, -1.5))

    path = planner.current_path()
    assert path is not None
    assert path.ndim == 3
    assert path.shape[1] >= 2
    assert not planner.is_done

