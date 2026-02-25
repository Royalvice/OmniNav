"""Tests for algorithm pipeline with global+local decoupling."""

from __future__ import annotations

import numpy as np
from omegaconf import OmegaConf

from omninav.core.types import Observation, RobotState


def _make_obs(pos=(0, 0, 0), quat=(1, 0, 0, 0), sim_time=0.0, lidar_ranges=None):
    obs = Observation(
        robot_state=RobotState(
            position=np.array([pos], dtype=np.float32),
            orientation=np.array([quat], dtype=np.float32),
            linear_velocity=np.zeros((1, 3), dtype=np.float32),
            angular_velocity=np.zeros((1, 3), dtype=np.float32),
            joint_positions=np.zeros((1, 12), dtype=np.float32),
            joint_velocities=np.zeros((1, 12), dtype=np.float32),
        ),
        sim_time=sim_time,
        sensors={},
    )
    if lidar_ranges is not None:
        obs["sensors"]["lidar"] = {"ranges": np.array([lidar_ranges], dtype=np.float32)}
    return obs


def test_registry_entries_exist():
    import omninav.algorithms  # noqa: F401
    from omninav.core.registry import ALGORITHM_REGISTRY

    assert "algorithm_pipeline" in ALGORITHM_REGISTRY
    assert "dwa_planner" in ALGORITHM_REGISTRY
    assert "global_sequential" in ALGORITHM_REGISTRY
    assert "global_route_opt" in ALGORITHM_REGISTRY


def test_global_sequential_done_when_empty():
    from omninav.algorithms.global_sequential import SequentialGlobalPlanner

    planner = SequentialGlobalPlanner(OmegaConf.create({"type": "global_sequential"}))
    planner.reset({"goal_set": []})
    cmd = planner.step(_make_obs())
    assert planner.is_done
    assert cmd.shape == (1, 3)
    assert np.allclose(cmd, 0.0)


def test_global_route_opt_reorders():
    from omninav.algorithms.global_route_opt import RouteOptimizedGlobalPlanner

    planner = RouteOptimizedGlobalPlanner(OmegaConf.create({"type": "global_route_opt", "strategy": "greedy"}))
    planner.reset({"goal_set": [[0, 0, 0], [10, 0, 0], [1, 0, 0]]})
    g = planner.current_goal()
    assert g is not None
    assert g.shape == (1, 3)


def test_dwa_planner_moves_toward_goal():
    from omninav.algorithms.local_planner import DWAPlanner

    planner = DWAPlanner(OmegaConf.create({"type": "dwa_planner", "max_speed": 1.0, "goal_tolerance": 0.5}))
    planner.reset()
    obs = _make_obs(pos=(0, 0, 0))
    obs["goal_position"] = np.array([[5.0, 0.0, 0.0]], dtype=np.float32)
    cmd = planner.step(obs)
    assert cmd.shape == (1, 3)
    assert cmd[0, 0] > 0


def test_pipeline_composes_global_and_local():
    from omninav.algorithms.pipeline import AlgorithmPipeline

    cfg = OmegaConf.create(
        {
            "type": "algorithm_pipeline",
            "global_planner": {"type": "global_sequential", "scan_at_goal": False, "waypoint_tolerance": 0.5},
            "local_planner": {"type": "dwa_planner", "goal_tolerance": 0.5, "max_speed": 0.6},
        }
    )
    pipe = AlgorithmPipeline(cfg)
    pipe.reset({"goal_set": [[5.0, 0.0, 0.0]]})

    cmd = pipe.step(_make_obs(pos=(0, 0, 0), lidar_ranges=[3, 3, 3]))
    assert cmd.shape == (1, 3)
    assert cmd[0, 0] > 0
    assert not pipe.is_done

    pipe.step(_make_obs(pos=(5.0, 0.0, 0.0), sim_time=10.0, lidar_ranges=[3, 3, 3]))
    assert pipe.is_done
