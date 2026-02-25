"""Tests for WaypointTask."""

from __future__ import annotations

import numpy as np
from omegaconf import OmegaConf

from omninav.core.types import Observation, RobotState
from omninav.evaluation.tasks.waypoint_task import WaypointTask


def _obs(pos=(0, 0, 0), sim_time=0.0):
    return Observation(
        robot_state=RobotState(
            position=np.array([pos], dtype=np.float32),
            orientation=np.array([[1, 0, 0, 0]], dtype=np.float32),
            linear_velocity=np.zeros((1, 3), dtype=np.float32),
            angular_velocity=np.zeros((1, 3), dtype=np.float32),
            joint_positions=np.zeros((1, 4), dtype=np.float32),
            joint_velocities=np.zeros((1, 4), dtype=np.float32),
        ),
        sim_time=sim_time,
        sensors={},
    )


def test_waypoint_task_builds_spec():
    task = WaypointTask(
        OmegaConf.create(
            {
                "type": "waypoint",
                "waypoints": [[1, 0, 0], [2, 0, 0]],
                "waypoint_tolerance": 0.5,
            }
        )
    )
    spec = task.reset()
    assert spec["task_type"] == "waypoint"
    assert len(spec["goal_set"]) == 2


def test_waypoint_task_termination_and_result():
    task = WaypointTask(
        OmegaConf.create(
            {
                "type": "waypoint",
                "waypoints": [[0, 0, 0]],
                "waypoint_tolerance": 0.5,
                "success_requirement": 1.0,
            }
        )
    )
    task.reset()
    obs = _obs(pos=(0, 0, 0), sim_time=1.0)
    task.step(obs)
    done = task.is_terminated(obs)
    assert done.shape == (1,)
    assert bool(done[0])
    result = task.compute_result()
    assert result.success
    assert result.metrics["waypoint_coverage"] == 1.0
