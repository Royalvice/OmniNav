"""Runtime tests for task spec handoff to algorithms."""

from __future__ import annotations

import numpy as np
from omegaconf import OmegaConf

from omninav.core.runtime import SimulationRuntime
from omninav.core.types import Observation, RobotState


class _DummySim:
    def build(self):
        return None

    def reset(self):
        return None

    def step(self):
        return None


class _DummyRobot:
    def post_build(self):
        return None

    def reset(self):
        return None

    def get_state(self):
        return RobotState(
            position=np.zeros((1, 3), dtype=np.float32),
            orientation=np.array([[1, 0, 0, 0]], dtype=np.float32),
            linear_velocity=np.zeros((1, 3), dtype=np.float32),
            angular_velocity=np.zeros((1, 3), dtype=np.float32),
            joint_positions=np.zeros((1, 4), dtype=np.float32),
            joint_velocities=np.zeros((1, 4), dtype=np.float32),
        )


class _DummyTask:
    lifecycle_state = None

    def reset(self):
        return {"task_type": "dummy", "goal_set": [[1, 0, 0]]}

    def build_task_spec(self):
        return {"task_type": "dummy", "goal_set": [[2, 0, 0]]}

    def step(self, obs, action):
        _ = (obs, action)

    def is_terminated(self, obs):
        _ = obs
        return np.array([False], dtype=bool)

    def update_task_feedback(self, info):
        _ = info


class _DummyAlgo:
    lifecycle_state = None

    def __init__(self):
        self.last_reset = None
        self._is_done = False

    def reset(self, task_info=None):
        self.last_reset = dict(task_info or {})

    def step(self, obs: Observation):
        _ = obs
        return np.zeros((1, 3), dtype=np.float32)

    @property
    def is_done(self):
        return self._is_done

    @property
    def info(self):
        return {"dummy": True}


def test_runtime_uses_task_build_task_spec():
    rt = SimulationRuntime(OmegaConf.create({"simulation": {"dt": 0.01}}))
    rt.sim = _DummySim()
    rt.robots = [_DummyRobot()]
    algo = _DummyAlgo()
    rt.algorithms = [algo]
    rt.task = _DummyTask()

    rt.build()
    obs = rt.reset()
    assert len(obs) == 1
    assert algo.last_reset == {"task_type": "dummy", "goal_set": [[2, 0, 0]]}
