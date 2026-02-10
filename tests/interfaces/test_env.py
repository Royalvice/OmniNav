"""
Tests for Interface layer: SimulationRuntime, OmniNavEnv, OmniNavGymWrapper.
"""

import pytest
import numpy as np
from omegaconf import OmegaConf
from unittest.mock import MagicMock

from omninav.core.types import Observation, RobotState, Action


def _gymnasium_available():
    try:
        import gymnasium
        return True
    except ImportError:
        return False


# =============================================================================
# Helpers
# =============================================================================

def _make_mock_robot():
    """Create a mock robot that returns valid RobotState."""
    robot = MagicMock()
    robot.get_state.return_value = RobotState(
        position=np.zeros((1, 3), dtype=np.float32),
        orientation=np.array([[1, 0, 0, 0]], dtype=np.float32),
        linear_velocity=np.zeros((1, 3), dtype=np.float32),
        angular_velocity=np.zeros((1, 3), dtype=np.float32),
        joint_positions=np.zeros((1, 12), dtype=np.float32),
        joint_velocities=np.zeros((1, 12), dtype=np.float32),
    )
    robot.sensors = {}
    return robot


# =============================================================================
# SimulationRuntime Tests
# =============================================================================

class TestSimulationRuntime:

    def test_empty_runtime_reset(self):
        """Runtime with no components should reset without error."""
        from omninav.core.runtime import SimulationRuntime

        cfg = OmegaConf.create({})
        runtime = SimulationRuntime(cfg)
        obs = runtime.reset()

        assert obs == []
        assert runtime.step_count == 0

    def test_runtime_with_mock_robot(self):
        """Runtime should collect observations from robots."""
        from omninav.core.runtime import SimulationRuntime

        cfg = OmegaConf.create({"simulation": {"dt": 0.01}})
        runtime = SimulationRuntime(cfg)
        runtime.robots.append(_make_mock_robot())

        obs_list = runtime.reset()
        assert len(obs_list) == 1
        assert "robot_state" in obs_list[0]
        assert "sim_time" in obs_list[0]

    def test_runtime_step_increments_counters(self):
        """Step should increment step count and sim time."""
        from omninav.core.runtime import SimulationRuntime

        cfg = OmegaConf.create({"simulation": {"dt": 0.01}})
        runtime = SimulationRuntime(cfg)
        runtime.robots.append(_make_mock_robot())

        runtime.reset()
        runtime.step([Action(cmd_vel=np.zeros(3))])

        assert runtime.step_count == 1
        assert runtime.sim_time == pytest.approx(0.01)

    def test_runtime_build_emits_events(self):
        """Build should emit PRE_BUILD and POST_BUILD events."""
        from omninav.core.runtime import SimulationRuntime
        from omninav.core.hooks import EventType

        cfg = OmegaConf.create({})
        runtime = SimulationRuntime(cfg)

        events = []
        runtime.hooks.register(EventType.PRE_BUILD, lambda **kw: events.append("pre"))
        runtime.hooks.register(EventType.POST_BUILD, lambda **kw: events.append("post"))

        runtime.build()
        assert events == ["pre", "post"]

    def test_runtime_reset_emits_event(self):
        """Reset should emit ON_RESET event."""
        from omninav.core.runtime import SimulationRuntime
        from omninav.core.hooks import EventType

        cfg = OmegaConf.create({})
        runtime = SimulationRuntime(cfg)

        events = []
        runtime.hooks.register(EventType.ON_RESET, lambda **kw: events.append("reset"))

        runtime.reset()
        assert "reset" in events

    def test_runtime_is_done_with_task(self):
        """is_done should delegate to task.is_terminated()."""
        from omninav.core.runtime import SimulationRuntime

        cfg = OmegaConf.create({})
        runtime = SimulationRuntime(cfg)
        runtime.robots.append(_make_mock_robot())

        mock_task = MagicMock()
        mock_task.is_terminated.return_value = True
        mock_task.reset.return_value = {}
        runtime.task = mock_task

        runtime.reset()
        assert runtime.is_done


# =============================================================================
# OmniNavEnv Tests
# =============================================================================

class TestOmniNavEnv:

    def test_init_with_empty_config(self):
        """Should initialize with empty config."""
        from omninav.interfaces.python_api import OmniNavEnv

        env = OmniNavEnv()
        assert env.cfg is not None
        assert env.step_count == 0

    def test_env_reset_initializes_runtime(self):
        """Reset should trigger lazy init of runtime."""
        from omninav.interfaces.python_api import OmniNavEnv

        env = OmniNavEnv()
        obs_list = env.reset()

        assert env._initialized
        assert isinstance(obs_list, list)

    def test_env_step_returns_observations(self):
        """Step should return tuple of observations and info."""
        from omninav.interfaces.python_api import OmniNavEnv

        env = OmniNavEnv()
        env.reset()
        obs, info = env.step()

        assert isinstance(obs, list)
        assert isinstance(info, dict)

    def test_context_manager(self):
        """Should support context manager protocol."""
        from omninav.interfaces.python_api import OmniNavEnv

        with OmniNavEnv() as env:
            env.reset()
        assert not env._initialized

    def test_hooks_access(self):
        """Should expose hooks from runtime."""
        from omninav.interfaces.python_api import OmniNavEnv
        from omninav.core.hooks import HookManager

        env = OmniNavEnv()
        assert isinstance(env.hooks, HookManager)


# =============================================================================
# OmniNavGymWrapper Tests (requires gymnasium)
# =============================================================================

@pytest.mark.skipif(not _gymnasium_available(), reason="gymnasium not installed")
class TestOmniNavGymWrapper:

    @pytest.fixture
    def mock_env(self):
        """Create a mock OmniNavEnv."""
        env = MagicMock()
        env.reset.return_value = [
            Observation(
                robot_state=RobotState(
                    position=np.zeros((1, 3), dtype=np.float32),
                    orientation=np.array([[1, 0, 0, 0]], dtype=np.float32),
                    linear_velocity=np.zeros((1, 3), dtype=np.float32),
                    angular_velocity=np.zeros((1, 3), dtype=np.float32),
                    joint_positions=np.zeros((1, 12), dtype=np.float32),
                    joint_velocities=np.zeros((1, 12), dtype=np.float32),
                ),
                sim_time=0.0,
                sensors={},
            )
        ]
        env.step.return_value = (env.reset.return_value, {"step": 1})
        env.is_done = False
        return env

    def test_reset_returns_flat_obs(self, mock_env):
        from omninav.interfaces.gym_wrapper import OmniNavGymWrapper

        wrapper = OmniNavGymWrapper(mock_env)
        obs, info = wrapper.reset()

        assert isinstance(obs, np.ndarray)
        assert obs.ndim == 1

    def test_step_returns_gym_tuple(self, mock_env):
        from omninav.interfaces.gym_wrapper import OmniNavGymWrapper

        wrapper = OmniNavGymWrapper(mock_env)
        wrapper.reset()

        action = np.array([0.5, 0.0, 0.1], dtype=np.float32)
        result = wrapper.step(action)

        assert len(result) == 5  # obs, reward, terminated, truncated, info
        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
