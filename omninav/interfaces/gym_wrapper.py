"""
OmniNav Gym Wrapper — Optional Gym-compatible wrapper for RL training.

Wraps OmniNavEnv with Gymnasium's Env interface, adding:
- observation_space / action_space definitions
- Reward function injection
- Standard step/reset signatures
"""

from typing import Any, Callable, Dict, Optional, Tuple, TYPE_CHECKING
import numpy as np

try:
    import gymnasium
    HAS_GYMNASIUM = True
except ImportError:
    HAS_GYMNASIUM = False

if TYPE_CHECKING:
    from omninav.interfaces.python_api import OmniNavEnv
    from omninav.core.types import Observation


def _default_reward_fn(obs: "Observation", action: np.ndarray, info: dict) -> float:
    """Default reward: negative time penalty."""
    return -0.01


class OmniNavGymWrapper:
    """
    Optional Gymnasium-compatible wrapper for RL training.

    Requires gymnasium package to be installed.

    Usage:
        env = OmniNavEnv(cfg=my_cfg)
        gym_env = OmniNavGymWrapper(env, reward_fn=my_reward)
        obs, info = gym_env.reset()
        obs, reward, terminated, truncated, info = gym_env.step(action)
    """

    def __init__(
        self,
        env: "OmniNavEnv",
        reward_fn: Optional[Callable] = None,
        obs_keys: Optional[list] = None,
        max_episode_steps: int = 1000,
    ):
        """
        Initialize Gym wrapper.

        Args:
            env: OmniNavEnv instance
            reward_fn: Callable(obs, action, info) -> float
            obs_keys: Keys to extract from Observation for flat obs vector
            max_episode_steps: Maximum steps per episode
        """
        if not HAS_GYMNASIUM:
            raise ImportError(
                "gymnasium package required for OmniNavGymWrapper. "
                "Install with: pip install gymnasium"
            )

        self.env = env
        self._reward_fn = reward_fn or _default_reward_fn
        self._obs_keys = obs_keys or ["robot_state"]
        self._max_episode_steps = max_episode_steps
        self._current_step = 0

        # Define spaces (default: cmd_vel action space)
        self.action_space = gymnasium.spaces.Box(
            low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Observation space: defined as flat vector
        # Size depends on what obs_keys are used
        obs_dim = self._estimate_obs_dim()
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

    def _estimate_obs_dim(self) -> int:
        """Estimate observation dimension from configured keys."""
        dim = 0
        for key in self._obs_keys:
            if key == "robot_state":
                # position(3) + orientation(4) + lin_vel(3) + ang_vel(3) + joints(12)*2
                dim += 3 + 4 + 3 + 3 + 12 + 12
            elif key == "goal_position":
                dim += 3
        return max(dim, 1)

    def _obs_to_vector(self, obs: "Observation") -> np.ndarray:
        """Flatten observation to a vector for Gym compatibility."""
        parts = []
        for key in self._obs_keys:
            if key == "robot_state" and "robot_state" in obs:
                rs = obs["robot_state"]
                for field in ["position", "orientation", "linear_velocity",
                              "angular_velocity", "joint_positions", "joint_velocities"]:
                    val = np.asarray(rs.get(field, np.zeros(1)))
                    if val.ndim > 1:
                        val = val[0]  # debatch
                    parts.append(val.flatten())
            elif key == "goal_position" and "goal_position" in obs:
                gp = np.asarray(obs["goal_position"])
                if gp.ndim > 1:
                    gp = gp[0]
                parts.append(gp.flatten())

        if not parts:
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        flat = np.concatenate(parts).astype(np.float32)

        # Pad or truncate to match expected dimension
        expected = self.observation_space.shape[0]
        if len(flat) < expected:
            flat = np.pad(flat, (0, expected - len(flat)))
        elif len(flat) > expected:
            flat = flat[:expected]

        return flat

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        """
        Reset environment.

        Returns:
            obs: Flat observation vector
            info: Reset info
        """
        self._current_step = 0
        obs_list = self.env.reset()

        obs = obs_list[0] if obs_list else {}
        flat_obs = self._obs_to_vector(obs)

        return flat_obs, {"raw_obs": obs}

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step.

        Args:
            action: Action array [vx, vy, wz]

        Returns:
            obs, reward, terminated, truncated, info
        """
        from omninav.core.types import Action as ActionType

        actions = [ActionType(cmd_vel=np.asarray(action, dtype=np.float32))]
        obs_list, info = self.env.step(actions)

        obs = obs_list[0] if obs_list else {}
        flat_obs = self._obs_to_vector(obs)

        # Compute reward
        reward = self._reward_fn(obs, action, info)

        # Check termination
        terminated = self.env.is_done
        self._current_step += 1
        truncated = self._current_step >= self._max_episode_steps

        info["raw_obs"] = obs

        return flat_obs, reward, terminated, truncated, info

    def close(self):
        """Close wrapped environment."""
        self.env.close()
