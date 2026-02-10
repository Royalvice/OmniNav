"""
RL Controller Placeholder

Interface for RL-based locomotion policies (not implemented).
"""

from typing import TYPE_CHECKING, Dict, Optional
import numpy as np
from omegaconf import DictConfig

from omninav.locomotion.base import LocomotionControllerBase
from omninav.core.registry import LOCOMOTION_REGISTRY

if TYPE_CHECKING:
    from omninav.robots.base import RobotBase


@LOCOMOTION_REGISTRY.register("rl_controller")
class RLController(LocomotionControllerBase):
    """
    Placeholder for RL-based locomotion policy.
    
    This class provides the interface for integrating reinforcement learning
    locomotion policies (e.g., from Isaac Gym / Legged Gym).
    
    NOT YET IMPLEMENTED - raises NotImplementedError.
    
    Future implementation will:
    - Load pre-trained policy from checkpoint
    - Process observations (joint positions, velocities, IMU, etc.)
    - Output joint position/torque targets
    
    Config example (configs/locomotion/rl_policy.yaml):
        type: rl_controller
        policy_path: "models/go2_locomotion.pt"
        observation_keys: ["joint_pos", "joint_vel", "imu"]
    """
    
    CONTROLLER_TYPE = "rl_controller"
    
    def __init__(self, cfg: DictConfig, robot: "RobotBase"):
        """
        Initialize RL controller.
        
        Args:
            cfg: Controller configuration
            robot: Robot instance
        """
        super().__init__(cfg, robot)
        
        self._policy_path = cfg.get("policy_path", None)
        self._observation_keys = list(cfg.get("observation_keys", []))
        
        # Placeholder for loaded policy
        self._policy = None
    
    def reset(self) -> None:
        """Reset controller state."""
        pass
    
    def compute_action(self, cmd_vel: np.ndarray) -> np.ndarray:
        """
        Compute action from RL policy.
        
        Args:
            cmd_vel: [vx, vy, wz] velocity command
        
        Returns:
            Joint targets
        
        Raises:
            NotImplementedError: RL policy not yet implemented
        """
        raise NotImplementedError(
            "RL locomotion policy is not yet implemented. "
            "Use 'ik_controller' for Go2 or 'wheel_controller' for Go2w."
        )
    
    def step(self, cmd_vel: np.ndarray, obs: Optional["Observation"] = None) -> None:
        """
        Execute one locomotion control step.
        
        Args:
            cmd_vel: [vx, vy, wz] velocity command
            obs: Optional observation data
        
        Raises:
            NotImplementedError: RL policy not yet implemented
        """
        raise NotImplementedError(
            "RL locomotion policy is not yet implemented. "
            "Use 'ik_controller' for Go2 or 'wheel_controller' for Go2w."
        )
