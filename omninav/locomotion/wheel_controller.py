"""
Wheel Controller for Go2w

Converts cmd_vel to wheel velocities for wheeled quadruped.
"""

from typing import TYPE_CHECKING
import numpy as np
from omegaconf import DictConfig

from omninav.locomotion.base import LocomotionControllerBase
from omninav.core.registry import LOCOMOTION_REGISTRY

if TYPE_CHECKING:
    from omninav.robots.base import RobotBase


@LOCOMOTION_REGISTRY.register("wheel_controller")
class WheelController(LocomotionControllerBase):
    """
    Wheel velocity controller for Go2w (wheeled quadruped).
    
    Converts [vx, vy, wz] velocity commands to individual wheel angular velocities.
    Assumes a 4-wheel configuration (Mecanum or differential drive).
    
    Wheel layout (top view):
        FL ---- FR
        |        |
        |   C    |
        |        |
        RL ---- RR
    
    Config example (configs/locomotion/wheel.yaml):
        type: wheel_controller
        wheel_radius: 0.05
        wheel_base: 0.4      # Distance between front and rear axles
        track_width: 0.3     # Distance between left and right wheels
        max_wheel_speed: 20.0  # rad/s
    """
    
    CONTROLLER_TYPE = "wheel_controller"
    
    def __init__(self, cfg: DictConfig, robot: "RobotBase"):
        """
        Initialize wheel controller.
        
        Args:
            cfg: Controller configuration
            robot: Robot instance
        """
        super().__init__(cfg, robot)
        
        # Wheel geometry from config
        self._wheel_radius = cfg.get("wheel_radius", 0.05)
        self._wheel_base = cfg.get("wheel_base", 0.4)    # L (front-rear distance)
        self._track_width = cfg.get("track_width", 0.3)  # W (left-right distance)
        self._max_wheel_speed = cfg.get("max_wheel_speed", 20.0)
        
        # Wheel joint names (should match URDF)
        self._wheel_joint_names = cfg.get("wheel_joints", [
            "FL_wheel_joint",
            "FR_wheel_joint",
            "RL_wheel_joint",
            "RR_wheel_joint",
        ])
        
        # Cached joint indices (set after robot spawns)
        self._wheel_joint_indices: np.ndarray = None
    
    def reset(self) -> None:
        """Reset controller state."""
        self._wheel_joint_indices = None
    
    def _get_wheel_indices(self) -> np.ndarray:
        """
        Get joint indices for wheel joints.
        
        Returns:
            Array of joint indices
        """
        if self._wheel_joint_indices is not None:
            return self._wheel_joint_indices
        
        # Find wheel joint indices from robot
        indices = []
        joint_names = self.robot.entity.joint_names
        for wheel_name in self._wheel_joint_names:
            if wheel_name in joint_names:
                idx = joint_names.index(wheel_name)
                indices.append(idx)
            else:
                raise ValueError(
                    f"Wheel joint '{wheel_name}' not found in robot. "
                    f"Available joints: {joint_names}"
                )
        
        self._wheel_joint_indices = np.array(indices, dtype=np.int32)
        return self._wheel_joint_indices
    
    def compute_action(self, cmd_vel: np.ndarray) -> np.ndarray:
        """
        Compute wheel velocities from cmd_vel.
        
        Uses inverse kinematics for Mecanum wheel configuration:
            v_FL = (vx - vy - (L+W)*wz) / R
            v_FR = (vx + vy + (L+W)*wz) / R
            v_RL = (vx + vy - (L+W)*wz) / R
            v_RR = (vx - vy + (L+W)*wz) / R
        
        Args:
            cmd_vel: [vx, vy, wz] velocity command
        
        Returns:
            Wheel angular velocities [FL, FR, RL, RR] in rad/s
        """
        vx, vy, wz = cmd_vel[0], cmd_vel[1], cmd_vel[2]
        R = self._wheel_radius
        L = self._wheel_base / 2
        W = self._track_width / 2
        
        # Mecanum wheel inverse kinematics
        v_FL = (vx - vy - (L + W) * wz) / R
        v_FR = (vx + vy + (L + W) * wz) / R
        v_RL = (vx + vy - (L + W) * wz) / R
        v_RR = (vx - vy + (L + W) * wz) / R
        
        wheel_velocities = np.array([v_FL, v_FR, v_RL, v_RR], dtype=np.float32)
        
        # Clip to max wheel speed
        wheel_velocities = np.clip(
            wheel_velocities, -self._max_wheel_speed, self._max_wheel_speed
        )
        
        return wheel_velocities
    
    def step(self, cmd_vel: np.ndarray) -> None:
        """
        Execute one locomotion control step.
        
        Computes wheel velocities and applies to robot.
        
        Args:
            cmd_vel: [vx, vy, wz] velocity command
        """
        wheel_velocities = self.compute_action(cmd_vel)
        indices = self._get_wheel_indices()
        
        # Apply wheel velocities using velocity control
        self.robot.entity.control_dofs_velocity(
            wheel_velocities,
            dof_indices=indices,
        )
