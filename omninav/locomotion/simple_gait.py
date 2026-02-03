"""
Simple Gait Controller for Go2 (Demo-ready)

Uses direct joint oscillation for a simple trot gait.
No IK needed - directly modulates thigh & hip joints based on cmd_vel.

This is NOT RL-based stable locomotion, but sufficient for demo visualization.
For real Sim2Real, use RLController with pre-trained policy.
"""

from typing import TYPE_CHECKING
import numpy as np
from omegaconf import DictConfig

from omninav.locomotion.base import LocomotionControllerBase
from omninav.core.registry import LOCOMOTION_REGISTRY

if TYPE_CHECKING:
    from omninav.robots.base import RobotBase


@LOCOMOTION_REGISTRY.register("simple_gait")
class SimpleGaitController(LocomotionControllerBase):
    """
    Simple trot gait controller using joint oscillation.
    
    Modulates thigh joints sinusoidally for forward/backward motion.
    Modulates hip joints for lateral motion and turning.
    
    Config example (configs/locomotion/simple_gait.yaml):
        type: simple_gait
        gait_frequency: 1.5  # Hz
        thigh_swing_amplitude: 0.35  # rad
        hip_swing_amplitude: 0.15  # rad
        kp: 40.0
        kd: 1.0
    """
    
    CONTROLLER_TYPE = "simple_gait"
    
    # Joint order in Go2 URDF (after base DOFs at indices 0-5)
    # Indices 6-17 are motor joints
    JOINT_NAMES = [
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    ]
    
    # Default standing pose (reference: Genesis go2_env.py)
    DEFAULT_ANGLES = {
        "FL_hip_joint": 0.0, "FL_thigh_joint": 0.8, "FL_calf_joint": -1.5,
        "FR_hip_joint": 0.0, "FR_thigh_joint": 0.8, "FR_calf_joint": -1.5,
        "RL_hip_joint": 0.0, "RL_thigh_joint": 1.0, "RL_calf_joint": -1.5,
        "RR_hip_joint": 0.0, "RR_thigh_joint": 1.0, "RR_calf_joint": -1.5,
    }
    
    def __init__(self, cfg: DictConfig, robot: "RobotBase"):
        """Initialize simple gait controller."""
        super().__init__(cfg, robot)
        
        self._gait_freq = cfg.get("gait_frequency", 1.5)
        self._thigh_amp = cfg.get("thigh_swing_amplitude", 0.35)
        self._hip_amp = cfg.get("hip_swing_amplitude", 0.15)
        self._dt = cfg.get("dt", 0.01)
        self._kp = cfg.get("kp", 40.0)
        self._kd = cfg.get("kd", 1.0)
        
        # Phase for gait oscillation
        self._phase = 0.0
        
        # Default joint positions array
        self._default_qpos = np.array(
            [self.DEFAULT_ANGLES[name] for name in self.JOINT_NAMES],
            dtype=np.float32
        )
        
        # Joint indices for specific modulation
        # Map joint name to index within the 12 motor joints
        self._joint_idx_map = {name: i for i, name in enumerate(self.JOINT_NAMES)}
        
        # Thigh indices for trot gait
        self._idx_FL_thigh = self._joint_idx_map["FL_thigh_joint"]
        self._idx_FR_thigh = self._joint_idx_map["FR_thigh_joint"]
        self._idx_RL_thigh = self._joint_idx_map["RL_thigh_joint"]
        self._idx_RR_thigh = self._joint_idx_map["RR_thigh_joint"]
        
        # Hip indices for lateral motion
        self._idx_FL_hip = self._joint_idx_map["FL_hip_joint"]
        self._idx_FR_hip = self._joint_idx_map["FR_hip_joint"]
        self._idx_RL_hip = self._joint_idx_map["RL_hip_joint"]
        self._idx_RR_hip = self._joint_idx_map["RR_hip_joint"]
        
        # Motors DOF slice (skipping base DOFs 0-5)
        self._motors_dof_slice = slice(6, 18)
        
        self._initialized = False
        
    def _init_pd(self):
        """Initialize PD gains (call after scene build)."""
        if self._initialized:
            return
        # Use robot's actual DOF indices (set in robot.post_build)
        dof_idx = self.robot.motors_dof_idx
        kp = [self._kp] * 12
        kd = [self._kd] * 12
        self.robot.entity.set_dofs_kp(kp, dof_idx)
        self.robot.entity.set_dofs_kv(kd, dof_idx)
        self._initialized = True
    
    def reset(self) -> None:
        """Reset gait phase."""
        self._phase = 0.0
        self._initialized = False
    
    def compute_action(self, cmd_vel: np.ndarray) -> np.ndarray:
        """
        Compute joint targets from cmd_vel.
        
        Args:
            cmd_vel: [vx, vy, wz]
        
        Returns:
            Target joint positions (12 joints)
        """
        vx, vy, wz = cmd_vel[0], cmd_vel[1], cmd_vel[2]
        
        # Start from default pose
        target = self._default_qpos.copy()
        
        # Only oscillate if there's meaningful command
        cmd_norm = np.abs(vx) + np.abs(vy) + np.abs(wz)
        if cmd_norm < 0.01:
            return target
        
        # Phase-based oscillation
        phase_rad = 2 * np.pi * self._phase
        
        # Trot gait: FL+RR in phase, FR+RL opposite
        # Forward velocity controls oscillation amplitude
        fwd_scale = np.clip(vx / 0.5, -1, 1)  # Normalize to max 0.5 m/s
        
        fl_rr_swing = np.sin(phase_rad) * self._thigh_amp * fwd_scale
        fr_rl_swing = np.sin(phase_rad + np.pi) * self._thigh_amp * fwd_scale
        
        target[self._idx_FL_thigh] += fl_rr_swing
        target[self._idx_RR_thigh] += fl_rr_swing
        target[self._idx_FR_thigh] += fr_rl_swing
        target[self._idx_RL_thigh] += fr_rl_swing
        
        # Lateral motion: hip abduction/adduction
        lat_scale = np.clip(vy / 0.3, -1, 1)  # Normalize
        hip_bias = self._hip_amp * lat_scale
        target[self._idx_FL_hip] += hip_bias
        target[self._idx_RL_hip] += hip_bias
        target[self._idx_FR_hip] -= hip_bias
        target[self._idx_RR_hip] -= hip_bias
        
        # Turning: differential hip angles
        yaw_scale = np.clip(wz / 1.0, -1, 1)  # Normalize
        yaw_bias = 0.12 * yaw_scale
        target[self._idx_FL_thigh] += yaw_bias
        target[self._idx_RR_thigh] += yaw_bias
        target[self._idx_FR_thigh] -= yaw_bias
        target[self._idx_RL_thigh] -= yaw_bias
        
        return target
    
    def step(self, cmd_vel: np.ndarray) -> None:
        """Execute one locomotion step."""
        self._init_pd()
        
        # Update phase
        self._phase = (self._phase + self._dt * self._gait_freq) % 1.0
        
        # Compute and apply via robot interface
        target_qpos = self.compute_action(cmd_vel)
        self.robot.control_joints_position(target_qpos)

