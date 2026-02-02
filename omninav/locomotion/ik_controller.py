"""
IK-based Gait Controller for Go2

Converts cmd_vel to joint positions using inverse kinematics and gait planning.
"""

from typing import TYPE_CHECKING, Dict, List, Tuple
import numpy as np
from omegaconf import DictConfig

from omninav.locomotion.base import LocomotionControllerBase
from omninav.core.registry import LOCOMOTION_REGISTRY

if TYPE_CHECKING:
    from omninav.robots.base import RobotBase


@LOCOMOTION_REGISTRY.register("ik_controller")
class IKController(LocomotionControllerBase):
    """
    IK-based gait controller for Go2 quadruped.
    
    Implements a simple trot gait using Bezier curve foot trajectories
    and Genesis inverse kinematics API.
    
    Gait phases (Trot):
        - Diagonal legs move together (FL+RR, FR+RL)
        - Phase 0-0.5: FL+RR swing, FR+RL stance
        - Phase 0.5-1.0: FR+RL swing, FL+RR stance
    
    Config example (configs/locomotion/ik_gait.yaml):
        type: ik_controller
        gait_type: trot
        step_height: 0.05
        step_length: 0.15
        gait_frequency: 2.0  # Hz
        body_height: 0.35
    """
    
    CONTROLLER_TYPE = "ik_controller"
    
    # Leg indices
    LEG_FL = 0
    LEG_FR = 1
    LEG_RL = 2
    LEG_RR = 3
    
    def __init__(self, cfg: DictConfig, robot: "RobotBase"):
        """
        Initialize IK controller.
        
        Args:
            cfg: Controller configuration
            robot: Robot instance
        """
        super().__init__(cfg, robot)
        
        # Gait parameters
        self._gait_type = cfg.get("gait_type", "trot")
        self._step_height = cfg.get("step_height", 0.05)
        self._step_length = cfg.get("step_length", 0.15)
        self._gait_frequency = cfg.get("gait_frequency", 2.0)
        self._body_height = cfg.get("body_height", 0.35)
        
        # Leg geometry (relative to body)
        self._hip_offsets = np.array([
            [+0.1881, +0.04675, 0.0],  # FL
            [+0.1881, -0.04675, 0.0],  # FR
            [-0.1881, +0.04675, 0.0],  # RL
            [-0.1881, -0.04675, 0.0],  # RR
        ], dtype=np.float32)
        
        # Foot link names (for IK targets)
        self._foot_link_names = cfg.get("foot_links", [
            "FL_foot", "FR_foot", "RL_foot", "RR_foot"
        ])
        
        # State
        self._phase = 0.0
        self._dt = cfg.get("dt", 0.01)
        
        # Default standing foot positions (relative to body)
        self._default_foot_pos = self._hip_offsets.copy()
        self._default_foot_pos[:, 2] = -self._body_height
    
    def reset(self) -> None:
        """Reset gait phase."""
        self._phase = 0.0
    
    def _get_trot_phases(self) -> Tuple[float, float]:
        """
        Get swing phases for trot gait.
        
        Returns:
            Tuple of (phase_FL_RR, phase_FR_RL)
        """
        # FL+RR and FR+RL are 180 degrees out of phase
        phase_FL_RR = self._phase
        phase_FR_RL = (self._phase + 0.5) % 1.0
        return phase_FL_RR, phase_FR_RL
    
    def _bezier_swing_trajectory(
        self, 
        t: float, 
        start: np.ndarray, 
        end: np.ndarray
    ) -> np.ndarray:
        """
        Compute Bezier curve for swing phase.
        
        Uses a 4-point Bezier curve with apex at step_height.
        
        Args:
            t: Normalized time in swing phase [0, 1]
            start: Starting foot position
            end: Ending foot position
        
        Returns:
            Foot position at time t
        """
        # Control points for swing trajectory
        mid = (start + end) / 2
        apex = mid.copy()
        apex[2] = start[2] + self._step_height
        
        # Cubic Bezier: B(t) = (1-t)^3*P0 + 3(1-t)^2*t*P1 + 3(1-t)*t^2*P2 + t^3*P3
        P0 = start
        P1 = start + np.array([0, 0, self._step_height * 0.5])
        P2 = end + np.array([0, 0, self._step_height * 0.5])
        P3 = end
        
        pos = (
            (1 - t) ** 3 * P0 +
            3 * (1 - t) ** 2 * t * P1 +
            3 * (1 - t) * t ** 2 * P2 +
            t ** 3 * P3
        )
        return pos
    
    def _compute_foot_targets(
        self, 
        cmd_vel: np.ndarray
    ) -> np.ndarray:
        """
        Compute target foot positions from velocity command.
        
        Args:
            cmd_vel: [vx, vy, wz] velocity command
        
        Returns:
            Array of shape (4, 3) with target foot positions
        """
        vx, vy, wz = cmd_vel[0], cmd_vel[1], cmd_vel[2]
        
        # Get trot phases
        phase_FL_RR, phase_FR_RL = self._get_trot_phases()
        
        foot_targets = self._default_foot_pos.copy()
        
        # Compute step displacement from velocity
        step_x = vx * self._step_length
        step_y = vy * self._step_length
        
        for leg_idx in range(4):
            # Determine which phase group this leg belongs to
            if leg_idx in [self.LEG_FL, self.LEG_RR]:
                phase = phase_FL_RR
            else:
                phase = phase_FR_RL
            
            # Default position
            default_pos = self._default_foot_pos[leg_idx].copy()
            
            # Swing phase (0-0.5 of leg's cycle)
            if phase < 0.5:
                swing_t = phase * 2  # Normalize to [0, 1]
                
                # Start and end positions for swing
                start_pos = default_pos - np.array([step_x / 2, step_y / 2, 0])
                end_pos = default_pos + np.array([step_x / 2, step_y / 2, 0])
                
                foot_targets[leg_idx] = self._bezier_swing_trajectory(
                    swing_t, start_pos, end_pos
                )
            else:
                # Stance phase (0.5-1.0) - foot moves backward on ground
                stance_t = (phase - 0.5) * 2  # Normalize to [0, 1]
                
                # Linear interpolation during stance
                start_pos = default_pos + np.array([step_x / 2, step_y / 2, 0])
                end_pos = default_pos - np.array([step_x / 2, step_y / 2, 0])
                
                foot_targets[leg_idx] = start_pos + stance_t * (end_pos - start_pos)
        
        return foot_targets
    
    def compute_action(self, cmd_vel: np.ndarray) -> np.ndarray:
        """
        Compute joint positions from cmd_vel using IK.
        
        Args:
            cmd_vel: [vx, vy, wz] velocity command
        
        Returns:
            Joint positions array (12 joints for 4 legs x 3 joints each)
        """
        # Compute foot targets
        foot_targets = self._compute_foot_targets(cmd_vel)
        
        # Get current joint positions as initial guess
        current_qpos = self.robot.entity.get_dofs_position().cpu().numpy()
        if current_qpos.ndim > 1:
            current_qpos = current_qpos[0]
        
        joint_positions = current_qpos.copy()
        
        # Solve IK for each leg
        for leg_idx in range(4):
            foot_link_name = self._foot_link_names[leg_idx]
            foot_link = self.robot.entity.get_link(foot_link_name)
            
            if foot_link is None:
                continue
            
            # Call Genesis IK solver
            try:
                result = self.robot.entity.inverse_kinematics(
                    link=foot_link,
                    pos=foot_targets[leg_idx],
                    quat=None,  # No orientation constraint
                    init_qpos=current_qpos,
                    max_iterations=50,
                    return_error=False,
                )
                
                # Extract leg joint positions (3 joints per leg)
                # Assuming joint order: hip, thigh, calf for each leg
                leg_joint_start = leg_idx * 3
                joint_positions[leg_joint_start:leg_joint_start + 3] = (
                    result[leg_joint_start:leg_joint_start + 3]
                )
            except Exception:
                # Keep current positions if IK fails
                pass
        
        return joint_positions.astype(np.float32)
    
    def step(self, cmd_vel: np.ndarray) -> None:
        """
        Execute one locomotion control step.
        
        Args:
            cmd_vel: [vx, vy, wz] velocity command
        """
        # Update gait phase
        self._phase = (self._phase + self._dt * self._gait_frequency) % 1.0
        
        # Compute and apply joint positions
        joint_positions = self.compute_action(cmd_vel)
        
        # Apply position control
        self.robot.entity.control_dofs_position(joint_positions)
