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
        Compute joint positions from cmd_vel using Multilink IK.
        
        Args:
            cmd_vel: [vx, vy, wz] velocity command
        
        Returns:
            Joint positions array (12 joints for 4 legs x 3 joints each)
        """
        import genesis as gs
        
        # 1. Compute foot targets (Cartesian)
        dist_x = cmd_vel[0] * self._dt
        dist_y = cmd_vel[1] * self._dt
        ang_z = cmd_vel[2] * self._dt
        
        foot_targets = self._compute_foot_targets(cmd_vel)
        
        # 2. Prepare for IK
        target_links = []
        target_pos_list = []
        target_quat_list = []
        
        # Default foot orientation
        default_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Offset from Calf to Foot (approximate)
        # We want Foot at foot_target, so Calf should be at foot_target - offset
        # detailed: P_foot = P_calf + R * offset. Assuming R roughly identity (vertical leg) -> P_calf = P_foot - offset
        # This is an approximation. A better way requires full kinematic chain logic or Genesis feature.
        foot_offset = np.array(self.cfg.get("foot_offset", [0.0, 0.0, -0.213]), dtype=np.float32)
        
        for leg_idx in range(4):
            foot_link_name = self._foot_link_names[leg_idx]
            foot_link = self.robot.entity.get_link(foot_link_name)
            
            if foot_link is None:
                # Try to fallback to original name if config update failed/cached? 
                # Or just skip
                continue
                
            target_links.append(foot_link)
            
            # Apply offset: Target Calf Pos = Foot Target - Foot Offset
            # Assuming standing orientation (vertical leg, offset is local Z)
            # Since offset is [0, 0, -0.213], minus offset means adding 0.213 to Z.
            target_pos = foot_targets[leg_idx] - foot_offset
            
            target_pos_list.append(target_pos)
            target_quat_list.append(default_quat)
            
        if not target_links:
            return self.robot.entity.get_dofs_position().cpu().numpy()

        # 3. Call Genesis Multilink IK
        # rot_mask=[False, False, False] means we don't strictly enforce orientation 
        # (allowing feet to tilt slightly avoids singularities), but we pass quat anyway
        try:
            # Current Q as initialization
            current_q = self.robot.entity.get_dofs_position()
            if hasattr(current_q, 'cpu'):
                current_q = current_q.cpu().numpy()
            if current_q.ndim > 1:
                current_q = current_q[0]

            q_sol = self.robot.entity.inverse_kinematics_multilink(
                links=target_links,
                poss=target_pos_list,
                quats=target_quat_list,
                rot_mask=[False] * len(target_links), # Relax orientation constraint
                init_qpos=current_q,
                max_iterations=20,
                lr=0.5,
            )
            
            return q_sol
            
        except Exception as e:
            # Fallback on failure
            return current_q
            
    def step(self, cmd_vel: np.ndarray) -> None:
        """
        Execute one locomotion control step.
        
        Args:
            cmd_vel: [vx, vy, wz] velocity command
        """
        # Update gait phase
        self._phase = (self._phase + self._dt * self._gait_frequency) % 1.0
        
        # Compute IK solution
        joint_positions = self.compute_action(cmd_vel)
        
        # Apply position control
        self.robot.control_joints_position(joint_positions)

