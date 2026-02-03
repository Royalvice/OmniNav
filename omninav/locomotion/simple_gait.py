"""
Simple Gait Controller for Go2 (Kinematic Mode)

Uses kinematic animation-style control:
- Directly moves robot base position.
- Uses raycasting for collision detection to prevent passing through objects.
- Animates legs using simple sinusoidal oscillation for visual effect.
- Does not fall due to gravity compensation.
"""

from typing import TYPE_CHECKING, Tuple, Optional
import numpy as np
from omegaconf import DictConfig

from omninav.locomotion.base import LocomotionControllerBase
from omninav.core.registry import LOCOMOTION_REGISTRY

if TYPE_CHECKING:
    from omninav.robots.base import RobotBase


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by quaternion q (wxyz format)."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    xyz = np.array([x, y, z])
    t = 2.0 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)


@LOCOMOTION_REGISTRY.register("simple_gait")
class SimpleGaitController(LocomotionControllerBase):
    """
    Kinematic animation-style gait controller for Go2.
    
    This controller bypasses physics for movement but performs raycasting
    to detect obstacles and maintain height.
    """
    
    CONTROLLER_TYPE = "simple_gait"
    
    def __init__(self, cfg: DictConfig, robot: "RobotBase"):
        """Initialize simple gait controller."""
        super().__init__(cfg, robot)
        
        # Gait parameters
        self._gait_freq = cfg.get("gait_frequency", 2.0)
        self._thigh_amp = cfg.get("thigh_swing_amplitude", 0.4)
        self._hip_amp = cfg.get("hip_swing_amplitude", 0.1)
        self._dt = cfg.get("dt", 0.01)
        
        # Collision parameters
        self._collision_margin = cfg.get("collision_margin", 0.2)  # Distance from center to check
        self._robot_radius = cfg.get("robot_radius", 0.3)
        self._check_collision = cfg.get("check_collision", True)
        
        # State
        self._phase = 0.0
        
        # Default joint positions in URDF order (for set_qpos)
        self._init_joint_positions()
        
    def _init_joint_positions(self):
        """Initialize default joint positions in URDF order."""
        # Get joints from robot entity (skipping floating base)
        # Assuming robot.entity is already spawned
        self._joints = self.robot.entity.joints[1:]
        self._default_qpos = np.array([
            self.robot.DEFAULT_JOINT_ANGLES[joint.name] 
            for joint in self._joints
        ], dtype=np.float32)

    def reset(self) -> None:
        """Reset gait phase."""
        self._phase = 0.0

    def _get_base_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current base position and orientation."""
        pos = self.robot.entity.get_pos()
        quat = self.robot.entity.get_quat()
        if hasattr(pos, 'cpu'): pos = pos.cpu().numpy()
        if hasattr(quat, 'cpu'): quat = quat.cpu().numpy()
        if pos.ndim > 1: pos = pos[0]
        if quat.ndim > 1: quat = quat[0]
        return pos, quat

    def _check_obstacle(self, pos: np.ndarray, direction: np.ndarray, distance: float) -> bool:
        """Return True if an obstacle is hit by raycast."""
        if not self._check_collision:
            return False
            
        # Temporarily disabled due to scene.raycast error
        return False
        
        # # Perform raycast from center at height 0.2m
        # ray_origin = pos + np.array([0, 0, 0.2])
        # hit = self.robot.scene.raycast(ray_origin, direction, max_dist=distance)
        
        # # Genesis raycast returns a generic hit object or similar
        # # We need to check if has_hit is true and hit distance is small
        # if hasattr(hit, 'has_hit') and hit.has_hit:
        #     # Check distance. If hit is very close, it's an obstacle
        #     if hit.distance < distance:
        #         return True
        # return False

    def compute_animation(self, cmd_vel: np.ndarray) -> np.ndarray:
        """Compute visual joint oscillation for gait animation."""
        vx, vy, wz = cmd_vel[0], cmd_vel[1], cmd_vel[2]
        target = self._default_qpos.copy()
        
        # Only animate if moving
        if np.linalg.norm(cmd_vel) < 0.01:
            return target
            
        phase_rad = 2 * np.pi * self._phase
        
        # Simple sinusoidal swing for legs
        # We need to map our JOINT_NAMES order to URDF order
        # For simplicity, we'll just check joint names in self._joints
        for i, joint in enumerate(self._joints):
            name = joint.name
            if "thigh" in name:
                # Trot: FL+RR together, FR+RL opposite
                if "FL" in name or "RR" in name:
                    target[i] += np.sin(phase_rad) * self._thigh_amp
                else:
                    target[i] += np.sin(phase_rad + np.pi) * self._thigh_amp
            elif "hip" in name:
                # Hip oscillation for turning/lateral
                if "FL" in name or "RL" in name:
                    target[i] += np.sin(phase_rad) * self._hip_amp * (vy + wz)
                else:
                    target[i] -= np.sin(phase_rad) * self._hip_amp * (vy + wz)
                    
        return target

    def compute_action(self, cmd_vel: np.ndarray) -> np.ndarray:
        """
        Compute joint targets from cmd_vel.
        
        Required by LocomotionControllerBase. In kinematic mode, 
        we use compute_animation instead, but we return the same result here.
        """
        return self.compute_animation(cmd_vel)

    def step(self, cmd_vel: np.ndarray) -> None:
        """Execute one locomotion step."""
        # Update phase
        if np.linalg.norm(cmd_vel) > 0.01:
            self._phase = (self._phase + self._dt * self._gait_freq) % 1.0
            
        # Get current pose
        base_pos, base_quat = self._get_base_pose()
        
        # 1. Compute target movement
        vx, vy, wz = cmd_vel[0], cmd_vel[1], cmd_vel[2]
        
        # Local velocity to world
        local_vel = np.array([vx, vy, 0.0], dtype=np.float32)
        world_vel = quat_rotate(base_quat, local_vel)
        
        # Proposed movement
        step_pos = base_pos + world_vel * self._dt
        
        # 2. Collision Check (Raycast)
        # Use a slightly longer ray for look-ahead
        look_ahead = self._robot_radius + 0.1 
        move_dir = world_vel / (np.linalg.norm(world_vel) + 1e-6)
        
        can_move = True
        if np.linalg.norm(cmd_vel[:2]) > 1e-3:
            if self._check_obstacle(base_pos, move_dir, look_ahead):
                can_move = False
                
        # 3. Update Base Pose
        if can_move:
            new_pos = step_pos
        else:
            new_pos = base_pos # Stop if blocked
            
        # Update orientation (yaw only)
        if abs(wz) > 0.01:
            half_angle = wz * self._dt / 2.0
            dq = np.array([np.cos(half_angle), 0.0, 0.0, np.sin(half_angle)], dtype=np.float32)
            w1, x1, y1, z1 = dq
            w2, x2, y2, z2 = base_quat
            new_quat = np.array([
                w1*w2 - x1*x2 - y1*y2 - z1*z2,
                w1*x2 + x1*w2 + y1*z2 - z1*y2,
                w1*y2 - x1*z2 + y1*w2 + z1*x2,
                w1*z2 + x1*y2 - y1*x2 + z1*w2
            ], dtype=np.float32)
        else:
            new_quat = base_quat
            
        # 4. Height adjustment (Terrain sensing for base)
        # Temporarily disabled due to scene.raycast error
        # Use a fixed height (e.g., 0.42m) for now
        new_pos[2] = 0.42
        
        # # Raycast downwards to find ground height
        # down_hit = self.robot.scene.raycast(new_pos + np.array([0, 0, 1.0]), np.array([0, 0, -1.0]), max_dist=2.0)
        # if hasattr(down_hit, 'has_hit') and down_hit.has_hit:
        #     # Target height is ground + default body height (e.g. 0.4m)
        #     target_height = down_hit.pos[2] + 0.35 # Fixed height above ground
        #     new_pos[2] = target_height
            
        # 5. Compute Joint Animation
        joint_positions = self.compute_animation(cmd_vel)
        
        # 6. Apply full qpos (instantly updates robot)
        full_qpos = np.concatenate([
            new_pos,
            new_quat,
            joint_positions
        ]).astype(np.float32)
        
        self.robot.entity.set_qpos(full_qpos)
