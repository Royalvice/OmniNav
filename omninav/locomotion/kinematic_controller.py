"""
Kinematic Gait Controller for Go2 Quadruped (Ultra-Simple Game-Style)

Simplest possible implementation for navigation algorithm testing:
- Pure kinematic control (no physics, no gravity)
- Direct joint angle animation (no IK)
- Raycast-based terrain following (for stairs)
- Guaranteed stability (won't fall, won't rotate)
- Minimal code, maximum performance

Design Philosophy:
- Base: Manual position/orientation control (pure kinematic)
- Legs: Simple sine wave joint animation
- Terrain: Raycast to adjust height
- Goal: Navigation algorithm testing, NOT Sim2Real

This is the SIMPLEST possible quadruped controller.
"""

from typing import TYPE_CHECKING, Tuple
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


@LOCOMOTION_REGISTRY.register("kinematic_gait")
class KinematicController(LocomotionControllerBase):
    """
    Ultra-simple game-style kinematic controller.
    
    No IK, no physics, just:
    1. Move base manually (kinematic)
    2. Animate legs with sine waves
    3. Raycast for terrain height
    4. Apply via set_qpos
    
    That's it!
    """
    
    CONTROLLER_TYPE = "kinematic_gait"
    
    def __init__(self, cfg: DictConfig, robot: "RobotBase"):
        """Initialize controller."""
        super().__init__(cfg, robot)
        
        # Parameters
        self._gait_freq = cfg.get("gait_frequency", 2.0)
        self._body_height = cfg.get("body_height", 0.28)
        self._dt = cfg.get("dt", 0.01)
        self._vel_alpha = cfg.get("velocity_alpha", 0.15)
        
        # Terrain adaptation parameters
        self._max_climb_height = cfg.get("max_climb_height", 0.21)  # 3/4 of body_height (0.28 * 0.75)
        self._forward_raycast_dist = cfg.get("forward_raycast_dist", 0.3)  # Look ahead distance
        self._collision_check_dist = cfg.get("collision_check_dist", 0.25)  # Collision detection distance
        
        # State
        self._phase = 0.0
        self._smoothed_cmd_vel = np.zeros(3, dtype=np.float32)
        self._current_yaw = 0.0
        self._current_height = self._body_height
        self._last_pos = None  # Track last position to prevent drift
        self._step_count = 0  # For debug output
        
        # IMPORTANT: We need to use URDF-ordered joints for set_qpos!
        # The robot.default_dof_pos_urdf property gives us the correct order
        self._default_joints = None  # Will be set in reset() after robot is built
        
    def reset(self) -> None:
        """Reset controller state."""
        self._phase = 0.0
        self._smoothed_cmd_vel.fill(0.0)
        self._current_yaw = 0.0
        self._current_height = self._body_height
        self._last_pos = None
        
        # Get default joints in URDF order (correct for set_qpos)
        if hasattr(self.robot, 'default_dof_pos_urdf'):
            self._default_joints = self.robot.default_dof_pos_urdf.copy()
        else:
            # Fallback: assume JOINT_NAMES order matches URDF
            # This will be wrong if orders differ!
            print("[WARNING] Robot doesn't have default_dof_pos_urdf, using fallback")
            self._default_joints = np.array([
                0.0, 0.8, -1.5,   # FL
                0.0, 0.8, -1.5,   # FR
                0.0, 1.0, -1.5,   # RL
                0.0, 1.0, -1.5,   # RR
            ], dtype=np.float32)
    
    def _get_base_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current base position and orientation."""
        pos = self.robot.entity.get_pos()
        quat = self.robot.entity.get_quat()
        
        if hasattr(pos, 'cpu'):
            pos = pos.cpu().numpy()
        if hasattr(quat, 'cpu'):
            quat = quat.cpu().numpy()
        if pos.ndim > 1:
            pos = pos[0]
        if quat.ndim > 1:
            quat = quat[0]
            
        return np.array(pos, dtype=np.float32), np.array(quat, dtype=np.float32)
    
    def _yaw_to_quat(self, yaw: float) -> np.ndarray:
        """Convert yaw to quaternion (wxyz, rotation around Z)."""
        half_yaw = yaw * 0.5
        return np.array([
            np.cos(half_yaw),  # w
            0.0,               # x
            0.0,               # y
            np.sin(half_yaw)   # z
        ], dtype=np.float32)
    
    def _raycast_ground(self, x: float, y: float) -> float:
        """Raycast downward to find ground height."""
        try:
            ray_origin = np.array([x, y, self._current_height + 0.5])
            ray_dir = np.array([0.0, 0.0, -1.0])
            hit = self.robot.scene.raycast(ray_origin, ray_dir, max_dist=1.0)
            if hasattr(hit, 'has_hit') and hit.has_hit:
                return float(hit.pos[2])
        except:
            pass
        return 0.0
    
    def _check_forward_obstacle(self, pos: np.ndarray, forward_dir: np.ndarray) -> Tuple[bool, float]:
        """
        Check for obstacles ahead using forward raycast.
        
        Returns:
            (has_obstacle, obstacle_height): 
                - has_obstacle: True if there's an obstacle blocking the path
                - obstacle_height: Height of the obstacle (0.0 if no obstacle)
        """
        try:
            # Cast multiple rays at different heights to detect obstacles
            # Ray 1: At foot level (detect steps/stairs)
            ray_origin_low = pos + np.array([0.0, 0.0, 0.05])
            # Ray 2: At body level (detect walls)
            ray_origin_mid = pos + np.array([0.0, 0.0, self._body_height * 0.5])
            
            # Check both rays
            for i, ray_origin in enumerate([ray_origin_low, ray_origin_mid]):
                hit = self.robot.scene.raycast(
                    ray_origin, 
                    forward_dir, 
                    max_dist=self._forward_raycast_dist
                )
                
                if hasattr(hit, 'has_hit') and hit.has_hit:
                    hit_pos = hit.pos if isinstance(hit.pos, np.ndarray) else np.array(hit.pos)
                    
                    # Calculate obstacle height relative to current ground
                    ground_z = self._raycast_ground(pos[0], pos[1])
                    obstacle_height = hit_pos[2] - ground_z
                    
                    # Debug output
                    ray_type = "FOOT" if i == 0 else "BODY"
                    print(f"[{ray_type}] Hit! pos={hit_pos}, ground_z={ground_z:.3f}, obs_h={obstacle_height:.3f}")
                    
                    # If obstacle is too high, it's a wall (block movement)
                    if obstacle_height > self._max_climb_height:
                        print(f"  → BLOCKED (too high: {obstacle_height:.3f} > {self._max_climb_height:.3f})")
                        return True, obstacle_height
                    
                    # If it's a climbable step, return the height
                    if obstacle_height > 0.02:  # Ignore tiny bumps
                        print(f"  → CLIMBABLE (height: {obstacle_height:.3f})")
                        return False, obstacle_height
            
        except Exception as e:
            print(f"[ERROR] Raycast failed: {e}")
        
        return False, 0.0
    
    def _check_collision(self, pos: np.ndarray, forward_dir: np.ndarray) -> bool:
        """
        Check if robot is colliding with obstacles (close-range detection).
        
        Returns:
            True if collision detected (should stop movement)
        """
        try:
            # Cast ray from body center forward at very close range
            ray_origin = pos + np.array([0.0, 0.0, self._body_height * 0.5])
            hit = self.robot.scene.raycast(
                ray_origin, 
                forward_dir, 
                max_dist=self._collision_check_dist
            )
            
            if hasattr(hit, 'has_hit') and hit.has_hit:
                hit_pos = hit.pos if isinstance(hit.pos, np.ndarray) else np.array(hit.pos)
                dist = np.linalg.norm(hit_pos[:2] - pos[:2])
                print(f"[COLLISION] Detected at dist={dist:.3f}m, stopping!")
                return True
                
        except Exception as e:
            print(f"[ERROR] Collision check failed: {e}")
        
        return False
    
    def _animate_legs(self, phase: float, speed: float) -> np.ndarray:
        """
        Simplest possible leg animation: just swing thighs back and forth.
        
        CRITICAL: URDF joint order is NOT leg-by-leg!
        URDF order: [FL_hip, FR_hip, RL_hip, RR_hip, FL_thigh, FR_thigh, RL_thigh, RR_thigh, FL_calf, FR_calf, RL_calf, RR_calf]
        
        Indices:
        - Hips: 0=FL, 1=FR, 2=RL, 3=RR
        - Thighs: 4=FL, 5=FR, 6=RL, 7=RR
        - Calfs: 8=FL, 9=FR, 10=RL, 11=RR
        
        Trot gait: FL+RR move together, FR+RL move together.
        """
        if self._default_joints is None:
            return np.zeros(12, dtype=np.float32)
        
        joints = self._default_joints.copy()
        
        if speed < 0.02:  # Standing still
            return joints
        
        # Amplitude scales with speed
        amp = min(speed / 0.5, 1.0) * 0.3
        
        # Trot pattern: FL+RR (phase), FR+RL (phase+0.5)
        # Leg indices in URDF: FL=0, FR=1, RL=2, RR=3
        leg_phases = [
            (0, 0.0),   # FL: phase offset 0.0
            (1, 0.5),   # FR: phase offset 0.5
            (2, 0.5),   # RL: phase offset 0.5
            (3, 0.0),   # RR: phase offset 0.0
        ]
        
        for leg_idx, phase_offset in leg_phases:
            leg_phase = (phase + phase_offset) % 1.0
            
            # Simple sine wave
            swing = np.sin(leg_phase * 2 * np.pi) * amp
            
            # URDF indices:
            # hip_idx = leg_idx (0-3)
            # thigh_idx = leg_idx + 4 (4-7)
            # calf_idx = leg_idx + 8 (8-11)
            
            # Only animate thigh joint
            thigh_idx = leg_idx + 4
            joints[thigh_idx] += swing
        
        return joints
    
    def step(self, cmd_vel: np.ndarray) -> None:
        """
        Ultra-simple kinematic step with collision detection and stair climbing.
        
        1. Smooth velocity input
        2. Update phase
        3. Check for collisions and obstacles
        4. Move base manually (kinematic) if safe
        5. Raycast for terrain height and obstacle climbing
        6. Animate legs with sine waves
        7. Apply via set_qpos
        """
        # 1. Smooth velocity
        self._smoothed_cmd_vel = (
            self._vel_alpha * cmd_vel + 
            (1.0 - self._vel_alpha) * self._smoothed_cmd_vel
        )
        vx, vy, wz = self._smoothed_cmd_vel
        speed = np.sqrt(vx**2 + vy**2)
        
        # 2. Update phase (only if moving)
        if speed > 0.02 or abs(wz) > 0.01:
            self._phase = (self._phase + self._dt * self._gait_freq) % 1.0
        
        # 3. Get current pose
        base_pos, base_quat = self._get_base_pose()
        
        # Initialize last_pos on first call
        if self._last_pos is None:
            self._last_pos = base_pos.copy()
            self._current_yaw = self._quat_to_yaw(base_quat)
        
        # 4. Calculate movement direction in world frame
        vel_body = np.array([vx, vy, 0.0])
        vel_world = quat_rotate(base_quat, vel_body)
        
        # Normalize to get forward direction (for raycasting)
        forward_dir = vel_world.copy()
        if np.linalg.norm(forward_dir) > 0.001:
            forward_dir = forward_dir / np.linalg.norm(forward_dir)
        else:
            forward_dir = np.array([1.0, 0.0, 0.0])  # Default forward
        
        # 5. Collision and obstacle detection
        is_blocked = False
        obstacle_height = 0.0
        
        if speed > 0.01:  # Only check when moving forward
            # Check for collision (close range)
            if self._check_collision(base_pos, forward_dir):
                is_blocked = True
            
            # Check for obstacles ahead (medium range)
            has_wall, obs_height = self._check_forward_obstacle(base_pos, forward_dir)
            if has_wall:
                is_blocked = True
            else:
                obstacle_height = obs_height
        
        # 6. Move base
        if speed > 0.01 or abs(wz) > 0.01:
            if not is_blocked:
                # Move base forward (kinematic)
                new_pos = self._last_pos + vel_world * self._dt
                
                # Update yaw
                self._current_yaw += wz * self._dt
                new_quat = self._yaw_to_quat(self._current_yaw)
                
                # Store for next frame
                self._last_pos = new_pos.copy()
            else:
                # Collision or wall ahead:
                # slightly push the robot backwards along the forward direction
                # so that it won't get stuck "inside" the obstacle.
                backoff_dist = self._collision_check_dist * 0.5
                new_pos = base_pos - forward_dir * backoff_dist
                self._last_pos = new_pos.copy()
                new_quat = self._yaw_to_quat(self._current_yaw)
        else:
            # No input: stay at last position (prevent drift)
            new_pos = self._last_pos.copy()
            new_quat = self._yaw_to_quat(self._current_yaw)
        
        # 7. Raycast for terrain height (smooth)
        ground_z = self._raycast_ground(new_pos[0], new_pos[1])
        
        # If there's an obstacle ahead, lift body to climb over it
        if obstacle_height > 0.02:
            # Smoothly raise body to clear the obstacle.
            # Limit how much extra height we add so the robot doesn't
            # jump too high, but still clears reasonable steps.
            climb_h = min(obstacle_height, self._max_climb_height)
            target_height = ground_z + self._body_height + climb_h * 0.8
            if abs(target_height - self._current_height) > 0.01:
                print(f"[CLIMB] Lifting body: {self._current_height:.3f} → {target_height:.3f} (obs_h={obstacle_height:.3f})")
        else:
            # Normal height
            target_height = ground_z + self._body_height
        
        # Smooth height transition
        self._current_height += (target_height - self._current_height) * 0.1
        new_pos[2] = self._current_height
        
        # 8. Animate legs
        joint_angles = self._animate_legs(self._phase, speed)
        
        # 9. Apply full qpos (pure kinematic, no physics)
        full_qpos = np.concatenate([
            new_pos,       # 3: position
            new_quat,      # 4: quaternion (wxyz)
            joint_angles   # 12: joint angles
        ])
        self.robot.entity.set_qpos(full_qpos)
    
    def _quat_to_yaw(self, quat: np.ndarray) -> float:
        """Extract yaw from quaternion (wxyz)."""
        w, x, y, z = quat
        return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    
    def compute_action(self, cmd_vel: np.ndarray) -> np.ndarray:
        """Not used in kinematic mode."""
        return np.zeros(12, dtype=np.float32)


