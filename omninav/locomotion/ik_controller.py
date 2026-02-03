"""
IK-based Gait Controller for Go2

Converts cmd_vel to joint positions using inverse kinematics and gait planning.
Uses Genesis inverse_kinematics_multilink API with WORLD coordinate targets.
"""

from typing import TYPE_CHECKING, Dict, List, Tuple
import numpy as np
from omegaconf import DictConfig

from omninav.locomotion.base import LocomotionControllerBase
from omninav.core.registry import LOCOMOTION_REGISTRY

if TYPE_CHECKING:
    from omninav.robots.base import RobotBase


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Rotate vector v by quaternion q (wxyz format).
    
    Args:
        q: Quaternion [w, x, y, z]
        v: Vector [x, y, z]
    
    Returns:
        Rotated vector
    """
    w, x, y, z = q[0], q[1], q[2], q[3]
    
    # Quaternion rotation: v' = q * v * q^(-1)
    # Using the formula: v' = v + 2*w*(cross(xyz, v)) + 2*cross(xyz, cross(xyz, v))
    xyz = np.array([x, y, z])
    t = 2.0 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)


@LOCOMOTION_REGISTRY.register("ik_controller")
class IKController(LocomotionControllerBase):
    """
    IK-based gait controller for Go2 quadruped.
    
    Implements a simple trot gait using Bezier curve foot trajectories
    and Genesis inverse kinematics API.
    
    IMPORTANT: All IK targets are in WORLD coordinates.
    
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
    
    # State constants
    STATE_WALK = 0
    STATE_STAND = 1

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
        self._step_height = cfg.get("step_height", 0.04)
        self._step_length = cfg.get("step_length", 0.10)
        self._gait_frequency = cfg.get("gait_frequency", 2.0)
        self._body_height = cfg.get("body_height", 0.30)  # Standing height from ground
        
        # Leg geometry (relative to body center)
        # These are the hip joint positions in body frame
        self._hip_offsets = np.array([
            [+0.1881, +0.04675, 0.0],  # FL
            [+0.1881, -0.04675, 0.0],  # FR
            [-0.1881, +0.04675, 0.0],  # RL
            [-0.1881, -0.04675, 0.0],  # RR
        ], dtype=np.float32)
        
        # Foot link names in Go2 URDF (configurable - calf or foot links)
        self._foot_link_names = cfg.get("foot_links", [
            "FL_foot", "FR_foot", "RL_foot", "RR_foot"
        ])
        
        # Offset from calf link to foot (used if targeting calf links)
        foot_offset_cfg = cfg.get("foot_offset", [0.0, 0.0, -0.213])
        self._foot_offset = np.array(foot_offset_cfg, dtype=np.float32)
        
        # State
        self._phase = 0.0
        self._dt = cfg.get("dt", 0.01)
        self._state = self.STATE_WALK
        
        # Default standing foot positions (in BODY FRAME, Z pointing up)
        # Feet are directly below hips at body_height distance
        self._default_foot_pos_body = self._hip_offsets.copy()
        self._default_foot_pos_body[:, 2] = -self._body_height
        
        # Cache for foot links (initialized on first use)
        self._foot_links = None
        
        # Velocity threshold for stopping gait (m/s)
        self._vel_threshold = cfg.get("vel_threshold", 0.02)
        
        # Ground height smoothing (low-pass filter)
        self._last_ground_z = np.zeros(4, dtype=np.float32)
        self._ground_alpha = cfg.get("ground_smoothing", 0.2) # Filter factor
        
        # Target Locking (for Stand state - strictly lock JOINTS)
        self._stand_joints = None
        self._first_run = True
        
        # Debug flag
        self._debug = cfg.get("debug", False)
        
    def reset(self) -> None:
        """Reset gait phase."""
        self._phase = 0.0
        self._foot_links = None
        self._last_ground_z.fill(0.0)
        self._state = self.STATE_WALK
        self._stand_joints = None
        self._first_run = True
    
    def _get_foot_links(self):
        """Get foot link objects, caching them for efficiency."""
        if self._foot_links is None:
            self._foot_links = []
            for name in self._foot_link_names:
                link = self.robot.entity.get_link(name)
                if link is None:
                    print(f"[IK] Warning: Link '{name}' not found!")
                self._foot_links.append(link)
            if self._debug:
                print(f"[IK] Using foot links: {self._foot_link_names}")
        return self._foot_links
    
    def _get_base_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get robot base position and orientation.
        
        Returns:
            (position, quaternion) both as numpy arrays
        """
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
    
    def _body_to_world(self, local_pos: np.ndarray, base_pos: np.ndarray, base_quat: np.ndarray) -> np.ndarray:
        """
        Transform position from body frame to world frame.
        
        Args:
            local_pos: Position in body frame (3,)
            base_pos: Robot base position in world frame (3,)
            base_quat: Robot base quaternion [w,x,y,z] (4,)
        
        Returns:
            Position in world frame (3,)
        """
        # Rotate local position by base orientation and add base position
        rotated = quat_rotate(base_quat, local_pos)
        return rotated + base_pos
    
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
        end: np.ndarray,
        height: float
    ) -> np.ndarray:
        """
        Compute Bezier curve for swing phase.
        
        Args:
            t: Normalized time in swing phase [0, 1]
            start: Starting foot position (world frame)
            end: Ending foot position (world frame)
            height: Step height (added to Z)
        
        Returns:
            Foot position at time t (world frame)
        """
        # Cubic Bezier with lift in Z direction
        P0 = start
        P1 = start + np.array([0, 0, height * 0.6])
        P2 = end + np.array([0, 0, height * 0.6])
        P3 = end
        
        pos = (
            (1 - t) ** 3 * P0 +
            3 * (1 - t) ** 2 * t * P1 +
            3 * (1 - t) * t ** 2 * P2 +
            t ** 3 * P3
        )
        return pos
    
    def _raycast_terrain(self, leg_idx: int) -> float:
        """
        Query ground height under a foot using sensors.
        
        Args:
            leg_idx: Index of leg (0:FL, 1:FR, 2:RL, 3:RR)
            
        Returns:
            Measured ground height (Z coordinate) in world frame.
            Falls back to 0.0 if sensor is missing or out of range.
        """
        sensor_names = ["FL_foot_sensor", "FR_foot_sensor", "RL_foot_sensor", "RR_foot_sensor"]
        name = sensor_names[leg_idx]
        
        raw_z = 0.0
        if name in self.robot.sensors:
            data = self.robot.sensors[name].get_data()
            ranges = data.get("ranges", [])
            if len(ranges) > 0 and ranges[0] < self.robot.sensors[name]._max_range:
                # Sensor points down from link. height = link_z - sensed_dist
                foot_links = self._get_foot_links()
                link = foot_links[leg_idx]
                link_pos = link.get_pos()
                if hasattr(link_pos, 'cpu'): link_pos = link_pos.cpu().numpy()
                if link_pos.ndim > 1: link_pos = link_pos[0]
                
                # Sensed ground Z
                raw_z = link_pos[2] - ranges[0]
        
        # Low-pass filter for ground height to reduce jitter
        smoothed_z = self._ground_alpha * raw_z + (1.0 - self._ground_alpha) * self._last_ground_z[leg_idx]
        self._last_ground_z[leg_idx] = smoothed_z
        
        return smoothed_z

    def _update_state(self, cmd_vel: np.ndarray, base_pos: np.ndarray, base_quat: np.ndarray) -> None:
        """
        Update controller state (WALK <-> STAND) based on velocity command.
        """
        is_moving = np.linalg.norm(cmd_vel) > self._vel_threshold
        
        if self._state == self.STATE_WALK:
            if not is_moving:
                # Transition to STAND: LOCK JOINTS
                # Instead of locking foot targets (which causes IK jitter due to base drift),
                # we directly capture the current joint angles and hold them.
                # This makes the robot "stiff" in standing, but perfectly stable.
                
                qpos = self.robot.entity.get_qpos()
                if hasattr(qpos, 'cpu'): qpos = qpos.cpu().numpy()
                if qpos.ndim > 1: qpos = qpos[0]
                
                # Capture current joint angles (skip base 7)
                self._stand_joints = qpos[7:].copy()
                
                self._state = self.STATE_STAND
                if self._debug: print("[IK] Transition to STAND (Locked JOINTS)")
                
        elif self._state == self.STATE_STAND:
            if is_moving:
                # Transition to WALK
                self._state = self.STATE_WALK
                self._stand_joints = None
                if self._debug: print("[IK] Transition to WALK")

    def _compute_foot_targets_world(
        self, 
        cmd_vel: np.ndarray,
        base_pos: np.ndarray,
        base_quat: np.ndarray
    ) -> np.ndarray:
        """
        Compute target foot positions in WORLD coordinates.
        Only used in WALK state.
        """
        vx, vy, wz = cmd_vel[0], cmd_vel[1], cmd_vel[2]
        
        # Get trot phases
        phase_FL_RR, phase_FR_RL = self._get_trot_phases()
        
        # Compute step displacement in body frame
        step_x = vx * self._step_length
        step_y = vy * self._step_length
        
        foot_targets_world = np.zeros((4, 3), dtype=np.float32)
        
        for leg_idx in range(4):
            # Sensed ground height for this foot
            ground_z = self._raycast_terrain(leg_idx)
            
            # Determine which phase group this leg belongs to
            if leg_idx in [self.LEG_FL, self.LEG_RR]:
                phase = phase_FL_RR
            else:
                phase = phase_FR_RL
            
            # Default foot position in body frame
            default_pos_body = self._default_foot_pos_body[leg_idx].copy()
            
            # Compute swing/stance in body frame
            if phase < 0.5:
                # Swing phase
                swing_t = phase * 2  # Normalize to [0, 1]
                
                # Start and end positions in body frame
                start_body = default_pos_body - np.array([step_x / 2, step_y / 2, 0])
                end_body = default_pos_body + np.array([step_x / 2, step_y / 2, 0])
                
                # Convert to world frame
                start_world = self._body_to_world(start_body, base_pos, base_quat)
                end_world = self._body_to_world(end_body, base_pos, base_quat)
                
                # Snap extremes to ground
                start_world[2] = ground_z
                end_world[2] = ground_z
                
                foot_targets_world[leg_idx] = self._bezier_swing_trajectory(
                    swing_t, start_world, end_world, self._step_height
                )
            else:
                # Stance phase
                stance_t = (phase - 0.5) * 2  # Normalize to [0, 1]
                
                # Linear interpolation in body frame
                start_body = default_pos_body + np.array([step_x / 2, step_y / 2, 0])
                end_body = default_pos_body - np.array([step_x / 2, step_y / 2, 0])
                
                pos_body = start_body + stance_t * (end_body - start_body)
                
                # Convert to world frame and snap to ground
                target = self._body_to_world(pos_body, base_pos, base_quat)
                target[2] = ground_z
                foot_targets_world[leg_idx] = target
        
        return foot_targets_world
    
    def compute_action(self, cmd_vel: np.ndarray) -> np.ndarray:
        """
        Compute joint positions from cmd_vel.
        """
        base_pos, base_quat = self._get_base_pose()
        
        # --- Handle First Run (Initialization) ---
        if self._first_run:
            # On first run, we want to start in a CLEAN standing pose, not capture a falling one.
            # So we pretend we are walking with 0 velocity (neutral stance), compute IK, and lock it.
            if self._debug: print("[IK] First Run: Computing Default Stand Pose")
            
            # Compute targets for neutral stance (cmd_vel=0)
            foot_targets = self._compute_foot_targets_world(np.zeros(3), base_pos, base_quat)
            foot_links = self._get_foot_links()
            
            target_links = []
            target_pos_list = []
            for leg_idx in range(4):
                link = foot_links[leg_idx]
                if link is None: continue
                target_links.append(link)
                target_pos_list.append(foot_targets[leg_idx] - self._foot_offset)
                
            if target_links:
                try:
                    q_sol = self.robot.entity.inverse_kinematics_multilink(
                        links=target_links,
                        poss=target_pos_list,
                        rot_mask=[True, True, True],
                        max_iterations=30,
                        lr=0.3
                    )
                    if hasattr(q_sol, 'cpu'): q_sol = q_sol.cpu().numpy()
                    if q_sol.ndim > 1: q_sol = q_sol[0]
                    self._stand_joints = q_sol[7:]
                except Exception:
                    self._stand_joints = self.robot.default_dof_pos.copy()
            else:
                self._stand_joints = self.robot.default_dof_pos.copy()

            self._state = self.STATE_STAND
            self._first_run = False
            return self._stand_joints

        # Update State Machine
        self._update_state(cmd_vel, base_pos, base_quat)
        
        # If in STAND state, return locked joints directly!
        # Bypass IK entirely to prevent feedback loops.
        if self._state == self.STATE_STAND and self._stand_joints is not None:
             return self._stand_joints
        
        # --- WALK STATE (Compute IK) ---
        
        foot_targets = self._compute_foot_targets_world(cmd_vel, base_pos, base_quat)
        
        foot_links = self._get_foot_links()
        
        target_links = []
        target_pos_list = []
        
        for leg_idx in range(4):
            link = foot_links[leg_idx]
            if link is None: continue
            target_links.append(link)
            target_pos = foot_targets[leg_idx] - self._foot_offset
            target_pos_list.append(target_pos)
        
        if not target_links:
            return self.robot.default_dof_pos.copy()
        
        try:
            q_sol = self.robot.entity.inverse_kinematics_multilink(
                links=target_links,
                poss=target_pos_list,
                rot_mask=[True, True, True],
                max_iterations=30,
                lr=0.3, # Learning rate for IK
            )
            
            if hasattr(q_sol, 'cpu'): q_sol = q_sol.cpu().numpy()
            if q_sol.ndim > 1: q_sol = q_sol[0]
            
            # Extract joints
            return q_sol[7:]
            
        except Exception:
            init_qpos = self.robot.entity.get_qpos()
            if hasattr(init_qpos, 'cpu'): init_qpos = init_qpos.cpu().numpy()
            if init_qpos.ndim > 1: init_qpos = init_qpos[0]
            return init_qpos[7:]
    
    def _apply_kinematic_motion(self, cmd_vel: np.ndarray) -> None:
        """
        Directly move robot base position (kinematic control).
        """
        base_pos, base_quat = self._get_base_pose()
        vx, vy, wz = cmd_vel[0], cmd_vel[1], cmd_vel[2]
        
        local_vel = np.array([vx, vy, 0.0], dtype=np.float32)
        world_vel = quat_rotate(base_quat, local_vel)
        new_pos = base_pos + world_vel * self._dt
        
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
        
        self.robot.entity.set_pos(new_pos)
        self.robot.entity.set_quat(new_quat)
        return new_pos, new_quat
            
    def step(self, cmd_vel: np.ndarray) -> None:
        """
        Execute one locomotion control step.
        """
        mode = self.cfg.get("mode", "physics")
        
        # Advance phase only if moving or near a phase transition (for settling)
        # To keep it simple, stop phase advancement if velocity is low
        is_moving = np.linalg.norm(cmd_vel) > self._vel_threshold
        if is_moving:
            self._phase = (self._phase + self._dt * self._gait_frequency) % 1.0
        else:
            # Settle phase towards 0.0 or 0.5 where feet are on ground
            # For now, just stop phase
            pass
            
        base_pos, base_quat = self._get_base_pose()
        joint_positions = self.compute_action(cmd_vel)
        
        if mode == "kinematic":
            if is_moving:
                new_pos, new_quat = self._apply_kinematic_motion(cmd_vel)
            else:
                new_pos, new_quat = base_pos, base_quat
            
            full_qpos = np.concatenate([new_pos, new_quat, joint_positions]).astype(np.float32)
            self.robot.entity.set_qpos(full_qpos)
        else:
            self.robot.control_joints_position(joint_positions)
