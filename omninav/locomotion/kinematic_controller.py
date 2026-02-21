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
        
        # Core parameters
        self._body_height = cfg.get("body_height", 0.28)
        self._dt = cfg.get("dt", 0.01)
        self._vel_alpha = cfg.get("velocity_alpha", 0.15)
        
        # Terrain & collision
        self._max_climb_height = cfg.get("max_climb_height", 0.21)
        self._front_stop_dist = cfg.get("front_stop_dist", 0.02)
        self._front_backoff_dist = cfg.get("front_backoff_dist", 0.01)
        self._front_slow_far = cfg.get("front_slow_far", 0.5)
        self._front_slow_mid = cfg.get("front_slow_mid", 0.3)
        self._front_slow_scale_1 = cfg.get("front_slow_scale_1", 0.7)
        self._front_slow_scale_2 = cfg.get("front_slow_scale_2", 0.4)
        
        # Side collision
        side_x = cfg.get("side_sensor_x", 0.0)
        side_y = cfg.get("side_sensor_y", 0.25)
        side_z = cfg.get("side_sensor_z", 0.2)
        self._side_sensor_offset_left = (side_x, side_y, side_z)
        self._side_sensor_offset_right = (side_x, -side_y, side_z)
        self._side_stop_dist = cfg.get("side_stop_dist", 0.15)
        
        # Posture
        self._roll_gain = cfg.get("roll_gain", 0.6)
        self._robot_width = cfg.get("robot_width", None)
        
        # Gait
        self._base_freq = cfg.get("base_freq", 1.5)
        self._freq_gain = cfg.get("freq_gain", 1.0)
        self._max_gait_speed = cfg.get("max_gait_speed", 1.0)
        self._gait_start_threshold = cfg.get("gait_start_threshold", 0.05)
        self._gait_stop_threshold = cfg.get("gait_stop_threshold", 0.02)
        self._phase_transition_alpha = cfg.get("phase_transition_alpha", 0.1)
        
        # Foot locking
        self._foot_lock_support_start = cfg.get("foot_lock_support_start", 0.25)
        self._foot_lock_support_end = cfg.get("foot_lock_support_end", 0.75)
        self._foot_lock_stair_threshold = cfg.get("foot_lock_stair_threshold", 0.05)
        self._stair_lift_multiplier = cfg.get("stair_lift_multiplier", 2.5)  # How much to lift legs on stairs
        
        # Initial height
        self._foot_clearance = cfg.get("foot_clearance", 0.02)
        
        # Foot sensor offsets
        front_x = cfg.get("front_foot_x", 0.25)
        rear_x = cfg.get("rear_foot_x", -0.25)
        foot_y = cfg.get("foot_y", 0.12)
        sensor_z = cfg.get("foot_sensor_z", 0.35)
        self._foot_offsets = {
            "FL": (front_x, foot_y, sensor_z),
            "FR": (front_x, -foot_y, sensor_z),
            "RL": (rear_x, foot_y, sensor_z),
            "RR": (rear_x, -foot_y, sensor_z),
        }
        
        # State
        self._phase = 0.0
        self._target_phase = 0.0
        self._smoothed_cmd_vel = np.zeros(3, dtype=np.float32)
        self._current_yaw = 0.0
        self._current_height = self._body_height
        self._current_pitch = 0.0
        self._current_roll = 0.0
        self._last_pos = None
        self._default_joints = None
        
        # Sensors
        self._foot_sensors = {}
        self._front_sensor = None
        self._left_sensor = None
        self._right_sensor = None
        
        # Sensor data cache
        self._foot_heights = {k: 0.0 for k in self._foot_offsets.keys()}
        self._front_blocked = False
        self._front_min_dist = None
        self._front_speed_scale = 1.0
        self._left_blocked = False
        self._right_blocked = False
        # Smoothed stair level to reduce jitter
        self._stairs_level = 0.0

    @property
    def required_sensors(self) -> dict:
        """Dynamically generate required sensors based on layout."""
        sensors = {}
        
        # 1. Foot sensors
        for name, offset in self._foot_offsets.items():
            sensors[name] = {
                "type": "raycaster",
                "resolution": 0.01,
                "size": [0.0, 0.0],
                "direction": [0.0, 0.0, -1.0],
                "pos_offset": list(offset),
                "return_points": True,  # needed for height calc
            }
            
        # 2. Front sensor (Forward fan)
        sensors["front_ray"] = {
            "type": "raycaster",
            "resolution": 0.1,
            "size": [0.4, 0.2],
            "direction": [0.0, 0.0, 1.0],
            "pos_offset": [0.35, 0.0, 0.15],
            "euler_offset": [0.0, 90.0, 0.0],
            "return_points": False,
        }
        
        # 3. Side sensors
        # Left
        side_dir_left = [0.0, 1.0, 0.0]
        sensors["side_left"] = {
            "type": "raycaster",
            "resolution": 0.1,
            "size": [0.0, 0.0],
            "direction": side_dir_left,
            "pos_offset": [0.0, self._side_sensor_offset_left[1], self._side_sensor_offset_left[2]],
            "return_points": False,
        }
        
        # Right
        side_dir_right = [0.0, -1.0, 0.0]
        sensors["side_right"] = {
            "type": "raycaster",
            "resolution": 0.1,
            "size": [0.0, 0.0],
            "direction": side_dir_right,
            "pos_offset": [0.0, self._side_sensor_offset_right[1], self._side_sensor_offset_right[2]],
            "return_points": False,
        }
        
        # Merge with config-defined sensors
        cfg_sensors = super().required_sensors
        sensors.update(cfg_sensors)
        return sensors

    def bind_sensors(self, sensors: dict) -> None:
        """Bind instantiated sensors."""
        super().bind_sensors(sensors)
        
        # Bind foot sensors
        self._foot_sensors = {}
        for name in self._foot_offsets.keys():
            if name in sensors:
                self._foot_sensors[name] = sensors[name]
        
        # Bind other sensors
        self._front_sensor = sensors.get("front_ray")
        self._left_sensor = sensors.get("side_left")
        self._right_sensor = sensors.get("side_right")

    # [REMOVED] add_sensors method

        
    def reset(self) -> None:
        """Reset controller state."""
        self._phase = 0.0
        self._target_phase = 0.0
        self._smoothed_cmd_vel.fill(0.0)
        self._current_yaw = 0.0
        self._current_pitch = 0.0
        self._current_roll = 0.0
        self._last_pos = None
        for k in self._foot_heights:
            self._foot_heights[k] = 0.0
        self._front_blocked = False
        self._front_speed_scale = 1.0
        self._left_blocked = False
        self._right_blocked = False
        
        # Get default joints
        if hasattr(self.robot, 'default_dof_pos_urdf'):
            self._default_joints = self.robot.default_dof_pos_urdf.copy()
        else:
            print("[WARNING] Robot doesn't have default_dof_pos_urdf, using fallback")
            self._default_joints = np.array([
                0.0, 0.8, -1.5,   # FL
                0.0, 0.8, -1.5,   # FR
                0.0, 1.0, -1.5,   # RL
                0.0, 1.0, -1.5,   # RR
            ], dtype=np.float32)
        
        # Initialize height from foot sensors
        if self._foot_sensors:
            self._update_foot_heights()
            max_foot_height = max(self._foot_heights.values()) if self._foot_heights.values() else 0.0
            self._current_height = max_foot_height + self._body_height + self._foot_clearance
        else:
            self._current_height = self._body_height + self._foot_clearance
    
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
        """Convert yaw to quaternion (wxyz format)."""
        half_yaw = yaw * 0.5
        return np.array([
            np.cos(half_yaw),  # w
            0.0,               # x
            0.0,               # y
            np.sin(half_yaw)   # z
        ], dtype=np.float32)
    
    def _update_foot_heights(self) -> None:
        """Update per-foot ground heights."""
        for name, sensor in self._foot_sensors.items():
            try:
                # Use standard get_data() interface
                data = sensor.get_data()
                
                # Try to get points first (best for height)
                if "points" in data:
                    pts = data["points"] # (B, N, 3)
                    if pts.ndim > 2:
                        pts = pts[0]
                    if pts.shape[0] > 0:
                        # Use z-coordinate of first valid point
                        self._foot_heights[name] = float(pts[0, 2])
                elif "ranges" in data:
                    # Fallback to ranges
                    rg = data["ranges"] # (B, N)
                    if rg.ndim > 1:
                        rg = rg[0]
                    if rg.size > 0:
                        dist = float(rg[0])
                        if dist < 5.0:
                            base_pos, _ = self._get_base_pose()
                            # Rough estimate: current base z - distance
                            # Note: this assumes vertical ray
                            self._foot_heights[name] = base_pos[2] - dist
            except Exception:
                continue
    
    def _update_front_block(self) -> None:
        """Update front blocked flag and speed scaling."""
        self._front_blocked = False
        self._front_min_dist = None
        self._front_speed_scale = 1.0
        if self._front_sensor is None:
            return
        try:
            data = self._front_sensor.get_data()
            if "ranges" not in data:
                return
            rg = data["ranges"] # (B, N)
            if rg.ndim > 1:
                rg = rg[0]
            ranges = rg.flatten()
            if ranges.size == 0:
                return
            d_min = float(np.min(ranges))
            self._front_min_dist = d_min
            
            if d_min <= self._front_stop_dist:
                self._front_blocked = True
                self._front_speed_scale = 0.0
            elif d_min <= self._front_slow_mid:
                self._front_speed_scale = self._front_slow_scale_2
            elif d_min <= self._front_slow_far:
                self._front_speed_scale = self._front_slow_scale_1
            else:
                self._front_speed_scale = 1.0
        except Exception:
            pass
    
    def _update_side_blocks(self) -> None:
        """Update left/right blocked flags using simple horizontal rays."""
        self._left_blocked = False
        self._right_blocked = False
        
        min_valid_dist = 0.05  # ignore self-collisions
        
        # Left sensor: block if any hit within side_stop_dist
        if self._left_sensor is not None:
            try:
                data = self._left_sensor.get_data()
                if "ranges" in data:
                    rg = data["ranges"]
                    if rg.ndim > 1:
                        rg = rg[0]
                    ranges = rg.flatten()
                if ranges.size > 0:
                    d_min = float(np.min(ranges))
                    if min_valid_dist <= d_min < self._side_stop_dist:
                        self._left_blocked = True
            except Exception:
                pass
        
        # Right sensor: block if any hit within side_stop_dist
        if self._right_sensor is not None:
            try:
                data = self._right_sensor.get_data()
                if "ranges" in data:
                    rg = data["ranges"]
                    if rg.ndim > 1:
                        rg = rg[0]
                    ranges = rg.flatten()
                if ranges.size > 0:
                    d_min = float(np.min(ranges))
                    if min_valid_dist <= d_min < self._side_stop_dist:
                        self._right_blocked = True
            except Exception:
                pass
    
    def _animate_legs(self, phase: float, speed: float, vx: float, is_climbing_stairs: bool, stair_height_factor: float) -> np.ndarray:
        """Animate legs with sine waves. Enhanced lift for stairs."""
        if self._default_joints is None:
            return np.zeros(12, dtype=np.float32)
        
        joints = self._default_joints.copy()
        
        if speed < self._gait_stop_threshold:
            return joints
        
        # Base amplitude
        forward_speed = abs(vx)
        base_amp = min(forward_speed / 0.5, 1.0) * 0.3
        base_amp = max(base_amp, 0.1)
        
        # Trot pattern: FL+RR (phase), FR+RL (phase+0.5)
        leg_names = ["FL", "FR", "RL", "RR"]
        leg_phases_config = [
            (0, 0.0),   # FL
            (1, 0.5),   # FR
            (2, 0.5),   # RL
            (3, 0.0),   # RR
        ]
        
        for (leg_idx, phase_offset), leg_name in zip(leg_phases_config, leg_names):
            leg_phase = (phase + phase_offset) % 1.0
            is_front_leg = leg_name in ["FL", "FR"]
            is_support = self._foot_lock_support_start <= leg_phase <= self._foot_lock_support_end
            
            # Swing / stance logic:
            # - 平地：简单正弦摆腿；支撑期略微减小摆幅，表现 foot lock 感觉
            # - 楼梯：前腿在摆动期使用“抬-前-落”轨迹，支撑期不锁脚（避免卡住）
            if is_climbing_stairs and is_front_leg:
                # 楼梯模式：前腿
                if not is_support:
                    # 摆动期：抬-前-落
                    # 归一化摆动相位（假设 0~0.5 为 swing）
                    swing_phase = (leg_phase * 2.0) % 1.0
                    # 基础前后摆动
                    base_swing = np.sin(swing_phase * 2.0 * np.pi) * base_amp * self._stair_lift_multiplier
                    # 额外抬高度：在摆动中点最高
                    lift = np.sin(swing_phase * np.pi) * 0.3 * (1.0 + stair_height_factor)
                    swing = base_swing + lift
                else:
                    # 支撑期：不锁脚，只保持较小摆动，避免卡在台阶边缘
                    swing = np.sin(leg_phase * 2.0 * np.pi) * base_amp * 0.5
            else:
                # 普通模式：所有腿
                if is_support:
                    # 支撑期略微减小摆幅（视觉 foot lock）
                    swing = np.sin(leg_phase * 2.0 * np.pi) * base_amp * 0.7
                else:
                    swing = np.sin(leg_phase * 2.0 * np.pi) * base_amp
            
            thigh_idx = leg_idx + 4
            joints[thigh_idx] += swing
        
        return joints
    
    def step(self, cmd_vel: np.ndarray, obs: Optional["Observation"] = None) -> None:
        """Main control step."""
        # 0. Read sensors
        if self._foot_sensors:
            self._update_foot_heights()
        if self._front_sensor is not None:
            self._update_front_block()
        if self._left_sensor is not None or self._right_sensor is not None:
            self._update_side_blocks()

        # 1. Smooth velocity
        self._smoothed_cmd_vel = (
            self._vel_alpha * cmd_vel + 
            (1.0 - self._vel_alpha) * self._smoothed_cmd_vel
        )
        vx, vy, wz = self._smoothed_cmd_vel
        
        # Apply front speed scaling
        if vx > 0.0:
            vx *= self._front_speed_scale
        
        # Apply side blocking - CRITICAL: must block before calculating speed
        if vy > 0.0 and self._left_blocked:
            vy = 0.0
        elif vy < 0.0 and self._right_blocked:
            vy = 0.0
        
        speed = np.sqrt(vx**2 + vy**2)
        
        # 2. Update phase
        if speed > self._gait_start_threshold or abs(wz) > 0.01:
            speed_clamped = np.clip(speed, 0.0, self._max_gait_speed)
            gait_freq = self._base_freq + self._freq_gain * speed_clamped
            self._target_phase = (self._target_phase + self._dt * gait_freq) % 1.0
        else:
            self._target_phase = 0.0
        
        # Smooth phase transition
        phase_diff = self._target_phase - self._phase
        if abs(phase_diff) > 0.5:
            phase_diff = phase_diff - 1.0 if phase_diff > 0 else phase_diff + 1.0
        self._phase = (self._phase + phase_diff * self._phase_transition_alpha) % 1.0
        
        # 3. Get current pose
        base_pos, base_quat = self._get_base_pose()
        if self._last_pos is None:
            self._last_pos = base_pos.copy()
            self._current_yaw = self._quat_to_yaw(base_quat)
        
        # 4. Calculate movement
        vel_body = np.array([vx, vy, 0.0])
        vel_world = quat_rotate(base_quat, vel_body)
        forward_dir = vel_world.copy()
        if np.linalg.norm(forward_dir) > 0.001:
            forward_dir = forward_dir / np.linalg.norm(forward_dir)
        else:
            forward_dir = np.array([1.0, 0.0, 0.0])
        
        # 5. Move base
        is_blocked = speed > 0.01 and self._front_blocked and vx > 0.0
        
        if speed > 0.01 or abs(wz) > 0.01:
            if not is_blocked:
                new_pos = self._last_pos + vel_world * self._dt
                self._current_yaw += wz * self._dt
                new_quat = self._yaw_to_quat(self._current_yaw)
                self._last_pos = new_pos.copy()
            else:
                backoff_dist = 0.0
                if self._front_min_dist is not None and self._front_min_dist < self._front_backoff_dist:
                    backoff_dist = max(0.0, self._front_backoff_dist - self._front_min_dist)
                new_pos = base_pos - forward_dir * backoff_dist
                self._last_pos = new_pos.copy()
                new_quat = self._yaw_to_quat(self._current_yaw)
        else:
            new_pos = self._last_pos.copy()
            new_quat = self._yaw_to_quat(self._current_yaw)
        
        # 6. Terrain following
        z_fl = self._foot_heights.get("FL", 0.0)
        z_fr = self._foot_heights.get("FR", 0.0)
        z_rl = self._foot_heights.get("RL", 0.0)
        z_rr = self._foot_heights.get("RR", 0.0)
        
        z_front = 0.5 * (z_fl + z_fr)
        z_rear = 0.5 * (z_rl + z_rr)
        z_mean = 0.5 * (z_front + z_rear)
        
        front_rear_diff = max(0.0, z_front - z_rear)
        extra_climb = np.clip(front_rear_diff, 0.0, self._max_climb_height)
        target_height = z_mean + self._body_height + 0.5 * extra_climb
        
        # Pitch
        leg_span = abs(self._foot_offsets["FL"][0] - self._foot_offsets["RL"][0]) + 1e-6
        raw_pitch = np.arctan2(front_rear_diff, leg_span)
        target_pitch = 0.7 * raw_pitch
        
        # Roll
        z_left = 0.5 * (z_fl + z_rl)
        z_right = 0.5 * (z_fr + z_rr)
        left_right_diff = z_left - z_right
        foot_y = self._foot_offsets["FL"][1]
        robot_width = self._robot_width if self._robot_width is not None else (2.0 * abs(foot_y))
        raw_roll = np.arctan2(left_right_diff, robot_width + 1e-6)
        target_roll = self._roll_gain * raw_roll
        
        # Smooth stair level to reduce jitter in stair detection
        self._stairs_level += 0.2 * (front_rear_diff - self._stairs_level)
        is_climbing_stairs = self._stairs_level > self._foot_lock_stair_threshold
        stair_height_factor = (
            min(self._stairs_level / self._max_climb_height, 2.0) if is_climbing_stairs else 0.0
        )
        
        # Smooth transitions
        self._current_height += (target_height - self._current_height) * 0.1
        self._current_pitch += (target_pitch - self._current_pitch) * 0.1
        self._current_roll += (target_roll - self._current_roll) * 0.1
        new_pos[2] = self._current_height
        
        # 7. Animate legs
        joint_angles = self._animate_legs(self._phase, speed, vx, is_climbing_stairs, stair_height_factor)
        
        # 8. Apply qpos
        half_yaw = self._current_yaw * 0.5
        half_pitch = self._current_pitch * 0.5
        half_roll = self._current_roll * 0.5
        
        cy = np.cos(half_yaw)
        sy = np.sin(half_yaw)
        cp = np.cos(half_pitch)
        sp = np.sin(half_pitch)
        cr = np.cos(half_roll)
        sr = np.sin(half_roll)
        
        combined_quat = np.array([
            cr * cp * cy - sr * sp * sy,  # w
            sr * cp * cy + cr * sp * sy,  # x
            cr * sp * cy - sr * cp * sy,  # y
            cr * cp * sy + sr * sp * cy,  # z
        ], dtype=np.float32)
        
        full_qpos = np.concatenate([
            new_pos,      # 3: position
            combined_quat,  # 4: quaternion (wxyz)
            joint_angles,   # 12: joint angles
        ])
        self.robot.entity.set_qpos(full_qpos)
    
    def _quat_to_yaw(self, quat: np.ndarray) -> float:
        """Extract yaw from quaternion (wxyz)."""
        w, x, y, z = quat
        return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    
    def compute_action(self, cmd_vel: np.ndarray) -> np.ndarray:
        """Not used in kinematic mode."""
        return np.zeros(12, dtype=np.float32)


@LOCOMOTION_REGISTRY.register("kinematic_wheel_position")
class KinematicWheelPositionController(LocomotionControllerBase):
    """Pure kinematic Go2w controller using direct position updates."""

    CONTROLLER_TYPE = "kinematic_wheel_position"

    def __init__(self, cfg: DictConfig, robot: "RobotBase"):
        super().__init__(cfg, robot)
        self._dt = float(cfg.get("dt", 0.02))
        self._cmd_alpha = float(cfg.get("cmd_alpha", 0.25))
        self._max_linear_vel = float(cfg.get("max_linear_vel", 1.2))
        self._max_angular_vel = float(cfg.get("max_angular_vel", 1.5))
        self._wheel_radius = float(cfg.get("wheel_radius", 0.05))
        self._wheel_base = float(cfg.get("wheel_base", 0.4))
        self._track_width = float(cfg.get("track_width", 0.3))
        self._wheel_joints = list(
            cfg.get(
                "wheel_joints",
                ["FL_foot_joint", "FR_foot_joint", "RL_foot_joint", "RR_foot_joint"],
            )
        )
        self._fixed_height = cfg.get("fixed_height", None)

        self._smoothed_cmd = np.zeros(3, dtype=np.float32)
        self._base_pos = np.zeros(3, dtype=np.float32)
        self._base_yaw = 0.0
        self._joint_indices = None
        self._joint_targets = None
        self._wheel_target_indices = None
        self._wheel_target_angles = None

    @staticmethod
    def _quat_to_yaw(quat: np.ndarray) -> float:
        w, x, y, z = quat
        return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))

    @staticmethod
    def _yaw_to_quat(yaw: float) -> np.ndarray:
        half = 0.5 * yaw
        return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float32)

    def _ensure_joint_targets(self) -> None:
        if self._joint_indices is not None:
            return

        default_dof_pos = self.robot.cfg.get("default_dof_pos", {})
        indices = []
        targets = []
        wheel_idx_to_local = {}
        for name, pos in default_dof_pos.items():
            try:
                joint = self.robot.entity.get_joint(name)
                if joint is None:
                    continue
                dof_idx = int(joint.dofs_idx_local[0])
                indices.append(dof_idx)
                targets.append(float(pos))
                wheel_idx_to_local[name] = len(indices) - 1
            except Exception:
                continue

        self._joint_indices = np.asarray(indices, dtype=np.int32)
        self._joint_targets = np.asarray(targets, dtype=np.float32)

        wheel_target_indices = []
        wheel_target_angles = []
        for wheel_name in self._wheel_joints:
            local_idx = wheel_idx_to_local.get(wheel_name, None)
            if local_idx is None:
                continue
            wheel_target_indices.append(local_idx)
            wheel_target_angles.append(self._joint_targets[local_idx])
        self._wheel_target_indices = np.asarray(wheel_target_indices, dtype=np.int32)
        self._wheel_target_angles = np.asarray(wheel_target_angles, dtype=np.float32)

    def reset(self) -> None:
        self._smoothed_cmd.fill(0.0)
        pos = self.robot.entity.get_pos()
        quat = self.robot.entity.get_quat()
        if hasattr(pos, "cpu"):
            pos = pos.cpu().numpy()
        if hasattr(quat, "cpu"):
            quat = quat.cpu().numpy()
        if np.asarray(pos).ndim > 1:
            pos = pos[0]
        if np.asarray(quat).ndim > 1:
            quat = quat[0]
        self._base_pos = np.asarray(pos, dtype=np.float32).copy()
        if self._fixed_height is None:
            self._fixed_height = float(self._base_pos[2])
        self._base_pos[2] = float(self._fixed_height)
        self._base_yaw = self._quat_to_yaw(np.asarray(quat, dtype=np.float32))
        self._ensure_joint_targets()
        if self._joint_indices.size > 0:
            self.robot.entity.set_dofs_position(self._joint_targets, self._joint_indices)

    def compute_action(self, cmd_vel: np.ndarray) -> np.ndarray:
        return np.asarray(cmd_vel, dtype=np.float32)

    def step(self, cmd_vel: np.ndarray, obs: Optional["Observation"] = None) -> None:
        self._ensure_joint_targets()
        cmd = np.asarray(cmd_vel, dtype=np.float32).reshape(-1)[:3]
        if cmd.shape[0] < 3:
            cmd = np.pad(cmd, (0, 3 - cmd.shape[0]))
        self._smoothed_cmd = self._cmd_alpha * cmd + (1.0 - self._cmd_alpha) * self._smoothed_cmd

        vx = float(np.clip(self._smoothed_cmd[0], -self._max_linear_vel, self._max_linear_vel))
        vy = float(np.clip(self._smoothed_cmd[1], -self._max_linear_vel, self._max_linear_vel))
        wz = float(np.clip(self._smoothed_cmd[2], -self._max_angular_vel, self._max_angular_vel))

        c = float(np.cos(self._base_yaw))
        s = float(np.sin(self._base_yaw))
        self._base_pos[0] += (c * vx - s * vy) * self._dt
        self._base_pos[1] += (s * vx + c * vy) * self._dt
        self._base_pos[2] = float(self._fixed_height)
        self._base_yaw += wz * self._dt

        self.robot.entity.set_pos(self._base_pos)
        self.robot.entity.set_quat(self._yaw_to_quat(self._base_yaw))

        # Keep all joints in position control and integrate wheel angle targets.
        if self._wheel_target_indices.size > 0:
            L = self._wheel_base * 0.5
            W = self._track_width * 0.5
            R = max(self._wheel_radius, 1e-4)
            wheel_omega = np.array(
                [
                    (vx - vy - (L + W) * wz) / R,
                    (vx + vy + (L + W) * wz) / R,
                    (vx + vy - (L + W) * wz) / R,
                    (vx - vy + (L + W) * wz) / R,
                ],
                dtype=np.float32,
            )
            count = min(self._wheel_target_angles.shape[0], wheel_omega.shape[0])
            self._wheel_target_angles[:count] += wheel_omega[:count] * self._dt
            for i in range(count):
                local_idx = int(self._wheel_target_indices[i])
                self._joint_targets[local_idx] = float(self._wheel_target_angles[i])

        if self._joint_indices.size > 0:
            self.robot.entity.set_dofs_position(self._joint_targets, self._joint_indices)
