"""
Unitree Go2 Quadruped Robot Implementation

Go2 robot wrapper based on Genesis physics engine.

Design principles:
1. Use gs.morphs.URDF to load robot model
2. Control joints via robot.control_dofs_position()
3. Get state via robot.get_pos/quat()
4. Follow Genesis go2_env.py control patterns

References:
- examples/locomotion/go2_env.py
- examples/tutorials/control_your_robot.py
"""

from typing import Any, Dict, List, Optional, TYPE_CHECKING
from omegaconf import DictConfig
import numpy as np

from omninav.robots.base import RobotBase, RobotState
from omninav.assets import resolve_urdf_path
from omninav.core.registry import ROBOT_REGISTRY

if TYPE_CHECKING:
    import genesis as gs


@ROBOT_REGISTRY.register("unitree_go2")
class Go2Robot(RobotBase):
    """
    Unitree Go2 Quadruped Robot.
    
    Genesis-based Go2 robot class with support for:
    - URDF model loading (Genesis built-in or project custom)
    - Joint position/velocity control
    - State reading (position, orientation, velocity, joint state)
    - Sensor mounting
    
    Configuration file: configs/robot/go2.yaml
    
    Attributes:
        cfg: Robot configuration
        scene: Genesis scene object
        entity: Genesis entity object
    """
    
    ROBOT_TYPE: str = "unitree_go2"
    
    # Go2's 12 joint names (following Genesis go2_env.py)
    JOINT_NAMES = [
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",  # Front Left
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",  # Front Right
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",  # Rear Left
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",  # Rear Right
    ]
    
    # Default standing joint angles (reference: go2_env.py)
    DEFAULT_JOINT_ANGLES = {
        "FL_hip_joint": 0.0,
        "FL_thigh_joint": 0.8,
        "FL_calf_joint": -1.5,
        "FR_hip_joint": 0.0,
        "FR_thigh_joint": 0.8,
        "FR_calf_joint": -1.5,
        "RL_hip_joint": 0.0,
        "RL_thigh_joint": 1.0,
        "RL_calf_joint": -1.5,
        "RR_hip_joint": 0.0,
        "RR_thigh_joint": 1.0,
        "RR_calf_joint": -1.5,
    }
    
    def __init__(self, cfg: DictConfig, scene: "gs.Scene"):
        """
        Initialize Go2 robot.
        
        Args:
            cfg: Robot configuration (from configs/robot/go2.yaml)
            scene: Genesis scene object
        """
        super().__init__(cfg, scene)
        
        self._motors_dof_idx: Optional[np.ndarray] = None
        self._default_dof_pos: Optional[np.ndarray] = None
        
        # PD control parameters
        self._kp = cfg.get("control", {}).get("kp", 20.0)
        self._kd = cfg.get("control", {}).get("kd", 0.5)
        
        # Velocity limits
        self._max_linear_vel = cfg.get("control", {}).get("max_linear_vel", 1.0)
        self._max_angular_vel = cfg.get("control", {}).get("max_angular_vel", 2.0)
    
    def spawn(self) -> None:
        """
        Spawn Go2 robot in the scene.
        
        Uses gs.morphs.URDF to load model, path resolved by path_resolver.
        """
        import genesis as gs
        
        # Resolve URDF path (transparently handles genesis_builtin vs project)
        urdf_path = resolve_urdf_path(self.cfg)
        
        # Initial pose
        initial_pos = tuple(self.cfg.get("initial_pos", [0.0, 0.0, 0.4]))
        initial_quat = tuple(self.cfg.get("initial_quat", [1.0, 0.0, 0.0, 0.0]))
        
        # Save initial state
        self._initial_pos = np.array(initial_pos)
        self._initial_quat = np.array(initial_quat)
        
        # Gravity compensation (1.0 = fully compensated = kinematic/animation-style control)
        gravity_compensation = float(self.cfg.get("gravity_compensation", 0.0))
        
        # Add robot entity with optional gravity compensation for kinematic control
        if gravity_compensation > 0:
            self.entity = self.scene.add_entity(
                gs.morphs.URDF(
                    file=urdf_path,
                    pos=initial_pos,
                    quat=initial_quat,
                ),
                material=gs.materials.Rigid(gravity_compensation=gravity_compensation),
            )
        else:
            self.entity = self.scene.add_entity(
                gs.morphs.URDF(
                    file=urdf_path,
                    pos=initial_pos,
                    quat=initial_quat,
                )
            )
        
        # Note: Joint indices can only be obtained after scene.build()
        # Here we just record joint names for deferred initialization
    
    def _init_joint_indices(self) -> None:
        """
        Initialize joint indices and default positions.
        
        Must be called after scene.build().
        """
        import genesis as gs
        
        if self._motors_dof_idx is not None:
            return  # Already initialized
        
        # Get joint DOF indices (reference: go2_env.py)
        # Note: Go2 URDF has a floating base (6 DoFs) + 12 motor joints
        self._motors_dof_idx = np.array([
            self.entity.get_joint(name).dof_start 
            for name in self.JOINT_NAMES
        ], dtype=np.int32)
        
        # Default joint positions (indexed by JOINT_NAMES order for control)
        self._default_dof_pos = np.array([
            self.DEFAULT_JOINT_ANGLES[name] for name in self.JOINT_NAMES
        ], dtype=np.float32)
        
        # Build full initial qpos: [pos(3), quat(4), joints(12)] = 19 elements
        # IMPORTANT: qpos joint order must match URDF order (robot.joints[1:]),
        # NOT our JOINT_NAMES order! Reference: go2_env.py line 97-101
        init_dof_pos_urdf_order = np.array([
            self.DEFAULT_JOINT_ANGLES[joint.name] 
            for joint in self.entity.joints[1:]  # Skip floating base (joints[0])
        ], dtype=np.float32)
        
        # Store URDF-ordered default positions for set_qpos
        self._default_dof_pos_urdf = init_dof_pos_urdf_order.copy()
        
        self._init_qpos = np.concatenate([
            self._initial_pos,         # 3: base position
            self._initial_quat,        # 4: base quaternion (wxyz)
            init_dof_pos_urdf_order    # 12: joints in URDF order
        ]).astype(np.float32)
        
        # Set PD control gains (higher values for quadruped stability)
        n_joints = len(self.JOINT_NAMES)
        self.entity.set_dofs_kp(
            [self._kp] * n_joints, 
            self._motors_dof_idx
        )
        self.entity.set_dofs_kv(
            [self._kd] * n_joints, 
            self._motors_dof_idx
        )
    
    def get_state(self) -> RobotState:
        """
        Get current robot state.
        
        Returns:
            RobotState: Contains position, orientation, velocity, joint state
        """
        self._init_joint_indices()
        
        # Get base state
        position = self.entity.get_pos()  # [n_envs, 3] or [3]
        orientation = self.entity.get_quat()  # [n_envs, 4] or [4]
        linear_velocity = self.entity.get_vel()
        angular_velocity = self.entity.get_ang()
        
        # Get joint state
        joint_positions = self.entity.get_dofs_position(self._motors_dof_idx)
        joint_velocities = self.entity.get_dofs_velocity(self._motors_dof_idx)
        
        # Convert to numpy (Genesis may return torch tensor)
        def to_numpy(x):
            if hasattr(x, 'cpu'):
                return x.cpu().numpy()
            return np.array(x)
        
        return RobotState(
            position=to_numpy(position),
            orientation=to_numpy(orientation),
            linear_velocity=to_numpy(linear_velocity),
            angular_velocity=to_numpy(angular_velocity),
            joint_positions=to_numpy(joint_positions),
            joint_velocities=to_numpy(joint_velocities),
        )
    
    def apply_command(self, cmd_vel: np.ndarray) -> None:
        """
        Apply velocity command (high-level interface).
        
        Note: This is a simplified kinematic interface. Actual quadruped control
        requires conversion to joint targets via LocomotionController.
        
        For simple navigation testing, can directly set base velocity (kinematic mode only).
        
        Args:
            cmd_vel: [vx, vy, wz] linear velocity (m/s) + angular velocity (rad/s)
        """
        # Clamp velocity
        cmd_vel = np.array(cmd_vel)
        cmd_vel[:2] = np.clip(cmd_vel[:2], -self._max_linear_vel, self._max_linear_vel)
        cmd_vel[2] = np.clip(cmd_vel[2], -self._max_angular_vel, self._max_angular_vel)
        
        # TODO: Actual implementation needs LocomotionController conversion
        # This is just an interface placeholder
        pass
    
    def control_joints_position(self, target_positions: np.ndarray) -> None:
        """
        Direct joint position control (low-level interface).
        
        Called by LocomotionController.
        
        Args:
            target_positions: Target positions for 12 joints (rad)
        """
        self._init_joint_indices()
        self.entity.control_dofs_position(target_positions, self._motors_dof_idx)
    
    def control_joints_velocity(self, target_velocities: np.ndarray) -> None:
        """
        Direct joint velocity control (low-level interface).
        
        Args:
            target_velocities: Target velocities for 12 joints (rad/s)
        """
        self._init_joint_indices()
        self.entity.control_dofs_velocity(target_velocities, self._motors_dof_idx)
    
    def post_build(self) -> None:
        """
        Called after scene.build() to initialize joints.
        
        Sets up:
        - Joint DOF indices
        - PD control gains  
        - Initial robot state (full qpos including base position/orientation)
        - Initial PD control target for standing
        """
        self._init_joint_indices()
        
        # Set full initial state using set_qpos (position + quaternion + joints)
        # This properly initializes both base pose and joint positions
        # zero_velocity=True clears all velocities
        self.entity.set_qpos(self._init_qpos, zero_velocity=True)
        
        # Set PD control target to maintain standing pose
        # This tells the PD controller what position to hold
        self.entity.control_dofs_position(self._default_dof_pos, self._motors_dof_idx)
    
    @property
    def motors_dof_idx(self) -> np.ndarray:
        """Get motor DOF indices (available after post_build)."""
        return self._motors_dof_idx
    
    @property
    def default_dof_pos(self) -> np.ndarray:
        """Get default joint positions (in JOINT_NAMES order, for control)."""
        return self._default_dof_pos
    
    @property
    def default_dof_pos_urdf(self) -> np.ndarray:
        """Get default joint positions in URDF order (for set_qpos)."""
        return self._default_dof_pos_urdf
    
    def reset(self) -> None:
        """
        Reset robot to initial state.
        """
        if self.entity is None:
            return
        
        # Set initial pose and zero velocity
        if self._initial_pos is not None:
            self.entity.set_pos(self._initial_pos)
        if self._initial_quat is not None:
            self.entity.set_quat(self._initial_quat)
        
        # Reset joints to default positions
        if self._default_dof_pos is not None:
            self.entity.set_dofs_position(
                self._default_dof_pos, 
                self._motors_dof_idx
            )



