"""
Unitree Go2 Quadruped Robot Implementation

Go2 robot wrapper based on Genesis physics engine.

Design principles:
1. Use gs.morphs.URDF to load robot model
2. Control joints via robot.control_dofs_position()
3. Get state via robot.get_pos/quat()
4. Follow Genesis go2_env.py control patterns
5. Batch-First: all state arrays are (B, ...) shaped

References:
- examples/locomotion/go2_env.py
- examples/tutorials/control_your_robot.py
"""

from typing import Any, Dict, List, Optional, TYPE_CHECKING
from omegaconf import DictConfig
import numpy as np

from omninav.robots.base import RobotBase
from omninav.core.types import RobotState, JointInfo
from omninav.core.lifecycle import LifecycleState
from omninav.assets import resolve_urdf_path
from omninav.core.registry import ROBOT_REGISTRY

if TYPE_CHECKING:
    import genesis as gs


def _to_numpy(x) -> np.ndarray:
    """Convert Genesis tensor to numpy, ensuring Batch-First shape."""
    if hasattr(x, 'cpu'):
        arr = x.cpu().numpy()
    else:
        arr = np.asarray(x)
    # Ensure batch dimension: (3,) → (1, 3)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


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
        self._joint_info: Optional[JointInfo] = None
        
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
        self._require_state(LifecycleState.CREATED, "spawn")
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
        
        self._transition_to(LifecycleState.SPAWNED)
    
    def _init_joint_indices(self) -> None:
        """
        Initialize joint indices and default positions.
        
        Must be called after scene.build().
        """
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

        # Build JointInfo descriptor
        self._joint_info = JointInfo(
            names=tuple(self.JOINT_NAMES),
            dof_indices=self._motors_dof_idx.copy(),
            num_joints=n_joints,
            position_limits_lower=np.full(n_joints, -np.pi, dtype=np.float32),
            position_limits_upper=np.full(n_joints, np.pi, dtype=np.float32),
            velocity_limits=np.full(n_joints, 20.0, dtype=np.float32),
        )
    
    def get_state(self) -> RobotState:
        """
        Get current robot state (Batch-First).
        
        Returns:
            RobotState: TypedDict with all arrays shaped (B, ...)
        """
        self._require_state(LifecycleState.BUILT, "get_state")
        self._init_joint_indices()
        
        # Get base state
        position = self.entity.get_pos()
        orientation = self.entity.get_quat()
        linear_velocity = self.entity.get_vel()
        angular_velocity = self.entity.get_ang()
        
        # Get joint state
        joint_positions = self.entity.get_dofs_position(self._motors_dof_idx)
        joint_velocities = self.entity.get_dofs_velocity(self._motors_dof_idx)
        
        return RobotState(
            position=_to_numpy(position),
            orientation=_to_numpy(orientation),
            linear_velocity=_to_numpy(linear_velocity),
            angular_velocity=_to_numpy(angular_velocity),
            joint_positions=_to_numpy(joint_positions),
            joint_velocities=_to_numpy(joint_velocities),
        )
    
    def get_joint_info(self) -> JointInfo:
        """
        Get Go2 joint metadata.
        
        Returns:
            JointInfo with 12 motor joint names and indices
        """
        self._require_state(LifecycleState.BUILT, "get_joint_info")
        self._init_joint_indices()
        return self._joint_info
    
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

        # Transition lifecycle
        super().post_build()
    
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
        self._require_state(LifecycleState.BUILT, "reset")

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

        self._transition_to(LifecycleState.READY)
