"""
Unitree Go2w Wheeled Quadruped Robot Implementation

Go2w robot with Mecanum wheels for omnidirectional motion.

Design principles:
1. Batch-First: all state arrays are (B, ...) shaped
2. Lifecycle enforced via LifecycleMixin
3. No apply_command — wheel control is Locomotion layer's job
"""

from typing import Optional, TYPE_CHECKING
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
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


@ROBOT_REGISTRY.register("unitree_go2w")
class Go2wRobot(RobotBase):
    """
    Unitree Go2w Wheeled Quadruped Robot.

    Four-wheeled variant of Go2 with Mecanum wheels for omnidirectional motion.
    Uses velocity control for wheel joints.

    Configuration file: configs/robot/go2w.yaml
    """

    ROBOT_TYPE: str = "unitree_go2w"

    # Wheel joint names (should match URDF)
    WHEEL_JOINT_NAMES = [
        "FL_foot_joint",
        "FR_foot_joint",
        "RL_foot_joint",
        "RR_foot_joint",
    ]

    def __init__(self, cfg: DictConfig, scene: "gs.Scene"):
        """
        Initialize Go2w robot.

        Args:
            cfg: Robot configuration (from configs/robot/go2w.yaml)
            scene: Genesis scene object
        """
        super().__init__(cfg, scene)

        self._wheel_dof_idx: Optional[np.ndarray] = None
        self._joint_info: Optional[JointInfo] = None

        # Wheel parameters
        self._wheel_radius = cfg.get("wheel", {}).get("radius", 0.05)
        self._wheel_base = cfg.get("wheel", {}).get("base", 0.4)
        self._track_width = cfg.get("wheel", {}).get("track", 0.3)

    def spawn(self) -> None:
        """
        Spawn Go2w robot in the scene.

        Loads URDF model and adds to Genesis scene.
        """
        self._require_state(LifecycleState.CREATED, "spawn")
        import genesis as gs

        # Resolve URDF path
        urdf_path = resolve_urdf_path(self.cfg)

        # Initial pose
        initial_pos = tuple(self.cfg.get("initial_pos", [0.0, 0.0, 0.15]))
        initial_quat = tuple(self.cfg.get("initial_quat", [1.0, 0.0, 0.0, 0.0]))

        # Save initial state for reset
        self._initial_pos = np.array(initial_pos)
        self._initial_quat = np.array(initial_quat)

        # Add robot entity to scene
        self.entity = self.scene.add_entity(
            gs.morphs.URDF(
                file=urdf_path,
                pos=initial_pos,
                quat=initial_quat,
            )
        )

        self._transition_to(LifecycleState.SPAWNED)

    def _init_wheel_indices(self) -> None:
        """
        Initialize wheel DOF indices.

        Must be called after scene.build().
        """
        if self._wheel_dof_idx is not None:
            return  # Already initialized

        # Get wheel joint DOF indices
        self._wheel_dof_idx = np.array(
            [
                self.entity.get_joint(name).dofs_idx_local[0]
                for name in self.WHEEL_JOINT_NAMES
            ],
            dtype=np.int32,
        )

        # Build JointInfo for wheel joints
        self._joint_info = JointInfo(
            names=tuple(self.WHEEL_JOINT_NAMES),
            dof_indices=self._wheel_dof_idx.copy(),
            num_joints=len(self.WHEEL_JOINT_NAMES),
            position_limits_lower=np.full(4, -np.inf, dtype=np.float32),
            position_limits_upper=np.full(4, np.inf, dtype=np.float32),
            velocity_limits=np.full(4, 20.0, dtype=np.float32),
        )

    def get_state(self) -> RobotState:
        """
        Get current robot state (Batch-First).

        Returns:
            RobotState: TypedDict with all arrays shaped (B, ...)
        """
        self._require_state(LifecycleState.BUILT, "get_state")
        self._init_wheel_indices()

        # Get base state
        position = self.entity.get_pos()
        orientation = self.entity.get_quat()
        linear_velocity = self.entity.get_vel()
        angular_velocity = self.entity.get_ang()

        # Get wheel velocities
        joint_positions = self.entity.get_dofs_position(self._wheel_dof_idx)
        joint_velocities = self.entity.get_dofs_velocity(self._wheel_dof_idx)

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
        Get Go2w joint metadata (wheel joints).

        Returns:
            JointInfo with 4 wheel joint names and indices
        """
        self._require_state(LifecycleState.BUILT, "get_joint_info")
        self._init_wheel_indices()
        return self._joint_info

    def post_build(self) -> None:
        """
        Called after scene.build() to initialize wheel indices.
        """
        self._init_wheel_indices()
        super().post_build()

    def reset(self) -> None:
        """
        Reset robot to initial state.
        """
        self._require_state(LifecycleState.BUILT, "reset")

        if self.entity is None:
            return

        if self._initial_pos is not None:
            self.entity.set_pos(self._initial_pos)
        if self._initial_quat is not None:
            self.entity.set_quat(self._initial_quat)

        # Zero wheel velocities
        if self._wheel_dof_idx is not None:
            self.entity.control_dofs_velocity(
                np.zeros(4, dtype=np.float32), self._wheel_dof_idx
            )
            
        # Reset joint positions
        default_dof_pos = self.cfg.get("default_dof_pos", None)
        if default_dof_pos:
             try:
                dof_indices = []
                dof_pos = []
                for name, pos in default_dof_pos.items():
                    joint = self.entity.get_joint(name)
                    if joint:
                        dof_indices.append(joint.dofs_idx_local[0])
                        dof_pos.append(pos)
                if dof_indices:
                    self.entity.set_dofs_position(
                        np.array(dof_pos, dtype=np.float32), 
                        np.array(dof_indices, dtype=np.int32)
                    )
             except Exception as e:
                 print(f"Warning: Failed to reset joint positions: {e}")

        self._transition_to(LifecycleState.READY)
