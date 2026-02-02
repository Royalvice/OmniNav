"""
Unitree Go2w Wheeled Quadruped Robot Implementation

Go2w robot with Mecanum wheels for omnidirectional motion.
"""

from typing import Optional, TYPE_CHECKING
from omegaconf import DictConfig
import numpy as np

from omninav.robots.base import RobotBase, RobotState
from omninav.assets import resolve_urdf_path
from omninav.core.registry import ROBOT_REGISTRY

if TYPE_CHECKING:
    import genesis as gs


@ROBOT_REGISTRY.register("unitree_go2w")
class Go2wRobot(RobotBase):
    """
    Unitree Go2w Wheeled Quadruped Robot.

    Four-wheeled variant of Go2 with Mecanum wheels for omnidirectional motion.
    Uses velocity control for wheel joints.

    Configuration file: configs/robot/go2w.yaml

    Attributes:
        cfg: Robot configuration
        scene: Genesis scene object
        entity: Genesis entity object
    """

    ROBOT_TYPE: str = "unitree_go2w"

    # Wheel joint names (should match URDF)
    WHEEL_JOINT_NAMES = [
        "FL_wheel_joint",
        "FR_wheel_joint",
        "RL_wheel_joint",
        "RR_wheel_joint",
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

        # Wheel parameters
        self._wheel_radius = cfg.get("wheel", {}).get("radius", 0.05)
        self._wheel_base = cfg.get("wheel", {}).get("base", 0.4)
        self._track_width = cfg.get("wheel", {}).get("track", 0.3)

    def spawn(self) -> None:
        """
        Spawn Go2w robot in the scene.

        Loads URDF model and adds to Genesis scene.
        """
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
                self.entity.get_joint(name).dof_idx_local
                for name in self.WHEEL_JOINT_NAMES
            ],
            dtype=np.int32,
        )

    def get_state(self) -> RobotState:
        """
        Get current robot state.

        Returns:
            RobotState: Contains position, orientation, velocity, joint state
        """
        self._init_wheel_indices()

        # Get base state
        position = self.entity.get_pos()
        orientation = self.entity.get_quat()
        linear_velocity = self.entity.get_vel()
        angular_velocity = self.entity.get_ang()

        # Get wheel velocities
        joint_positions = self.entity.get_dofs_position(self._wheel_dof_idx)
        joint_velocities = self.entity.get_dofs_velocity(self._wheel_dof_idx)

        # Convert tensors to numpy
        def to_numpy(x):
            if hasattr(x, "cpu"):
                return x.cpu().numpy()
            return np.array(x)

        return RobotState(
            position=to_numpy(position).flatten(),
            orientation=to_numpy(orientation).flatten(),
            linear_velocity=to_numpy(linear_velocity).flatten(),
            angular_velocity=to_numpy(angular_velocity).flatten(),
            joint_positions=to_numpy(joint_positions).flatten(),
            joint_velocities=to_numpy(joint_velocities).flatten(),
        )

    def apply_command(self, cmd_vel: np.ndarray) -> None:
        """
        Apply velocity command via wheel control.

        Converts body velocity to wheel velocities using Mecanum kinematics.

        Args:
            cmd_vel: [vx, vy, wz] linear velocity (m/s) + angular velocity (rad/s)
        """
        self._init_wheel_indices()

        vx, vy, wz = cmd_vel[0], cmd_vel[1], cmd_vel[2]
        R = self._wheel_radius
        L = self._wheel_base / 2
        W = self._track_width / 2

        # Mecanum wheel inverse kinematics
        v_FL = (vx - vy - (L + W) * wz) / R
        v_FR = (vx + vy + (L + W) * wz) / R
        v_RL = (vx + vy - (L + W) * wz) / R
        v_RR = (vx - vy + (L + W) * wz) / R

        wheel_velocities = np.array([v_FL, v_FR, v_RL, v_RR], dtype=np.float32)

        # Apply wheel velocities
        self.entity.control_dofs_velocity(wheel_velocities, self._wheel_dof_idx)

    def apply_wheel_velocities(self, wheel_vels: np.ndarray) -> None:
        """
        Directly apply angular velocities to wheels.

        Args:
            wheel_vels: [FL, FR, RL, RR] wheel angular velocities (rad/s)
        """
        self._init_wheel_indices()
        self.entity.control_dofs_velocity(wheel_vels, self._wheel_dof_idx)

    def reset(self) -> None:
        """
        Reset robot to initial state.
        """
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
