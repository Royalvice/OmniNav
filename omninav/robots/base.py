"""
机器人和传感器抽象基类

定义机器人和传感器的接口规范。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, TYPE_CHECKING
import numpy as np
from omegaconf import DictConfig

if TYPE_CHECKING:
    import genesis as gs


@dataclass
class RobotState:
    """
    机器人状态数据类。
    
    Attributes:
        position: [x, y, z] 世界坐标位置
        orientation: [qw, qx, qy, qz] 四元数姿态
        linear_velocity: [vx, vy, vz] 线速度
        angular_velocity: [wx, wy, wz] 角速度
        joint_positions: 关节位置数组
        joint_velocities: 关节速度数组
    """
    position: np.ndarray
    orientation: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    joint_positions: np.ndarray
    joint_velocities: np.ndarray


@dataclass
class SensorMount:
    """
    传感器挂载配置。
    
    Attributes:
        sensor_name: 传感器配置名 (对应 configs/sensor/*.yaml 中定义的传感器)
        link_name: 挂载的 link 名称
        position: 相对于 link 的位置 [x, y, z]
        orientation: 相对于 link 的姿态 [qw, qx, qy, qz]
    """
    sensor_name: str
    link_name: str
    position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    orientation: List[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])


class SensorBase(ABC):
    """
    传感器抽象基类。
    
    所有传感器实现必须继承此类并实现抽象方法。
    """
    
    # 传感器类型标识 (用于注册)
    SENSOR_TYPE: str = ""
    
    def __init__(self, cfg: DictConfig):
        """
        初始化传感器。
        
        Args:
            cfg: 传感器配置
        """
        self.cfg = cfg
        self._attached_robot: Optional["RobotBase"] = None
        self._gs_sensor: Optional[Any] = None  # Genesis 传感器对象
    
    @abstractmethod
    def create(self, scene: "gs.Scene") -> None:
        """
        在场景中创建传感器。
        
        根据传感器类型调用相应的 Genesis API (如 scene.add_camera)。
        
        Args:
            scene: Genesis 场景对象
        """
        pass
    
    @abstractmethod
    def get_data(self) -> Any:
        """
        获取传感器数据。
        
        Returns:
            传感器数据 (类型取决于传感器类型)
        """
        pass
    
    def attach_to_robot(
        self, 
        robot: "RobotBase", 
        link_name: str,
        position: List[float],
        orientation: List[float]
    ) -> None:
        """
        将传感器挂载到机器人的指定 link。
        
        Args:
            robot: 机器人实例
            link_name: 挂载的 link 名称
            position: 相对位置 [x, y, z]
            orientation: 相对姿态 [qw, qx, qy, qz]
        """
        self._attached_robot = robot
        # 具体实现中需要调用 Genesis API 设置传感器的 parent link


class RobotBase(ABC):
    """
    机器人抽象基类。
    
    所有机器人实现必须继承此类并实现抽象方法。
    """
    
    # 机器人类型标识 (用于注册)
    ROBOT_TYPE: str = ""
    
    def __init__(self, cfg: DictConfig, scene: "gs.Scene"):
        """
        初始化机器人。
        
        Args:
            cfg: 机器人配置
            scene: Genesis 场景对象
        """
        self.cfg = cfg
        self.scene = scene
        self.entity: Optional[Any] = None  # Genesis entity 对象
        self.sensors: Dict[str, SensorBase] = {}
        self._sensor_mounts: List[SensorMount] = []
        self._initial_pos: Optional[np.ndarray] = None
        self._initial_quat: Optional[np.ndarray] = None
    
    @abstractmethod
    def spawn(self) -> None:
        """
        在场景中生成机器人。
        
        加载机器人模型并添加到 Genesis 场景。
        """
        pass
    
    @abstractmethod
    def get_state(self) -> RobotState:
        """
        获取机器人当前状态。
        
        Returns:
            RobotState: 包含位置、姿态、速度、关节状态的数据
        """
        pass
    
    @abstractmethod
    def apply_command(self, cmd_vel: np.ndarray) -> None:
        """
        应用速度指令。
        
        高层接口，由 LocomotionController 调用以控制机器人运动。
        
        Args:
            cmd_vel: [vx, vy, wz] 线速度 (m/s) + 角速度 (rad/s)
        """
        pass
    
    def mount_sensor(self, mount: SensorMount, sensor: SensorBase) -> None:
        """
        挂载传感器到机器人。
        
        Args:
            mount: 挂载配置
            sensor: 传感器实例
        """
        sensor.attach_to_robot(
            self, 
            mount.link_name, 
            mount.position, 
            mount.orientation
        )
        self.sensors[mount.sensor_name] = sensor
        self._sensor_mounts.append(mount)
    
    def get_observations(self) -> Dict[str, Any]:
        """
        获取所有传感器观测。
        
        Returns:
            字典，键为传感器名称，值为传感器数据
        """
        obs = {}
        for name, sensor in self.sensors.items():
            obs[name] = sensor.get_data()
        return obs
    
    def reset(self) -> None:
        """
        重置机器人到初始状态。
        """
        if self.entity is not None and self._initial_pos is not None:
            self.entity.set_pos(self._initial_pos)
        if self.entity is not None and self._initial_quat is not None:
            self.entity.set_quat(self._initial_quat)
