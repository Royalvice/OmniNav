"""
运动控制器抽象基类

将高层 cmd_vel 指令转换为关节级控制。
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
import numpy as np
from omegaconf import DictConfig

if TYPE_CHECKING:
    from omninav.robots.base import RobotBase


class LocomotionControllerBase(ABC):
    """
    运动控制器抽象基类。
    
    负责将高层速度指令 (cmd_vel) 转换为具体的关节控制信号，
    使机器人按照指定的线速度和角速度运动。
    """
    
    # 控制器类型标识 (用于注册)
    CONTROLLER_TYPE: str = ""
    
    def __init__(self, cfg: DictConfig, robot: "RobotBase"):
        """
        初始化运动控制器。
        
        Args:
            cfg: 控制器配置
            robot: 被控制的机器人实例
        """
        self.cfg = cfg
        self.robot = robot
    
    @abstractmethod
    def reset(self) -> None:
        """
        重置控制器状态。
        
        在仿真重置或开始新 episode 时调用。
        """
        pass
    
    @abstractmethod
    def compute_action(self, cmd_vel: np.ndarray) -> np.ndarray:
        """
        根据速度指令计算关节动作。
        
        Args:
            cmd_vel: [vx, vy, wz] 目标线速度 (m/s) 和角速度 (rad/s)
        
        Returns:
            关节位置/速度/力矩目标值 (取决于控制模式)
        """
        pass
    
    @abstractmethod
    def step(self, cmd_vel: np.ndarray) -> None:
        """
        执行一步运动控制。
        
        计算并应用关节控制到机器人。
        
        Args:
            cmd_vel: [vx, vy, wz] 目标线速度 (m/s) 和角速度 (rad/s)
        """
        pass
