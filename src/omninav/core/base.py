"""
仿真管理器抽象基类

定义仿真核心的接口规范。
"""

from abc import ABC, abstractmethod
from typing import Optional, Any, TYPE_CHECKING
from omegaconf import DictConfig

if TYPE_CHECKING:
    from omninav.robots.base import RobotBase


class SimulationManagerBase(ABC):
    """
    仿真管理器抽象基类。
    
    负责:
    - Genesis 引擎初始化
    - 场景管理
    - 仿真循环控制
    - 机器人和资产的添加
    """
    
    @abstractmethod
    def initialize(self, cfg: DictConfig) -> None:
        """
        初始化仿真环境。
        
        Args:
            cfg: 仿真配置 (来自 Hydra)
        """
        pass
    
    @abstractmethod
    def build(self) -> None:
        """
        构建场景。
        
        在所有实体添加完成后调用，触发 Genesis scene.build()。
        """
        pass
    
    @abstractmethod
    def step(self) -> None:
        """
        推进一步物理仿真。
        
        调用 Genesis scene.step()，执行一个仿真步长。
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """
        重置仿真状态。
        
        将所有实体恢复到初始状态。
        """
        pass
    
    @abstractmethod
    def get_sim_time(self) -> float:
        """
        获取当前仿真时间。
        
        Returns:
            仿真时间 (秒)
        """
        pass
    
    @abstractmethod
    def add_robot(self, robot: "RobotBase") -> None:
        """
        添加机器人到场景。
        
        Args:
            robot: 机器人实例
        """
        pass
    
    @abstractmethod
    def load_scene(self, scene_cfg: DictConfig) -> None:
        """
        加载场景资产。
        
        Args:
            scene_cfg: 场景配置
        """
        pass
    
    @property
    @abstractmethod
    def scene(self) -> Any:
        """
        获取 Genesis 场景对象。
        
        Returns:
            Genesis scene 对象
        """
        pass
