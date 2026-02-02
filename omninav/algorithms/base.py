"""
算法抽象基类

定义可插拔算法的接口规范。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np
from omegaconf import DictConfig


class AlgorithmBase(ABC):
    """
    可插拔算法抽象基类。
    
    所有导航、感知等算法必须继承此类并实现抽象方法。
    算法通过 ALGORITHM_REGISTRY 注册后可通过配置文件选择。
    """
    
    # 算法类型标识 (用于注册)
    ALGORITHM_TYPE: str = ""
    
    def __init__(self, cfg: DictConfig):
        """
        初始化算法。
        
        Args:
            cfg: 算法配置
        """
        self.cfg = cfg
    
    @abstractmethod
    def reset(self, task_info: Dict[str, Any]) -> None:
        """
        重置算法状态。
        
        在新任务开始时调用，传入任务信息以初始化算法。
        
        Args:
            task_info: 任务信息，如起点、终点、地图等
        """
        pass
    
    @abstractmethod
    def step(self, observation: Dict[str, Any]) -> np.ndarray:
        """
        根据观测计算动作。
        
        Args:
            observation: 传感器观测 + 机器人状态
                - 传感器数据 (如 "depth_camera", "lidar_2d")
                - "robot_state": RobotState 对象
                - "sim_time": 当前仿真时间
        
        Returns:
            cmd_vel: [vx, vy, wz] 速度指令
        """
        pass
    
    @property
    @abstractmethod
    def is_done(self) -> bool:
        """
        算法是否认为任务完成。
        
        Returns:
            True 如果算法判断任务已完成，否则 False
        """
        pass
    
    @property
    def info(self) -> Dict[str, Any]:
        """
        返回算法内部信息。
        
        用于调试、可视化或记录。默认返回空字典。
        
        Returns:
            包含算法内部状态的字典
        """
        return {}
