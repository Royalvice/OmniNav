"""
评测任务与指标抽象基类

定义任务和评价指标的接口规范。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List
import numpy as np
from omegaconf import DictConfig


@dataclass
class TaskResult:
    """
    任务执行结果。
    
    Attributes:
        success: 任务是否成功
        metrics: 各项评价指标值
        info: 额外信息 (如轨迹、步数等)
    """
    success: bool
    metrics: Dict[str, float] = field(default_factory=dict)
    info: Dict[str, Any] = field(default_factory=dict)


class MetricBase(ABC):
    """
    评价指标抽象基类。
    
    所有评价指标必须继承此类并实现抽象方法。
    """
    
    # 指标名称 (用于注册和结果字典键)
    METRIC_NAME: str = ""
    
    def __init__(self):
        """初始化指标。"""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """
        重置指标状态。
        
        在新任务开始时调用。
        """
        pass
    
    @abstractmethod
    def update(self, **kwargs) -> None:
        """
        更新指标。
        
        在每个仿真步后调用，传入相关数据。
        
        Args:
            **kwargs: 更新所需的数据，如 robot_position, goal_position 等
        """
        pass
    
    @abstractmethod
    def compute(self) -> float:
        """
        计算最终指标值。
        
        在任务结束时调用。
        
        Returns:
            指标值
        """
        pass


class TaskBase(ABC):
    """
    评测任务抽象基类。
    
    所有评测任务必须继承此类并实现抽象方法。
    任务定义了评测的目标、终止条件和评价指标。
    """
    
    # 任务类型标识 (用于注册)
    TASK_TYPE: str = ""
    
    def __init__(self, cfg: DictConfig):
        """
        初始化任务。
        
        Args:
            cfg: 任务配置
        """
        self.cfg = cfg
        self.metrics: List[MetricBase] = []
    
    @abstractmethod
    def reset(self) -> Dict[str, Any]:
        """
        重置任务状态。
        
        Returns:
            task_info: 任务信息，传递给算法用于初始化
                - 应包含 "start_position", "goal_position" 等
        """
        pass
    
    @abstractmethod
    def step(self, robot_state: Any, action: np.ndarray) -> None:
        """
        记录每步信息。
        
        用于计算累积指标 (如路径长度、碰撞次数等)。
        
        Args:
            robot_state: RobotState 对象
            action: 当前步的动作
        """
        pass
    
    @abstractmethod
    def is_terminated(self, robot_state: Any) -> bool:
        """
        判断任务是否终止。
        
        可能因为成功、失败或超时而终止。
        
        Args:
            robot_state: RobotState 对象
            
        Returns:
            True 如果任务应该终止，否则 False
        """
        pass
    
    @abstractmethod
    def compute_result(self) -> TaskResult:
        """
        计算最终任务结果。
        
        综合所有指标计算最终结果。
        
        Returns:
            TaskResult: 包含成功标志、各项指标和额外信息
        """
        pass
