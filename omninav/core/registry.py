"""
通用注册表 - 用于注册和发现各类模块

支持的注册表:
- ROBOT_REGISTRY: 机器人类型
- SENSOR_REGISTRY: 传感器类型
- LOCOMOTION_REGISTRY: 运动控制器
- ALGORITHM_REGISTRY: 算法
- TASK_REGISTRY: 评测任务
- METRIC_REGISTRY: 评价指标
- ASSET_LOADER_REGISTRY: 资产加载器
"""

from typing import Dict, Type, TypeVar, Callable, Optional

T = TypeVar("T")


class Registry:
    """
    通用注册表，用于注册和发现各类模块。
    
    使用示例:
        >>> @ROBOT_REGISTRY.register("my_robot")
        ... class MyRobot(RobotBase):
        ...     pass
        
        >>> robot_cls = ROBOT_REGISTRY.get("my_robot")
        >>> robot = ROBOT_REGISTRY.build("my_robot", cfg=cfg, scene=scene)
    """
    
    def __init__(self, name: str):
        """
        初始化注册表。
        
        Args:
            name: 注册表名称，用于错误信息
        """
        self.name = name
        self._registry: Dict[str, Type[T]] = {}
    
    def register(self, name: str) -> Callable[[Type[T]], Type[T]]:
        """
        装饰器：注册一个类。
        
        Args:
            name: 注册名称
            
        Returns:
            装饰器函数
            
        Raises:
            ValueError: 如果名称已被注册
        """
        def decorator(cls: Type[T]) -> Type[T]:
            if name in self._registry:
                raise ValueError(
                    f"'{name}' already registered in {self.name}. "
                    f"Existing: {self._registry[name]}, New: {cls}"
                )
            self._registry[name] = cls
            return cls
        return decorator
    
    def get(self, name: str) -> Type[T]:
        """
        根据名称获取已注册的类。
        
        Args:
            name: 注册名称
            
        Returns:
            已注册的类
            
        Raises:
            KeyError: 如果名称未注册
        """
        if name not in self._registry:
            raise KeyError(
                f"'{name}' not found in {self.name}. "
                f"Available: {list(self._registry.keys())}"
            )
        return self._registry[name]
    
    def build(self, name: str, **kwargs) -> T:
        """
        根据名称创建实例。
        
        Args:
            name: 注册名称
            **kwargs: 传递给构造函数的参数
            
        Returns:
            创建的实例
        """
        cls = self.get(name)
        return cls(**kwargs)
    
    def list(self) -> list:
        """列出所有已注册的名称。"""
        return list(self._registry.keys())
    
    def __contains__(self, name: str) -> bool:
        """检查名称是否已注册。"""
        return name in self._registry
    
    def __repr__(self) -> str:
        return f"Registry(name='{self.name}', items={list(self._registry.keys())})"


# 全局注册表实例
ROBOT_REGISTRY = Registry("robots")
SENSOR_REGISTRY = Registry("sensors")
LOCOMOTION_REGISTRY = Registry("locomotion")
ALGORITHM_REGISTRY = Registry("algorithms")
TASK_REGISTRY = Registry("tasks")
METRIC_REGISTRY = Registry("metrics")
ASSET_LOADER_REGISTRY = Registry("asset_loaders")
