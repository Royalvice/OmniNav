"""
Generic Registry - For registering and discovering various modules

Supported registries:
- ROBOT_REGISTRY: Robot types
- SENSOR_REGISTRY: Sensor types
- LOCOMOTION_REGISTRY: Locomotion controllers
- ALGORITHM_REGISTRY: Algorithms
- TASK_REGISTRY: Evaluation tasks
- METRIC_REGISTRY: Evaluation metrics
- ASSET_LOADER_REGISTRY: Asset loaders
"""

from typing import Dict, Type, TypeVar, Callable, Optional

T = TypeVar("T")


class Registry:
    """
    Generic registry for registering and discovering various modules.
    
    Usage example:
        >>> @ROBOT_REGISTRY.register("my_robot")
        ... class MyRobot(RobotBase):
        ...     pass
        
        >>> robot_cls = ROBOT_REGISTRY.get("my_robot")
        >>> robot = ROBOT_REGISTRY.build("my_robot", cfg=cfg, scene=scene)
    """
    
    def __init__(self, name: str):
        """
        Initialize registry.
        
        Args:
            name: Registry name, used in error messages
        """
        self.name = name
        self._registry: Dict[str, Type[T]] = {}
    
    def register(self, name: str) -> Callable[[Type[T]], Type[T]]:
        """
        Decorator: Register a class.
        
        Args:
            name: Registration name
            
        Returns:
            Decorator function
            
        Raises:
            ValueError: If name is already registered
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
        Get registered class by name.
        
        Args:
            name: Registration name
            
        Returns:
            Registered class
            
        Raises:
            KeyError: If name is not registered
        """
        if name not in self._registry:
            raise KeyError(
                f"'{name}' not found in {self.name}. "
                f"Available: {list(self._registry.keys())}"
            )
        return self._registry[name]
    
    def build(self, name: str, **kwargs) -> T:
        """
        Create instance by name.
        
        Args:
            name: Registration name
            **kwargs: Arguments passed to constructor
            
        Returns:
            Created instance
        """
        cls = self.get(name)
        return cls(**kwargs)
    
    def list(self) -> list:
        """List all registered names."""
        return list(self._registry.keys())
    
    def __contains__(self, name: str) -> bool:
        """Check if name is registered."""
        return name in self._registry
    
    def __repr__(self) -> str:
        return f"Registry(name='{self.name}', items={list(self._registry.keys())})"


# Global registry instances
ROBOT_REGISTRY = Registry("robots")
SENSOR_REGISTRY = Registry("sensors")
LOCOMOTION_REGISTRY = Registry("locomotion")
ALGORITHM_REGISTRY = Registry("algorithms")
TASK_REGISTRY = Registry("tasks")
METRIC_REGISTRY = Registry("metrics")
ASSET_LOADER_REGISTRY = Registry("asset_loaders")
