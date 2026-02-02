"""
Component Registry for OmniNav

Provides a generic registration mechanism for pluggable components.
Components are registered by name and can be instantiated from configuration.
"""

from typing import Type, Dict, Any, Optional, Callable, TypeVar
from omegaconf import DictConfig

T = TypeVar("T")


class Registry:
    """
    Generic registry for pluggable components.
    
    Allows registration of classes by name and dynamic instantiation
    from OmegaConf configuration dictionaries.
    
    Example:
        >>> ROBOT_REGISTRY = Registry("robot")
        >>> @ROBOT_REGISTRY.register("unitree_go2")
        ... class Go2Robot(RobotBase):
        ...     pass
        >>> robot = ROBOT_REGISTRY.build(cfg, scene=scene)
    """
    
    def __init__(self, name: str):
        """
        Initialize registry.
        
        Args:
            name: Registry name (for error messages)
        """
        self._name = name
        self._module_dict: Dict[str, Type] = {}
    
    @property
    def name(self) -> str:
        """Get registry name."""
        return self._name
    
    @property
    def registered_names(self) -> list:
        """Get list of registered component names."""
        return list(self._module_dict.keys())
    
    def register(self, name: Optional[str] = None) -> Callable[[Type[T]], Type[T]]:
        """
        Decorator to register a class.
        
        Args:
            name: Registration name. If None, uses class name.
        
        Returns:
            Decorator function
        
        Example:
            >>> @ROBOT_REGISTRY.register("my_robot")
            ... class MyRobot(RobotBase):
            ...     pass
        """
        def decorator(cls: Type[T]) -> Type[T]:
            key = name if name is not None else cls.__name__
            if key in self._module_dict:
                raise KeyError(
                    f"'{key}' is already registered in {self._name} registry. "
                    f"Existing: {self._module_dict[key]}, New: {cls}"
                )
            self._module_dict[key] = cls
            return cls
        return decorator
    
    def get(self, name: str) -> Type:
        """
        Get a registered class by name.
        
        Args:
            name: Registered name
        
        Returns:
            Registered class
        
        Raises:
            KeyError: If name is not registered
        """
        if name not in self._module_dict:
            raise KeyError(
                f"'{name}' is not registered in {self._name} registry. "
                f"Available: {list(self._module_dict.keys())}"
            )
        return self._module_dict[name]
    
    def build(self, cfg: DictConfig, **kwargs) -> Any:
        """
        Instantiate a class from configuration.
        
        The configuration must contain a 'type' field that specifies
        the registered name of the class to instantiate.
        
        Args:
            cfg: Configuration with 'type' field
            **kwargs: Additional arguments passed to constructor
        
        Returns:
            Instantiated object
        
        Raises:
            KeyError: If 'type' not in config or not registered
        """
        if "type" not in cfg:
            raise KeyError(
                f"Config must contain 'type' field to build from {self._name} registry. "
                f"Got keys: {list(cfg.keys())}"
            )
        cls = self.get(cfg.type)
        return cls(cfg, **kwargs)
    
    def __contains__(self, name: str) -> bool:
        """Check if name is registered."""
        return name in self._module_dict
    
    def __repr__(self) -> str:
        return f"Registry(name={self._name}, items={list(self._module_dict.keys())})"


# =============================================================================
# Global Registry Instances
# =============================================================================

ROBOT_REGISTRY = Registry("robot")
"""Registry for robot implementations (Go2, Go2w, etc.)"""

SENSOR_REGISTRY = Registry("sensor")
"""Registry for sensor implementations (Lidar, Camera, etc.)"""

LOCOMOTION_REGISTRY = Registry("locomotion")
"""Registry for locomotion controllers (WheelController, IKController, etc.)"""

ALGORITHM_REGISTRY = Registry("algorithm")
"""Registry for navigation/perception algorithms"""

TASK_REGISTRY = Registry("task")
"""Registry for evaluation tasks"""

METRIC_REGISTRY = Registry("metric")
"""Registry for evaluation metrics"""
