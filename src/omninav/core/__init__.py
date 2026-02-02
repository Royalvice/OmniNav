"""
OmniNav Core Layer - 仿真核心管理

提供 Genesis 引擎的封装、场景管理、仿真循环控制。
"""

from omninav.core.base import SimulationManagerBase
from omninav.core.registry import (
    ROBOT_REGISTRY,
    SENSOR_REGISTRY,
    LOCOMOTION_REGISTRY,
    ALGORITHM_REGISTRY,
    TASK_REGISTRY,
    METRIC_REGISTRY,
    ASSET_LOADER_REGISTRY,
)

__all__ = [
    "SimulationManagerBase",
    "ROBOT_REGISTRY",
    "SENSOR_REGISTRY",
    "LOCOMOTION_REGISTRY",
    "ALGORITHM_REGISTRY",
    "TASK_REGISTRY",
    "METRIC_REGISTRY",
    "ASSET_LOADER_REGISTRY",
]
