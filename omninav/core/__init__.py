"""
OmniNav Core Layer - Simulation Core

Provides simulation manager, component registry, data types,
event/hook system, and lifecycle management.
"""

from omninav.core.base import SimulationManagerBase
from omninav.core.simulation_manager import GenesisSimulationManager
from omninav.core.registry import Registry, BuildContext
from omninav.core.types import (
    RobotState,
    SensorData,
    Observation,
    Action,
    JointInfo,
    MountInfo,
    TaskResult,
)
from omninav.core.hooks import EventType, HookManager
from omninav.core.lifecycle import LifecycleState, LifecycleMixin

__all__ = [
    # Simulation
    "SimulationManagerBase",
    "GenesisSimulationManager",
    # Registry
    "Registry",
    "BuildContext",
    # Data types
    "RobotState",
    "SensorData",
    "Observation",
    "Action",
    "JointInfo",
    "MountInfo",
    "TaskResult",
    # Hooks
    "EventType",
    "HookManager",
    # Lifecycle
    "LifecycleState",
    "LifecycleMixin",
]
