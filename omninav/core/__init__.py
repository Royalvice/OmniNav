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
from omninav.core.map import (
    OccupancyMap2D,
    OccupancyMapSet,
    ConnectorNode,
    ConnectorEdge,
    ConnectorGraph,
    MapService,
    build_map_service_from_scene_cfg,
)

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
    # Map
    "OccupancyMap2D",
    "OccupancyMapSet",
    "ConnectorNode",
    "ConnectorEdge",
    "ConnectorGraph",
    "MapService",
    "build_map_service_from_scene_cfg",
]
