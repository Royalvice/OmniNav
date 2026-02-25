"""Core map data contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List
import numpy as np


@dataclass(frozen=True)
class OccupancyMap2D:
    """Single-floor occupancy map (0 free, 1 occupied)."""

    floor_id: str
    grid: np.ndarray
    resolution: float
    origin_xy: np.ndarray
    extent_xy: np.ndarray
    obstacles: List[dict] = field(default_factory=list)


@dataclass(frozen=True)
class OccupancyMapSet:
    """Collection of occupancy maps for one scene."""

    maps_by_floor: Dict[str, OccupancyMap2D]
    default_floor: str
    version: int = 1


@dataclass(frozen=True)
class ConnectorNode:
    node_id: str
    floor_id: str
    xy: np.ndarray


@dataclass(frozen=True)
class ConnectorEdge:
    edge_id: str
    from_node_id: str
    to_node_id: str
    connector_type: str
    cost: float


@dataclass(frozen=True)
class ConnectorGraph:
    nodes: Dict[str, ConnectorNode]
    edges: List[ConnectorEdge]

