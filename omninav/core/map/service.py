"""Runtime map service."""

from __future__ import annotations

from typing import Optional
import numpy as np

from omninav.core.map.types import OccupancyMap2D, OccupancyMapSet, ConnectorGraph


class MapService:
    """Read-only query service for occupancy maps and connectors."""

    def __init__(self, map_set: OccupancyMapSet, connectors: ConnectorGraph):
        self._map_set = map_set
        self._connectors = connectors

    def get_default_map(self) -> OccupancyMap2D:
        return self._map_set.maps_by_floor[self._map_set.default_floor]

    def get_map(self, floor_id: str) -> OccupancyMap2D:
        return self._map_set.maps_by_floor.get(floor_id, self.get_default_map())

    def get_version(self) -> int:
        return int(self._map_set.version)

    def get_connector_graph(self) -> ConnectorGraph:
        return self._connectors

    def list_floors(self) -> list[str]:
        return list(self._map_set.maps_by_floor.keys())

    def find_floor_by_xy(self, xy: np.ndarray, preferred_floor: Optional[str] = None) -> str:
        if preferred_floor and preferred_floor in self._map_set.maps_by_floor:
            return preferred_floor
        x = float(xy[0])
        y = float(xy[1])
        for floor_id, m in self._map_set.maps_by_floor.items():
            x0 = float(m.origin_xy[0])
            y0 = float(m.origin_xy[1])
            x1 = x0 + float(m.extent_xy[0])
            y1 = y0 + float(m.extent_xy[1])
            if x0 <= x <= x1 and y0 <= y <= y1:
                return floor_id
        return self._map_set.default_floor

    def world_to_grid(self, floor_id: str, xy: np.ndarray) -> tuple[int, int]:
        m = self.get_map(floor_id)
        gx = int(round((float(xy[0]) - float(m.origin_xy[0])) / float(m.resolution)))
        gy = int(round((float(xy[1]) - float(m.origin_xy[1])) / float(m.resolution)))
        gx = max(0, min(m.grid.shape[1] - 1, gx))
        gy = max(0, min(m.grid.shape[0] - 1, gy))
        return gx, gy

    def grid_to_world(self, floor_id: str, gx: int, gy: int) -> np.ndarray:
        m = self.get_map(floor_id)
        x = float(m.origin_xy[0]) + float(gx) * float(m.resolution)
        y = float(m.origin_xy[1]) + float(gy) * float(m.resolution)
        return np.array([x, y], dtype=np.float32)

