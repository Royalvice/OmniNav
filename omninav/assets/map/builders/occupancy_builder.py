"""Build occupancy maps from normalized scene map specs."""

from __future__ import annotations

from typing import Dict, Any, List, Tuple
import math

import numpy as np

from omninav.core.map.types import (
    OccupancyMap2D,
    OccupancyMapSet,
    ConnectorNode,
    ConnectorEdge,
    ConnectorGraph,
)


class OccupancyBuilder:
    """Rasterize obstacle primitives into per-floor 2D occupancy grids."""

    def __init__(self, inflation_radius: float = 0.15):
        self._inflation_radius = float(inflation_radius)

    def build(self, map_spec: Dict[str, Any]) -> Tuple[OccupancyMapSet, ConnectorGraph]:
        floors: List[Dict[str, Any]] = list(map_spec.get("floors", []))
        if not floors:
            raise ValueError("Map spec contains no floors.")

        maps: Dict[str, OccupancyMap2D] = {}
        for floor in floors:
            occ = self._build_floor(floor)
            maps[occ.floor_id] = occ

        default_floor = str(floors[0].get("id", "floor_0"))
        map_set = OccupancyMapSet(maps_by_floor=maps, default_floor=default_floor, version=1)
        graph = self._build_connector_graph(map_spec.get("connectors", []))
        return map_set, graph

    def _build_floor(self, floor: Dict[str, Any]) -> OccupancyMap2D:
        floor_id = str(floor.get("id", "floor_0"))
        origin = np.asarray(floor.get("origin", [-8.0, -8.0]), dtype=np.float32)[:2]
        extent = np.asarray(floor.get("extent", [16.0, 16.0]), dtype=np.float32)[:2]
        resolution = max(float(floor.get("resolution", 0.1)), 1e-3)
        width = max(1, int(math.ceil(float(extent[0]) / resolution)))
        height = max(1, int(math.ceil(float(extent[1]) / resolution)))
        grid = np.zeros((height, width), dtype=np.uint8)

        obstacles = list(floor.get("obstacles", []))
        for obstacle in obstacles:
            self._rasterize_obstacle(grid, origin, resolution, obstacle)

        if self._inflation_radius > 1e-6:
            grid = self._inflate(grid, cells=max(1, int(round(self._inflation_radius / resolution))))

        return OccupancyMap2D(
            floor_id=floor_id,
            grid=grid,
            resolution=resolution,
            origin_xy=origin,
            extent_xy=extent,
            obstacles=obstacles,
        )

    def _rasterize_obstacle(self, grid: np.ndarray, origin: np.ndarray, res: float, obstacle: Dict[str, Any]) -> None:
        typ = str(obstacle.get("type", "box")).lower()
        pos = obstacle.get("position", [0.0, 0.0, 0.0])
        cx = float(pos[0]) if len(pos) > 0 else 0.0
        cy = float(pos[1]) if len(pos) > 1 else 0.0
        h, w = grid.shape

        if typ == "box":
            size = obstacle.get("size", [1.0, 1.0, 1.0])
            sx = max(0.05, float(size[0]) * 0.5)
            sy = max(0.05, float(size[1]) * 0.5)
            x0 = int(math.floor((cx - sx - origin[0]) / res))
            x1 = int(math.ceil((cx + sx - origin[0]) / res))
            y0 = int(math.floor((cy - sy - origin[1]) / res))
            y1 = int(math.ceil((cy + sy - origin[1]) / res))
            x0 = max(0, min(w - 1, x0))
            x1 = max(0, min(w - 1, x1))
            y0 = max(0, min(h - 1, y0))
            y1 = max(0, min(h - 1, y1))
            if x0 <= x1 and y0 <= y1:
                grid[y0 : y1 + 1, x0 : x1 + 1] = 1
            return

        radius = float(obstacle.get("radius", 0.5))
        rad_cells = max(1, int(math.ceil(radius / res)))
        cx_i = int(round((cx - origin[0]) / res))
        cy_i = int(round((cy - origin[1]) / res))
        for dy in range(-rad_cells, rad_cells + 1):
            for dx in range(-rad_cells, rad_cells + 1):
                if dx * dx + dy * dy > rad_cells * rad_cells:
                    continue
                x = cx_i + dx
                y = cy_i + dy
                if 0 <= x < w and 0 <= y < h:
                    grid[y, x] = 1

    @staticmethod
    def _inflate(grid: np.ndarray, cells: int) -> np.ndarray:
        if cells <= 0:
            return grid
        occ = np.argwhere(grid > 0)
        if occ.size == 0:
            return grid
        h, w = grid.shape
        out = grid.copy()
        for y, x in occ:
            y0 = max(0, y - cells)
            y1 = min(h - 1, y + cells)
            x0 = max(0, x - cells)
            x1 = min(w - 1, x + cells)
            out[y0 : y1 + 1, x0 : x1 + 1] = 1
        return out

    @staticmethod
    def _build_connector_graph(connectors_cfg: List[Dict[str, Any]]) -> ConnectorGraph:
        nodes: Dict[str, ConnectorNode] = {}
        edges: List[ConnectorEdge] = []

        for i, c in enumerate(connectors_cfg):
            cid = str(c.get("id", f"connector_{i}"))
            from_floor = str(c.get("from_floor", "floor_0"))
            to_floor = str(c.get("to_floor", from_floor))
            from_xy = np.asarray(c.get("from_xy", c.get("entry_xy", [0.0, 0.0])), dtype=np.float32)[:2]
            to_xy = np.asarray(c.get("to_xy", c.get("exit_xy", [0.0, 0.0])), dtype=np.float32)[:2]
            ctype = str(c.get("type", "stairs"))
            cost = float(c.get("cost", 1.0))

            n_from = f"{cid}:from"
            n_to = f"{cid}:to"
            nodes[n_from] = ConnectorNode(node_id=n_from, floor_id=from_floor, xy=from_xy)
            nodes[n_to] = ConnectorNode(node_id=n_to, floor_id=to_floor, xy=to_xy)
            edges.append(
                ConnectorEdge(
                    edge_id=cid,
                    from_node_id=n_from,
                    to_node_id=n_to,
                    connector_type=ctype,
                    cost=cost,
                )
            )
            if bool(c.get("bidirectional", True)):
                edges.append(
                    ConnectorEdge(
                        edge_id=f"{cid}_rev",
                        from_node_id=n_to,
                        to_node_id=n_from,
                        connector_type=ctype,
                        cost=cost,
                    )
                )

        return ConnectorGraph(nodes=nodes, edges=edges)

