"""Export occupancy maps to Nav2-compatible map files."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from omninav.core.map.types import OccupancyMap2D


def _write_pgm(path: Path, image: np.ndarray) -> None:
    image = np.asarray(image, dtype=np.uint8)
    h, w = image.shape
    with path.open("wb") as f:
        f.write(f"P5\n{w} {h}\n255\n".encode("ascii"))
        f.write(image.tobytes())


def export_floor_map_to_nav2(floor_map: OccupancyMap2D, output_dir: str | Path, stem: str) -> tuple[Path, Path]:
    """Export one floor occupancy map to `<stem>.pgm` and `<stem>.yaml`."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pgm_path = out_dir / f"{stem}.pgm"
    yaml_path = out_dir / f"{stem}.yaml"

    # Nav2 trinary map convention: 0 occupied, 254 free.
    grid = np.asarray(floor_map.grid, dtype=np.uint8)
    img = np.where(grid > 0, 0, 254).astype(np.uint8)
    _write_pgm(pgm_path, np.flipud(img))

    yaml_text = (
        f"image: {pgm_path.name}\n"
        f"resolution: {float(floor_map.resolution):.6f}\n"
        f"origin: [{float(floor_map.origin_xy[0]):.6f}, {float(floor_map.origin_xy[1]):.6f}, 0.0]\n"
        "negate: 0\n"
        "occupied_thresh: 0.65\n"
        "free_thresh: 0.196\n"
        "mode: trinary\n"
    )
    yaml_path.write_text(yaml_text, encoding="utf-8")
    return pgm_path, yaml_path

