"""Catalog helpers for Getting Started GUI."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def list_scene_profiles() -> List[str]:
    scene_dir = _project_root() / "configs" / "scene"
    return sorted(p.stem for p in scene_dir.glob("*.yaml"))


def list_omninav_ready() -> Dict[str, List[str]]:
    # Import modules to ensure all registrations are executed.
    import omninav.robots  # noqa: F401
    import omninav.sensors  # noqa: F401
    import omninav.locomotion  # noqa: F401
    import omninav.algorithms  # noqa: F401
    import omninav.evaluation.tasks  # noqa: F401

    from omninav.core.registry import (
        ALGORITHM_REGISTRY,
        LOCOMOTION_REGISTRY,
        ROBOT_REGISTRY,
        SENSOR_REGISTRY,
        TASK_REGISTRY,
    )

    return {
        "robots": sorted(ROBOT_REGISTRY.registered_names),
        "sensors": sorted(SENSOR_REGISTRY.registered_names),
        "locomotions": sorted(LOCOMOTION_REGISTRY.registered_names),
        "algorithms": sorted(ALGORITHM_REGISTRY.registered_names),
        "tasks": sorted(TASK_REGISTRY.registered_names),
        "scenes": list_scene_profiles(),
    }


def list_genesis_candidates(max_items: int = 80) -> Dict[str, List[str]]:
    root = _project_root()
    urdf_root = root / "external" / "Genesis" / "genesis" / "assets" / "urdf"
    mjcf_root = root / "external" / "Genesis" / "genesis" / "assets" / "xml"

    urdf_candidates = sorted(str(p.relative_to(urdf_root)) for p in urdf_root.rglob("*.urdf"))[:max_items]
    mjcf_candidates = sorted(str(p.relative_to(mjcf_root)) for p in mjcf_root.rglob("*.xml"))[:max_items]

    return {
        "urdf_candidates": urdf_candidates,
        "mjcf_candidates": mjcf_candidates,
        "sensor_capabilities": [
            "Camera (RGB/Depth/Segmentation)",
            "Raycaster/Lidar",
            "IMU",
            "Contact/ContactForce",
        ],
    }
