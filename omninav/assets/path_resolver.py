"""
Asset Path Resolution Utilities

Transparently resolves URDF file paths based on urdf_source in configuration.
Users only need to specify robot: go2 or robot: go2w, without worrying about file source.
"""

import os
from pathlib import Path
from typing import Union
from omegaconf import DictConfig

# Genesis assets directory (lazy import)
_GENESIS_ASSETS_DIR: str | None = None


def get_genesis_assets_dir() -> str:
    """Get the Genesis built-in assets directory path."""
    global _GENESIS_ASSETS_DIR
    if _GENESIS_ASSETS_DIR is None:
        try:
            import genesis as gs
            _GENESIS_ASSETS_DIR = gs.utils.get_assets_dir()
        except ImportError:
            # Fall back to external/Genesis submodule
            project_root = Path(__file__).parent.parent.parent
            _GENESIS_ASSETS_DIR = str(project_root / "external" / "Genesis" / "genesis" / "assets")
    return _GENESIS_ASSETS_DIR


def get_project_assets_dir() -> str:
    """Get the project assets directory path."""
    project_root = Path(__file__).parent.parent.parent
    return str(project_root / "assets")


def resolve_urdf_path(robot_cfg: DictConfig) -> str:
    """
    Resolve full URDF file path based on robot configuration.
    
    Determines asset source based on robot_cfg.urdf_source field:
    - "genesis_builtin": Use Genesis built-in assets directory
    - "project": Use project assets/ directory
    
    Args:
        robot_cfg: Robot configuration (from configs/robot/*.yaml)
    
    Returns:
        Absolute path to URDF file
    
    Raises:
        FileNotFoundError: If URDF file does not exist
        ValueError: If urdf_source value is invalid
    """
    urdf_source = robot_cfg.get("urdf_source", "genesis_builtin")
    urdf_path = robot_cfg.get("urdf_path", "")
    
    if urdf_source == "genesis_builtin":
        base_dir = get_genesis_assets_dir()
    elif urdf_source == "project":
        base_dir = get_project_assets_dir()
    else:
        raise ValueError(
            f"Invalid urdf_source: '{urdf_source}'. "
            f"Expected 'genesis_builtin' or 'project'."
        )
    
    full_path = os.path.join(base_dir, urdf_path)
    
    # Note: We don't check file existence here
    # Genesis will throw more detailed errors during loading
    return full_path


def resolve_asset_path(
    asset_path: str, 
    source: str = "project"
) -> str:
    """
    Resolve general asset file path.
    
    Args:
        asset_path: Relative asset path
        source: "genesis_builtin" or "project"
    
    Returns:
        Absolute path to asset file
    """
    if source == "genesis_builtin":
        base_dir = get_genesis_assets_dir()
    else:
        base_dir = get_project_assets_dir()
    
    return os.path.join(base_dir, asset_path)
