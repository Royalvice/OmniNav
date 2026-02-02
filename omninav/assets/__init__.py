"""
OmniNav Assets Layer - Asset Management

Provides abstract interface for scene asset loaders and path resolution utilities.
"""

from omninav.assets.base import AssetLoaderBase
from omninav.assets.path_resolver import (
    resolve_urdf_path,
    resolve_asset_path,
    get_genesis_assets_dir,
    get_project_assets_dir,
)

__all__ = [
    "AssetLoaderBase",
    "resolve_urdf_path",
    "resolve_asset_path",
    "get_genesis_assets_dir",
    "get_project_assets_dir",
]
