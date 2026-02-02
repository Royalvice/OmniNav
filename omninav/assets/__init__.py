"""
OmniNav Assets Layer - 资产管理

提供场景资产加载器的抽象接口和路径解析工具。
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
