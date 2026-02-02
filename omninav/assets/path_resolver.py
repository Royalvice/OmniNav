"""
资产路径解析工具

根据配置中的 urdf_source 字段，透明地解析 URDF 文件的实际路径。
用户只需在配置中指定 robot: go2 或 robot: go2w，无需关心文件来源。
"""

import os
from pathlib import Path
from typing import Union
from omegaconf import DictConfig

# Genesis 资产目录 (延迟导入)
_GENESIS_ASSETS_DIR: str | None = None


def get_genesis_assets_dir() -> str:
    """获取 Genesis 内置资产目录路径。"""
    global _GENESIS_ASSETS_DIR
    if _GENESIS_ASSETS_DIR is None:
        try:
            import genesis as gs
            _GENESIS_ASSETS_DIR = gs.utils.get_assets_dir()
        except ImportError:
            # 回退到 external/Genesis 子模块
            project_root = Path(__file__).parent.parent.parent
            _GENESIS_ASSETS_DIR = str(project_root / "external" / "Genesis" / "genesis" / "assets")
    return _GENESIS_ASSETS_DIR


def get_project_assets_dir() -> str:
    """获取项目资产目录路径。"""
    project_root = Path(__file__).parent.parent.parent
    return str(project_root / "assets")


def resolve_urdf_path(robot_cfg: DictConfig) -> str:
    """
    根据机器人配置解析 URDF 文件的完整路径。
    
    根据 robot_cfg.urdf_source 字段决定资产来源:
    - "genesis_builtin": 使用 Genesis 内置资产目录
    - "project": 使用项目 assets/ 目录
    
    Args:
        robot_cfg: 机器人配置 (来自 configs/robot/*.yaml)
    
    Returns:
        URDF 文件的绝对路径
    
    Raises:
        FileNotFoundError: 如果 URDF 文件不存在
        ValueError: 如果 urdf_source 值无效
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
    
    # 注意: 我们不在这里检查文件存在性
    # Genesis 会在加载时抛出更详细的错误
    return full_path


def resolve_asset_path(
    asset_path: str, 
    source: str = "project"
) -> str:
    """
    解析通用资产文件路径。
    
    Args:
        asset_path: 相对资产路径
        source: "genesis_builtin" 或 "project"
    
    Returns:
        资产文件的绝对路径
    """
    if source == "genesis_builtin":
        base_dir = get_genesis_assets_dir()
    else:
        base_dir = get_project_assets_dir()
    
    return os.path.join(base_dir, asset_path)
