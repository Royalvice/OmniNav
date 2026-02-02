"""
资产加载器抽象基类

定义资产加载的接口规范。
"""

from abc import ABC, abstractmethod
from typing import Any, List, TYPE_CHECKING
from omegaconf import DictConfig

if TYPE_CHECKING:
    import genesis as gs


class AssetLoaderBase(ABC):
    """
    资产加载器抽象基类。
    
    所有资产加载器 (USD, GLB, Mesh 等) 必须继承此类。
    """
    
    # 支持的文件扩展名
    SUPPORTED_EXTENSIONS: List[str] = []
    
    @abstractmethod
    def load(
        self, 
        file_path: str, 
        scene: "gs.Scene", 
        cfg: DictConfig
    ) -> Any:
        """
        加载资产到 Genesis 场景。
        
        Args:
            file_path: 资产文件路径
            scene: Genesis 场景对象
            cfg: 加载配置 (位置、缩放等)
        
        Returns:
            加载后的 Genesis entity 或 entity 列表
        """
        pass
    
    @classmethod
    def can_load(cls, file_path: str) -> bool:
        """
        检查是否支持该文件类型。
        
        Args:
            file_path: 文件路径
            
        Returns:
            True 如果支持，否则 False
        """
        return any(
            file_path.lower().endswith(ext) 
            for ext in cls.SUPPORTED_EXTENSIONS
        )
