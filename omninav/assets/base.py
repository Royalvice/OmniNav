"""
Asset Loader Abstract Base Class

Defines the interface for asset loading.
"""

from abc import ABC, abstractmethod
from typing import Any, List, TYPE_CHECKING
from omegaconf import DictConfig

if TYPE_CHECKING:
    import genesis as gs


class AssetLoaderBase(ABC):
    """
    Abstract base class for asset loaders.
    
    All asset loaders (USD, GLB, Mesh, etc.) must inherit from this class.
    """
    
    # Supported file extensions
    SUPPORTED_EXTENSIONS: List[str] = []
    
    @abstractmethod
    def load(
        self, 
        file_path: str, 
        scene: "gs.Scene", 
        cfg: DictConfig
    ) -> Any:
        """
        Load asset into Genesis scene.
        
        Args:
            file_path: Asset file path
            scene: Genesis scene object
            cfg: Load configuration (position, scale, etc.)
        
        Returns:
            Loaded Genesis entity or list of entities
        """
        pass
    
    @classmethod
    def can_load(cls, file_path: str) -> bool:
        """
        Check if this loader supports the file type.
        
        Args:
            file_path: File path
            
        Returns:
            True if supported, False otherwise
        """
        return any(
            file_path.lower().endswith(ext) 
            for ext in cls.SUPPORTED_EXTENSIONS
        )
