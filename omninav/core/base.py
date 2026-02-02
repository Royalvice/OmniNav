"""
Simulation Manager Abstract Base Class

Defines the interface for simulation core management.
"""

from abc import ABC, abstractmethod
from typing import Optional, Any, TYPE_CHECKING
from omegaconf import DictConfig

if TYPE_CHECKING:
    from omninav.robots.base import RobotBase


class SimulationManagerBase(ABC):
    """
    Abstract base class for simulation manager.
    
    Responsibilities:
    - Genesis engine initialization
    - Scene management
    - Simulation loop control
    - Robot and asset management
    """
    
    @abstractmethod
    def initialize(self, cfg: DictConfig) -> None:
        """
        Initialize the simulation environment.
        
        Args:
            cfg: Simulation configuration (from Hydra)
        """
        pass
    
    @abstractmethod
    def build(self) -> None:
        """
        Build the scene.
        
        Called after all entities are added, triggers Genesis scene.build().
        """
        pass
    
    @abstractmethod
    def step(self) -> None:
        """
        Advance one physics simulation step.
        
        Calls Genesis scene.step() to execute one simulation timestep.
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """
        Reset simulation state.
        
        Restores all entities to their initial state.
        """
        pass
    
    @abstractmethod
    def get_sim_time(self) -> float:
        """
        Get current simulation time.
        
        Returns:
            Simulation time in seconds
        """
        pass
    
    @abstractmethod
    def add_robot(self, robot: "RobotBase") -> None:
        """
        Add a robot to the scene.
        
        Args:
            robot: Robot instance
        """
        pass
    
    @abstractmethod
    def load_scene(self, scene_cfg: DictConfig) -> None:
        """
        Load scene assets.
        
        Args:
            scene_cfg: Scene configuration
        """
        pass
    
    @property
    @abstractmethod
    def scene(self) -> Any:
        """
        Get the Genesis scene object.
        
        Returns:
            Genesis scene object
        """
        pass
