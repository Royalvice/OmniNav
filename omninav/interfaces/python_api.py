"""
OmniNav Python API - Main Interface Class

Provides Gym-style simulation environment interface.
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from omninav.robots.base import RobotBase, RobotState
from omninav.locomotion.base import LocomotionControllerBase
from omninav.algorithms.base import AlgorithmBase
from omninav.evaluation.base import TaskBase, TaskResult


class OmniNavEnv:
    """
    OmniNav Main Interface Class (Gym-style).
    
    Provides clean API for:
    - Creating simulation environment
    - Controlling robots
    - Running evaluation tasks
    
    Usage example:
        >>> from omninav import OmniNavEnv
        >>> 
        >>> env = OmniNavEnv(config_path="configs")
        >>> obs = env.reset()
        >>> 
        >>> while not env.is_done:
        ...     action = env.algorithm.step(obs)  # or custom algorithm
        ...     obs, info = env.step(action)
        >>> 
        >>> result = env.get_result()
        >>> print(f"Success: {result.success}")
    """
    
    def __init__(
        self, 
        cfg: Optional[DictConfig] = None,
        config_path: Optional[str] = None,
        config_name: str = "config",
    ):
        """
        Initialize OmniNav environment.
        
        Args:
            cfg: Configuration object passed directly (highest priority)
            config_path: Path to Hydra configuration directory
            config_name: Configuration file name (without extension)
        """
        self.cfg = self._load_config(cfg, config_path, config_name)
        
        # Component references (lazy initialization)
        self.sim = None  # SimulationManager
        self.robot: Optional[RobotBase] = None
        self.locomotion: Optional[LocomotionControllerBase] = None
        self.algorithm: Optional[AlgorithmBase] = None
        self.task: Optional[TaskBase] = None
        
        self._initialized = False
        self._step_count = 0
    
    def _load_config(
        self, 
        cfg: Optional[DictConfig],
        config_path: Optional[str],
        config_name: str,
    ) -> DictConfig:
        """Load configuration."""
        if cfg is not None:
            return cfg
        
        if config_path is not None:
            # Use Hydra to compose configuration
            try:
                from hydra import compose, initialize_config_dir
                from hydra.core.global_hydra import GlobalHydra
                
                # Clear any existing Hydra instance
                if GlobalHydra.instance().is_initialized():
                    GlobalHydra.instance().clear()
                
                config_dir = str(Path(config_path).absolute())
                with initialize_config_dir(config_dir=config_dir, version_base=None):
                    cfg = compose(config_name=config_name)
                return cfg
            except ImportError:
                # Fall back to direct YAML loading
                import yaml
                config_file = Path(config_path) / f"{config_name}.yaml"
                with open(config_file) as f:
                    return OmegaConf.create(yaml.safe_load(f))
        
        # Default empty configuration
        return OmegaConf.create({})
    
    def _initialize(self) -> None:
        """
        Initialize all components based on configuration.
        
        Lazy initialization, called on first reset().
        """
        if self._initialized:
            return
        
        # TODO: Implement component initialization
        # 1. Initialize SimulationManager
        # 2. Load scene
        # 3. Create robot
        # 4. Mount sensors
        # 5. Create locomotion controller
        # 6. Create algorithm (optional)
        # 7. Create evaluation task (optional)
        # 8. Build scene
        
        self._initialized = True
    
    def reset(self) -> Dict[str, Any]:
        """
        Reset environment.
        
        Returns:
            Initial observation
        """
        if not self._initialized:
            self._initialize()
        
        self._step_count = 0
        
        # Reset simulation
        if self.sim is not None:
            self.sim.reset()
        
        # Reset locomotion controller
        if self.locomotion is not None:
            self.locomotion.reset()
        
        # Reset task and get task info
        task_info = {}
        if self.task is not None:
            task_info = self.task.reset()
        
        # Reset algorithm
        if self.algorithm is not None:
            self.algorithm.reset(task_info)
        
        return self._get_observation()
    
    def step(self, action: Optional[np.ndarray] = None) -> Tuple[Dict[str, Any], Dict]:
        """
        Execute one simulation step.
        
        Args:
            action: cmd_vel [vx, vy, wz], if None uses built-in algorithm
        
        Returns:
            obs: New observation
            info: Additional information
        """
        # If no action provided, use built-in algorithm
        if action is None and self.algorithm is not None:
            obs = self._get_observation()
            action = self.algorithm.step(obs)
        
        if action is None:
            action = np.zeros(3)
        
        # Locomotion control
        if self.locomotion is not None:
            self.locomotion.step(action)
        
        # Physics simulation
        if self.sim is not None:
            self.sim.step()
        
        self._step_count += 1
        
        # Record task data
        if self.task is not None and self.robot is not None:
            robot_state = self.robot.get_state()
            self.task.step(robot_state, action)
        
        obs = self._get_observation()
        info = {"action": action, "step": self._step_count}
        
        return obs, info
    
    def _get_observation(self) -> Dict[str, Any]:
        """Get current observation."""
        obs = {}
        
        if self.robot is not None:
            obs.update(self.robot.get_observations())
            obs["robot_state"] = self.robot.get_state()
        
        if self.sim is not None:
            obs["sim_time"] = self.sim.get_sim_time()
        
        return obs
    
    @property
    def is_done(self) -> bool:
        """Whether task is finished."""
        if self.task is not None and self.robot is not None:
            return self.task.is_terminated(self.robot.get_state())
        if self.algorithm is not None:
            return self.algorithm.is_done
        return False
    
    def get_result(self) -> Optional[TaskResult]:
        """Get task result."""
        if self.task is not None:
            return self.task.compute_result()
        return None
    
    def close(self) -> None:
        """Close environment and release resources."""
        if self.sim is not None:
            # self.sim.scene.destroy()
            pass
        self._initialized = False
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
