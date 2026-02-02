"""
Genesis Simulation Manager Implementation

Concrete implementation of simulation manager based on Genesis physics engine.

Design principles (following Genesis official examples):
1. gs.init() is called once during module initialization
2. gs.Scene() creates scene with configuration via gs.options.*
3. scene.add_entity() adds entities, then scene.build() to compile
4. scene.step() advances simulation

References:
- examples/tutorials/parallel_simulation.py
- examples/locomotion/go2_env.py
- examples/tutorials/control_your_robot.py
"""

from typing import Any, Dict, List, Optional, TYPE_CHECKING
from omegaconf import DictConfig
import numpy as np

from omninav.core.base import SimulationManagerBase

if TYPE_CHECKING:
    from omninav.robots.base import RobotBase
    import genesis as gs


class GenesisSimulationManager(SimulationManagerBase):
    """
    Genesis Simulation Manager.
    
    Wraps the Genesis engine, providing a unified simulation management interface.
    
    Usage:
    ```python
    sim = GenesisSimulationManager()
    sim.initialize(cfg)
    sim.add_robot(robot)
    sim.load_scene(scene_cfg)
    sim.build()
    
    while not done:
        sim.step()
    ```
    
    Attributes:
        cfg: Hydra configuration
        _scene: Genesis scene object
        _robots: List of added robots
        _is_built: Whether scene has been built
        _n_envs: Number of parallel environments
    """
    
    def __init__(self):
        self.cfg: Optional[DictConfig] = None
        self._scene: Optional[Any] = None  # gs.Scene
        self._robots: List["RobotBase"] = []
        self._is_built: bool = False
        self._n_envs: int = 1
        self._gs_initialized: bool = False
    
    def initialize(self, cfg: DictConfig) -> None:
        """
        Initialize Genesis engine and simulation scene.
        
        Args:
            cfg: Complete Hydra configuration, should contain:
                - simulation.dt: Simulation timestep
                - simulation.substeps: Physics substeps
                - simulation.backend: "gpu" or "cpu"
                - simulation.n_envs: Number of parallel environments
                - simulation.show_viewer: Whether to show visualization window
        """
        import genesis as gs
        
        self.cfg = cfg
        
        # Get parameters from config
        sim_cfg = cfg.get("simulation", {})
        dt = sim_cfg.get("dt", 0.01)
        substeps = sim_cfg.get("substeps", 2)
        backend = sim_cfg.get("backend", "gpu")
        show_viewer = sim_cfg.get("show_viewer", False)
        self._n_envs = sim_cfg.get("n_envs", 1)
        
        # Initialize Genesis (only once)
        if not self._gs_initialized:
            gs_backend = gs.gpu if backend == "gpu" else gs.cpu
            gs.init(backend=gs_backend)
            self._gs_initialized = True
        
        # Create scene (following Genesis example patterns)
        self._scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=dt,
                substeps=substeps,
            ),
            rigid_options=gs.options.RigidOptions(
                enable_self_collision=sim_cfg.get("enable_self_collision", False),
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=tuple(sim_cfg.get("camera_pos", [2.0, 0.0, 2.5])),
                camera_lookat=tuple(sim_cfg.get("camera_lookat", [0.0, 0.0, 0.0])),
                camera_fov=sim_cfg.get("camera_fov", 40),
                max_FPS=int(1.0 / dt),
            ),
            show_viewer=show_viewer,
        )
    
    def build(self) -> None:
        """
        Build the simulation scene.
        
        Calls Genesis scene.build() to complete physics configuration.
        Must be called after adding all entities.
        """
        if self._scene is None:
            raise RuntimeError("Scene not initialized. Call initialize() first.")
        
        if self._is_built:
            raise RuntimeError("Scene already built. Cannot build twice.")
        
        # Get environment spacing configuration
        env_spacing = self.cfg.get("simulation", {}).get("env_spacing", (1.0, 1.0))
        
        # Build scene (with parallel environments)
        self._scene.build(n_envs=self._n_envs, env_spacing=env_spacing)
        self._is_built = True
    
    def step(self) -> None:
        """
        Advance one simulation step.
        
        Calls Genesis scene.step().
        """
        if not self._is_built:
            raise RuntimeError("Scene not built. Call build() first.")
        
        self._scene.step()
    
    def reset(self) -> None:
        """
        Reset all robots to initial state.
        """
        for robot in self._robots:
            robot.reset()
    
    def get_sim_time(self) -> float:
        """
        Get current simulation time.
        
        Returns:
            Simulation time in seconds
        """
        if self._scene is None:
            return 0.0
        return self._scene.t
    
    def add_robot(self, robot: "RobotBase") -> None:
        """
        Add a robot to the scene.
        
        The robot's spawn() method will be called to create the entity.
        
        Args:
            robot: Robot instance (configured but not spawned)
        """
        if self._is_built:
            raise RuntimeError("Cannot add robot after scene is built.")
        
        # Robot handles its own spawn logic
        robot.spawn()
        self._robots.append(robot)
    
    def load_scene(self, scene_cfg: DictConfig) -> None:
        """
        Load scene assets based on configuration.
        
        Supports:
        - ground_plane: Load ground plane
        - obstacles: List of static obstacles
        - terrain: Terrain configuration
        
        Args:
            scene_cfg: Scene configuration (from configs/scene/*.yaml)
        """
        import genesis as gs
        
        if self._scene is None:
            raise RuntimeError("Scene not initialized.")
        
        if self._is_built:
            raise RuntimeError("Cannot load scene after build.")
        
        # Ground plane
        if scene_cfg.get("ground_plane", {}).get("enabled", True):
            ground_cfg = scene_cfg.get("ground_plane", {})
            self._scene.add_entity(
                gs.morphs.Plane(
                    pos=tuple(ground_cfg.get("position", [0, 0, 0])),
                )
            )
        
        # Static obstacles
        for obstacle in scene_cfg.get("obstacles", []):
            morph_type = obstacle.get("type", "box")
            pos = tuple(obstacle.get("position", [0, 0, 0]))
            
            if morph_type == "box":
                size = tuple(obstacle.get("size", [1, 1, 1]))
                self._scene.add_entity(
                    gs.morphs.Box(size=size, pos=pos, fixed=True)
                )
            elif morph_type == "cylinder":
                self._scene.add_entity(
                    gs.morphs.Cylinder(
                        height=obstacle.get("height", 1.0),
                        radius=obstacle.get("radius", 0.5),
                        pos=pos,
                        fixed=True,
                    )
                )
            elif morph_type == "sphere":
                self._scene.add_entity(
                    gs.morphs.Sphere(
                        radius=obstacle.get("radius", 0.5),
                        pos=pos,
                        fixed=True,
                    )
                )
    
    @property
    def scene(self) -> Any:
        """Get the Genesis scene object."""
        return self._scene
    
    @property
    def n_envs(self) -> int:
        """Get the number of parallel environments."""
        return self._n_envs
    
    @property
    def dt(self) -> float:
        """Get the simulation timestep."""
        if self.cfg is None:
            return 0.01
        return self.cfg.get("simulation", {}).get("dt", 0.01)
    
    @property
    def robots(self) -> List["RobotBase"]:
        """Get all added robots."""
        return self._robots
