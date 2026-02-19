"""
OmniNav Python API - Main Interface Class

Provides Gym-style simulation environment interface.
Thin wrapper delegating to SimulationRuntime.
"""

from typing import Dict, Any, Optional, Tuple, List, TYPE_CHECKING
import numpy as np
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from omninav.core.runtime import SimulationRuntime
from omninav.core.hooks import HookManager
from omninav.core.types import Observation, Action, TaskResult
from omninav.interfaces.ros2.adapter import Ros2Adapter

if TYPE_CHECKING:
    from omninav.robots.base import RobotBase
    from omninav.locomotion.base import LocomotionControllerBase
    from omninav.algorithms.base import AlgorithmBase
    from omninav.evaluation.base import TaskBase
    from omninav.interfaces.ros2.bridge import ROS2Bridge


class OmniNavEnv:
    """
    OmniNav Main Interface Class (Gym-style).

    Thin wrapper over SimulationRuntime providing a clean API for:
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
        ...     obs, info = env.step()  # uses built-in algorithm
        >>>
        >>> result = env.get_result()
        >>> print(f"Success: {result.success}")

    Or with explicit actions:
        >>> obs, info = env.step(Action(cmd_vel=np.array([1, 0, 0])))
    """

    def __init__(
        self,
        cfg: Optional[DictConfig] = None,
        config_path: Optional[str] = None,
        config_name: str = "config",
        backend: str = "genesis",
        overrides: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Initialize OmniNav environment.

        Args:
            cfg: Configuration object passed directly (highest priority)
            config_path: Path to Hydra configuration directory
            config_name: Configuration file name (without extension)
            backend: Simulation backend ("genesis", "ros2", "replay")
            overrides: List of Hydra overrides (e.g. ["task=inspection"])
            **kwargs: Legacy support for kwargs overrides (deprecated)
        """
        # Merge kwargs into overrides if present (for backward compatibility)
        if kwargs:
            if overrides is None:
                overrides = []
            for k, v in kwargs.items():
                overrides.append(f"{k}={v}")

        self.cfg = self._load_config(cfg, config_path, config_name, overrides)
        self._runtime: Optional[SimulationRuntime] = None
        self._initialized = False
        self._ros2_bridge: Optional["ROS2Bridge"] = None

    @classmethod
    def from_config(cls, config_path: str, overrides: Optional[List[str]] = None) -> "OmniNavEnv":
        """
        Create OmniNavEnv from config directory with optional overrides.
        """
        return cls(config_path=config_path, overrides=overrides)

    def _load_config(
        self,
        cfg: Optional[DictConfig],
        config_path: Optional[str],
        config_name: str,
        overrides: Optional[List[str]],
    ) -> DictConfig:
        """Load configuration from various sources."""
        if cfg is not None:
            # If explicit cfg provided, apply overrides on top
            result = cfg
            if overrides:
                 override_conf = OmegaConf.from_dotlist(overrides)
                 result = OmegaConf.merge(result, override_conf)
            return result
        
        if config_path is not None:
            try:
                from hydra import compose, initialize_config_dir
                from hydra.core.global_hydra import GlobalHydra

                if GlobalHydra.instance().is_initialized():
                    GlobalHydra.instance().clear()

                config_dir = str(Path(config_path).absolute())
                with initialize_config_dir(config_dir=config_dir, version_base=None):
                    # Pass overrides to compose!
                    result = compose(config_name=config_name, overrides=overrides if overrides else [])
            except ImportError:
                import yaml
                # Fallback: manual load + dotlist merge
                config_file = Path(config_path) / f"{config_name}.yaml"
                if config_file.exists():
                    with open(config_file) as f:
                        result = OmegaConf.create(yaml.safe_load(f))
                else:
                    result = OmegaConf.create({})
                
                if overrides:
                    override_conf = OmegaConf.from_dotlist(overrides)
                    result = OmegaConf.merge(result, override_conf)
        else:
            result = OmegaConf.create({})
            if overrides:
                override_conf = OmegaConf.from_dotlist(overrides)
                result = OmegaConf.merge(result, override_conf)

        return result

    def _initialize(self) -> None:
        """
        Initialize runtime with all components.

        Lazy initialization, called on first reset().
        """
        if self._initialized:
            return

        self._runtime = SimulationRuntime(self.cfg)

        # Lazy initialization
        if self._runtime is None:
            self._runtime = SimulationRuntime(self.cfg)
        
        # 1. Initialize SimulationManager
        from omninav.core.simulation_manager import GenesisSimulationManager
        sim = GenesisSimulationManager()
        sim.initialize(self.cfg)
        self._runtime.sim = sim

        ros2_cfg = self.cfg.get("ros2", {})
        if ros2_cfg.get("enabled", False):
            from omninav.interfaces.ros2.bridge import ROS2Bridge

            self._ros2_bridge = ROS2Bridge(ros2_cfg, sim)

        # 2. Load scene
        if "scene" in self.cfg:
            sim.load_scene(self.cfg.scene)
        
        # Trigger registration by importing submodules
        import omninav.robots
        import omninav.sensors
        import omninav.locomotion
        import omninav.algorithms
        import omninav.evaluation.tasks
        import omninav.evaluation.metrics

        from omninav.core.registry import ROBOT_REGISTRY, SENSOR_REGISTRY, LOCOMOTION_REGISTRY, ALGORITHM_REGISTRY, TASK_REGISTRY
        
        robot_cfg = self.cfg.get("robot", {})
        robot_type = robot_cfg.get("type", "unitree_go2")
        if not isinstance(robot_cfg, DictConfig):
            robot_cfg = OmegaConf.create(robot_cfg)
        if "type" not in robot_cfg:
            OmegaConf.update(robot_cfg, "type", robot_type)
        robot = ROBOT_REGISTRY.build(robot_cfg, scene=sim.scene)
        sim.add_robot(robot)
        self._runtime.robots.append(robot)
        self.robot = robot  # main robot reference
        
        # 4. Create Locomotion (to get required sensors)
        loco_cfg = self.cfg.get("locomotion", {})
        loco_type = loco_cfg.get("type", "kinematic_gait")
        if not isinstance(loco_cfg, DictConfig):
            loco_cfg = OmegaConf.create(loco_cfg)
        if "type" not in loco_cfg:
            OmegaConf.update(loco_cfg, "type", loco_type)
        locomotion = LOCOMOTION_REGISTRY.build(loco_cfg, robot=robot)
        self._runtime.locomotions.append(locomotion)
        self.locomotion = locomotion
        
        # 5. Create and mount sensors
        # Merge config sensors and locomotion required sensors
        sensors_to_create = {}
        
        # From config
        if "sensor" in self.cfg:
            for name, s_cfg in self.cfg.sensor.items():
                if isinstance(s_cfg, (dict, DictConfig, dict)) and "type" in s_cfg:
                    sensors_to_create[name] = s_cfg
        
        # From locomotion
        if hasattr(locomotion, "required_sensors"):
            req_sensors = locomotion.required_sensors
            if req_sensors:
                sensors_to_create.update(req_sensors)
        
        created_sensors = {}
        for name, s_cfg in sensors_to_create.items():
            if not isinstance(s_cfg, (dict, DictConfig)):
                 # Skip invalid configs
                 continue
            s_type = s_cfg.get("type")
            if s_type:
                try:
                    if not isinstance(s_cfg, DictConfig):
                        s_cfg = OmegaConf.create(s_cfg)
                    sensor = SENSOR_REGISTRY.build(s_cfg, scene=sim.scene, robot=robot)
                    robot.mount_sensor(name, sensor)
                    created_sensors[name] = sensor
                    self._runtime.sensors[name] = sensor
                except Exception as e:
                    print(f"Failed to create sensor {name}: {e}")
        
        # Bind sensors to locomotion
        locomotion.bind_sensors(created_sensors)
        
        # 6. Create Algorithm
        if "algorithm" in self.cfg:
            algo_cfg = self.cfg.algorithm
            # Ensure type is set (Hydra might put it in defaults, but accessing node should have it)
            algorithm = ALGORITHM_REGISTRY.build(algo_cfg)
            self._runtime.algorithms.append(algorithm)
            self.algorithm = algorithm
            
        # 7. Create Task
        if "task" in self.cfg:
            task_cfg = self.cfg.task
            task = TASK_REGISTRY.build(task_cfg)
            self._runtime.task = task
            self.task = task
            
        # 8. Build Simulation
        self._runtime.build()

        if self._ros2_bridge is not None and hasattr(self, "robot") and self.robot is not None:
            self._ros2_bridge.setup(self.robot)
        
        self._initialized = True

    def reset(self) -> List[Observation]:
        """
        Reset environment.

        Returns:
            List of initial observations (one per robot)
        """
        if not self._initialized:
            self._initialize()

        return self._runtime.reset()

    def step(
        self,
        actions: Optional[List[Action]] = None,
    ) -> Tuple[List[Observation], Dict[str, Any]]:
        """
        Execute one simulation step.

        Args:
            actions: List of Action dicts (one per robot), or None to use built-in algorithms

        Returns:
            observations: List of Observation (one per robot)
            info: Step metadata
        """
        if self._ros2_bridge is not None and self._ros2_bridge.enabled:
            self._ros2_bridge.spin_once()
            if actions is None and self._ros2_bridge.control_source == "ros2":
                cmd_vel = self._ros2_bridge.get_external_cmd_vel()
                if cmd_vel is None:
                    cmd_vel = np.zeros(3, dtype=np.float32)
                actions = [Action(cmd_vel=Ros2Adapter.normalize_cmd_vel_batch(cmd_vel, "ros2_input"))]

        observations, info = self._runtime.step(actions)

        if self._ros2_bridge is not None and self._ros2_bridge.enabled:
            for obs in observations:
                self._ros2_bridge.publish_observation(obs)
            # Optional mirror output when running python-side control.
            if actions and self._ros2_bridge.control_source == "python":
                self._ros2_bridge.publish_cmd_vel(actions[0]["cmd_vel"])

        return observations, info

    @property
    def hooks(self) -> HookManager:
        """Access hook manager for event registration."""
        if self._runtime is None:
            self._initialize()
        return self._runtime.hooks

    @property
    def is_done(self) -> bool:
        """Whether task is finished."""
        if self._runtime is None:
            return False
        return self._runtime.is_done

    def get_result(self) -> Optional[TaskResult]:
        """Get task result."""
        if self._runtime is None:
            return None
        return self._runtime.get_result()

    @property
    def step_count(self) -> int:
        """Current step count."""
        return self._runtime.step_count if self._runtime else 0

    @property
    def sim_time(self) -> float:
        """Current simulation time."""
        return self._runtime.sim_time if self._runtime else 0.0

    def close(self) -> None:
        """Close environment and release resources."""
        if self._ros2_bridge is not None:
            self._ros2_bridge.shutdown()
            self._ros2_bridge = None
        self._initialized = False
        self._runtime = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
