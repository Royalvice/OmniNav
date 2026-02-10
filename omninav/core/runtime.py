"""
SimulationRuntime — Core lifecycle orchestrator.

Manages the creation, initialization, and step cycle of all components:
1. SimulationManager → 2. Robot.spawn() → 3. Sensor.create() →
4. Robot.mount_sensors() → 5. Locomotion.bind_sensors() →
6. SimulationManager.build() → 7. Robot.post_build() → 8. HookManager.emit(POST_BUILD)
"""

from typing import List, Optional, Dict, Any, Tuple, TYPE_CHECKING
import logging
import numpy as np
from omegaconf import DictConfig

from omninav.core.hooks import HookManager, EventType
from omninav.core.types import Observation, Action, RobotState

if TYPE_CHECKING:
    from omninav.robots.base import RobotBase
    from omninav.locomotion.base import LocomotionControllerBase
    from omninav.algorithms.base import AlgorithmBase
    from omninav.evaluation.base import TaskBase, TaskResult

logger = logging.getLogger(__name__)


class SimulationRuntime:
    """
    Lifecycle orchestrator for OmniNav simulation.

    Owns the creation order and step() loop for all components.
    This class is the single source of truth for component wiring.

    Components are injected from outside (by OmniNavEnv or tests).
    """

    def __init__(self, cfg: DictConfig):
        """
        Initialize runtime.

        Args:
            cfg: Full configuration
        """
        self.cfg = cfg
        self.hooks = HookManager()

        # Components (injected externally)
        self.sim = None  # SimulationManager
        self.robots: List["RobotBase"] = []
        self.locomotions: List["LocomotionControllerBase"] = []
        self.algorithms: List["AlgorithmBase"] = []
        self.task: Optional["TaskBase"] = None
        self.sensors: Dict[str, Any] = {}

        # State
        self._step_count: int = 0
        self._sim_time: float = 0.0
        self._built: bool = False
        self._dt: float = cfg.get("simulation", {}).get("dt", 0.01)

    def build(self) -> None:
        """
        Build simulation after all components have been registered.

        Emits PRE_BUILD and POST_BUILD events.
        """
        self.hooks.emit(EventType.PRE_BUILD)

        if self.sim is not None:
            self.sim.build()

        # Post-build for all robots
        for robot in self.robots:
            if hasattr(robot, 'post_build'):
                robot.post_build()

        self._built = True
        self.hooks.emit(EventType.POST_BUILD)
        logger.info("SimulationRuntime: build complete")

    def reset(self) -> List[Observation]:
        """
        Reset all components and return initial observations.

        Returns:
            List of Observation (one per robot)
        """
        self._step_count = 0
        self._sim_time = 0.0

        self.hooks.emit(EventType.ON_RESET)

        # Reset simulation
        if self.sim is not None:
            self.sim.reset()

        # Reset robots
        for robot in self.robots:
            if hasattr(robot, 'reset'):
                robot.reset()

        # Reset locomotion
        for loco in self.locomotions:
            loco.reset()

        # Reset task
        task_info = {}
        if self.task is not None:
            task_info = self.task.reset()

        # Reset algorithms
        for algo in self.algorithms:
            algo.reset(task_info)

        return self._get_observations()

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
        self.hooks.emit(EventType.PRE_STEP, step=self._step_count)

        observations = self._get_observations()

        # Compute actions from algorithms if not provided
        if actions is None:
            actions = []
            for i, algo in enumerate(self.algorithms):
                if i < len(observations):
                    cmd_vel = algo.step(observations[i])
                    actions.append(Action(cmd_vel=cmd_vel))

        # Apply locomotion control
        for i, loco in enumerate(self.locomotions):
            if i < len(actions):
                cmd_vel = actions[i].get("cmd_vel", np.zeros(3))
                obs = observations[i] if i < len(observations) else None
                loco.step(cmd_vel, obs)

        # Physics step
        if self.sim is not None:
            self.sim.step()

        self._step_count += 1
        self._sim_time += self._dt

        # Get new observations
        new_observations = self._get_observations()

        # Update task
        if self.task is not None:
            for i, obs in enumerate(new_observations):
                action = actions[i] if i < len(actions) else None
                self.task.step(obs, action)

        self.hooks.emit(EventType.POST_STEP, step=self._step_count)

        info = {
            "step": self._step_count,
            "sim_time": self._sim_time,
        }

        return new_observations, info

    def _get_observations(self) -> List[Observation]:
        """Build observation list from all robots."""
        observations = []
        for robot in self.robots:
            obs = Observation(
                robot_state=robot.get_state(),
                sim_time=self._sim_time,
                sensors={},
            )
            # Collect sensor data
            if hasattr(robot, 'sensors'):
                for name, sensor in robot.sensors.items():
                    if hasattr(sensor, 'get_data'):
                        try:
                            obs["sensors"][name] = sensor.get_data()
                        except Exception:
                            pass

            observations.append(obs)
        return observations

    @property
    def is_done(self) -> bool:
        """Check if task is complete."""
        if self.task is not None and self.robots:
            obs = self._get_observations()
            if obs:
                return self.task.is_terminated(obs[0])
        for algo in self.algorithms:
            if algo.is_done:
                return True
        return False

    def get_result(self) -> Optional["TaskResult"]:
        """Get task result if task is set."""
        if self.task is not None:
            return self.task.compute_result()
        return None

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def sim_time(self) -> float:
        return self._sim_time
