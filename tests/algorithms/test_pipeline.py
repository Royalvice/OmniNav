"""
Tests for Algorithm layer: Pipeline, DWA, InspectionPlanner.
"""

import pytest
import numpy as np
from omegaconf import OmegaConf

from omninav.core.types import Observation, RobotState


# =============================================================================
# Helpers
# =============================================================================

def _make_obs(pos=(0, 0, 0), quat=(1, 0, 0, 0), goal=None, lidar_ranges=None, sim_time=0.0):
    """Build a minimal Observation dict."""
    obs = Observation(
        robot_state=RobotState(
            position=np.array([pos], dtype=np.float32),
            orientation=np.array([quat], dtype=np.float32),
            linear_velocity=np.zeros((1, 3), dtype=np.float32),
            angular_velocity=np.zeros((1, 3), dtype=np.float32),
            joint_positions=np.zeros((1, 12), dtype=np.float32),
            joint_velocities=np.zeros((1, 12), dtype=np.float32),
        ),
        sim_time=sim_time,
        sensors={},
    )
    if goal is not None:
        obs["goal_position"] = np.array([goal], dtype=np.float32)
    if lidar_ranges is not None:
        obs["sensors"]["lidar"] = {"ranges": np.array([lidar_ranges], dtype=np.float32)}
    return obs


# =============================================================================
# InspectionPlanner Tests
# =============================================================================

class TestInspectionPlanner:
    """Test InspectionPlanner."""

    def test_registration(self):
        from omninav.core.registry import ALGORITHM_REGISTRY
        assert "inspection_planner" in ALGORITHM_REGISTRY

    def test_greedy_ordering(self):
        from omninav.algorithms.inspection_planner import InspectionPlanner

        cfg = OmegaConf.create({"type": "inspection_planner", "planning_strategy": "greedy"})
        planner = InspectionPlanner(cfg)

        waypoints = [[0, 0, 0], [10, 0, 0], [1, 0, 0]]  # nearest-neighbor: 0 → 2 → 1
        planner.reset({"waypoints": waypoints})

        assert planner._visit_order == [0, 2, 1]

    def test_tsp_2opt_ordering(self):
        from omninav.algorithms.inspection_planner import InspectionPlanner

        cfg = OmegaConf.create({"type": "inspection_planner", "planning_strategy": "tsp_2opt"})
        planner = InspectionPlanner(cfg)

        waypoints = [[0, 0, 0], [10, 0, 0], [5, 0, 0]]
        planner.reset({"waypoints": waypoints})

        # 2-opt should find optimal: 0 → 2 → 1 (total dist 10)
        total = 0
        for i in range(len(planner._visit_order) - 1):
            a = np.array(waypoints[planner._visit_order[i]])
            b = np.array(waypoints[planner._visit_order[i + 1]])
            total += np.linalg.norm(b - a)
        assert total <= 10.001

    def test_navigation_to_waypoint(self):
        from omninav.algorithms.inspection_planner import InspectionPlanner, InspectionState

        cfg = OmegaConf.create({
            "type": "inspection_planner",
            "scan_at_waypoint": False,
            "waypoint_tolerance": 0.5,
        })
        planner = InspectionPlanner(cfg)
        planner.reset({"waypoints": [[1, 0, 0]]})

        assert planner.state == InspectionState.NAVIGATING
        assert planner.current_waypoint is not None

        # Simulate arriving at waypoint
        obs = _make_obs(pos=(1.0, 0.0, 0.0))
        cmd = planner.step(obs)

        assert planner.is_done
        assert np.allclose(cmd, 0.0)

    def test_scanning_at_waypoint(self):
        from omninav.algorithms.inspection_planner import InspectionPlanner, InspectionState

        cfg = OmegaConf.create({
            "type": "inspection_planner",
            "scan_at_waypoint": True,
            "scan_duration": 2.0,
            "scan_angular_velocity": 1.0,
            "waypoint_tolerance": 0.5,
        })
        planner = InspectionPlanner(cfg)
        planner.reset({"waypoints": [[0, 0, 0]]})

        # Already at waypoint → first step transitions to SCANNING
        obs = _make_obs(pos=(0, 0, 0), sim_time=0.0)
        planner.step(obs)

        assert planner.state == InspectionState.SCANNING

        # Second step → returns rotation command
        cmd = planner.step(obs)
        assert cmd[2] == pytest.approx(1.0)  # wz = scan_angular_velocity

    def test_dynamic_waypoint_insertion(self):
        from omninav.algorithms.inspection_planner import InspectionPlanner

        cfg = OmegaConf.create({"type": "inspection_planner", "scan_at_waypoint": False})
        planner = InspectionPlanner(cfg)
        planner.reset({"waypoints": [[1, 0, 0]]})

        planner.add_waypoint(np.array([2, 0, 0]))
        assert len(planner._waypoints) == 2
        assert len(planner._visit_order) == 2

    def test_progress_tracking(self):
        from omninav.algorithms.inspection_planner import InspectionPlanner

        cfg = OmegaConf.create({"type": "inspection_planner", "scan_at_waypoint": False, "waypoint_tolerance": 0.5})
        planner = InspectionPlanner(cfg)
        planner.reset({"waypoints": [[1, 0, 0], [2, 0, 0]]})

        assert planner.progress == 0.0

        # Visit first waypoint
        obs = _make_obs(pos=(1, 0, 0))
        planner.step(obs)

        assert planner.progress == 0.5

    def test_info_dict(self):
        from omninav.algorithms.inspection_planner import InspectionPlanner

        cfg = OmegaConf.create({"type": "inspection_planner"})
        planner = InspectionPlanner(cfg)
        planner.reset({"waypoints": [[1, 0, 0]]})

        info = planner.info
        assert "state" in info
        assert "progress" in info
        assert "total_waypoints" in info


# =============================================================================
# DWAPlanner Tests
# =============================================================================

class TestDWAPlanner:
    """Test DWA local planner."""

    def test_registration(self):
        from omninav.core.registry import ALGORITHM_REGISTRY
        assert "dwa_planner" in ALGORITHM_REGISTRY

    def test_zero_at_goal(self):
        from omninav.algorithms.local_planner import DWAPlanner

        cfg = OmegaConf.create({
            "type": "dwa_planner",
            "goal_tolerance": 0.5,
        })
        planner = DWAPlanner(cfg)
        planner.reset()

        # Robot at origin, goal at origin
        obs = _make_obs(pos=(0, 0, 0), goal=(0, 0, 0))
        cmd = planner.step(obs)

        assert np.allclose(cmd, 0.0)
        assert planner.is_done

    def test_forward_toward_goal(self):
        from omninav.algorithms.local_planner import DWAPlanner

        cfg = OmegaConf.create({
            "type": "dwa_planner",
            "max_speed": 1.0,
            "goal_tolerance": 0.5,
        })
        planner = DWAPlanner(cfg)
        planner.reset()

        # Robot at origin, goal ahead at (5, 0, 0)
        obs = _make_obs(pos=(0, 0, 0), goal=(5, 0, 0))
        cmd = planner.step(obs)

        # Should command positive vx
        assert cmd[0] > 0

    def test_navigate_to_interface(self):
        from omninav.algorithms.local_planner import DWAPlanner

        cfg = OmegaConf.create({"type": "dwa_planner"})
        planner = DWAPlanner(cfg)
        planner.reset()

        obs = _make_obs(pos=(0, 0, 0))
        target = np.array([5.0, 0.0, 0.0])
        cmd = planner.navigate_to(obs, target)

        assert cmd.shape == (3,)
        assert cmd[0] > 0


# =============================================================================
# AlgorithmPipeline Tests
# =============================================================================

class TestAlgorithmPipeline:
    """Test AlgorithmPipeline composition."""

    def test_registration(self):
        from omninav.core.registry import ALGORITHM_REGISTRY
        assert "algorithm_pipeline" in ALGORITHM_REGISTRY

    def test_pipeline_composes(self):
        from omninav.algorithms.pipeline import AlgorithmPipeline

        cfg = OmegaConf.create({
            "type": "algorithm_pipeline",
            "global_planner": {
                "type": "inspection_planner",
                "scan_at_waypoint": False,
                "waypoint_tolerance": 0.5,
            },
            "local_planner": {
                "type": "dwa_planner",
                "max_speed": 0.5,
                "goal_tolerance": 0.5,
            },
        })

        pipeline = AlgorithmPipeline(cfg)
        pipeline.reset({"waypoints": [[5, 0, 0]]})

        obs = _make_obs(pos=(0, 0, 0))
        cmd = pipeline.step(obs)

        assert cmd.shape == (3,)
        # Should navigate forward toward (5, 0, 0)
        assert cmd[0] > 0

    def test_pipeline_is_done_delegates(self):
        from omninav.algorithms.pipeline import AlgorithmPipeline

        cfg = OmegaConf.create({
            "type": "algorithm_pipeline",
            "global_planner": {
                "type": "inspection_planner",
                "scan_at_waypoint": False,
                "waypoint_tolerance": 0.5,
            },
            "local_planner": {
                "type": "dwa_planner",
                "goal_tolerance": 0.5,
            },
        })

        pipeline = AlgorithmPipeline(cfg)
        pipeline.reset({"waypoints": [[0, 0, 0]]})

        # Already at goal
        obs = _make_obs(pos=(0, 0, 0))
        pipeline.step(obs)

        assert pipeline.is_done
