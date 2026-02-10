"""
Tests for Evaluation layer: InspectionTask, CoverageRate, DetectionRate,
InspectionTime, SafetyScore.
"""

import pytest
import numpy as np
from omegaconf import OmegaConf

from omninav.core.types import Observation, RobotState


# =============================================================================
# Helpers
# =============================================================================

def _make_obs(pos=(0, 0, 0), sim_time=0.0, lidar_ranges=None):
    obs = Observation(
        robot_state=RobotState(
            position=np.array([pos], dtype=np.float32),
            orientation=np.array([[1, 0, 0, 0]], dtype=np.float32),
            linear_velocity=np.zeros((1, 3), dtype=np.float32),
            angular_velocity=np.zeros((1, 3), dtype=np.float32),
            joint_positions=np.zeros((1, 12), dtype=np.float32),
            joint_velocities=np.zeros((1, 12), dtype=np.float32),
        ),
        sim_time=sim_time,
        sensors={},
    )
    if lidar_ranges is not None:
        obs["sensors"]["lidar"] = {"ranges": np.array([lidar_ranges], dtype=np.float32)}
    return obs


# =============================================================================
# InspectionTask Tests
# =============================================================================

class TestInspectionTask:

    def test_registration(self):
        from omninav.core.registry import TASK_REGISTRY
        assert "inspection" in TASK_REGISTRY

    def test_reset_returns_task_info(self):
        from omninav.evaluation.tasks.inspection_task import InspectionTask

        cfg = OmegaConf.create({
            "type": "inspection",
            "waypoints": [[1, 0, 0], [3, 0, 0]],
            "time_budget": 60.0,
            "coverage_requirement": 0.8,
        })
        task = InspectionTask(cfg)
        info = task.reset()

        assert "waypoints" in info
        assert len(info["waypoints"]) == 2
        assert info["time_budget"] == 60.0

    def test_waypoint_visit_tracking(self):
        from omninav.evaluation.tasks.inspection_task import InspectionTask

        cfg = OmegaConf.create({
            "type": "inspection",
            "waypoints": [[1, 0, 0]],
            "waypoint_tolerance": 0.5,
            "time_budget": 120.0,
        })
        task = InspectionTask(cfg)
        task.reset()

        # Not at waypoint yet
        task.step(_make_obs(pos=(0, 0, 0)))
        assert task.coverage == 0.0

        # At waypoint
        task.step(_make_obs(pos=(1, 0, 0)))
        assert task.coverage == 1.0

    def test_termination_on_coverage(self):
        from omninav.evaluation.tasks.inspection_task import InspectionTask

        cfg = OmegaConf.create({
            "type": "inspection",
            "waypoints": [[0, 0, 0]],
            "waypoint_tolerance": 0.5,
            "time_budget": 120.0,
        })
        task = InspectionTask(cfg)
        task.reset()

        obs = _make_obs(pos=(0, 0, 0))
        task.step(obs)
        assert task.is_terminated(obs)

    def test_termination_on_timeout(self):
        from omninav.evaluation.tasks.inspection_task import InspectionTask

        cfg = OmegaConf.create({
            "type": "inspection",
            "waypoints": [[100, 0, 0]],
            "time_budget": 10.0,
        })
        task = InspectionTask(cfg)
        task.reset()

        obs = _make_obs(pos=(0, 0, 0), sim_time=15.0)
        assert task.is_terminated(obs)

    def test_compute_result(self):
        from omninav.evaluation.tasks.inspection_task import InspectionTask

        cfg = OmegaConf.create({
            "type": "inspection",
            "waypoints": [[1, 0, 0], [3, 0, 0]],
            "time_budget": 120.0,
            "coverage_requirement": 0.5,
            "waypoint_tolerance": 0.5,
        })
        task = InspectionTask(cfg)
        task.reset()

        task.step(_make_obs(pos=(1, 0, 0), sim_time=1.0))
        task.step(_make_obs(pos=(2, 0, 0), sim_time=2.0))

        result = task.compute_result()
        assert "coverage_rate" in result.metrics
        assert result.metrics["coverage_rate"] == 0.5
        assert result.success  # 0.5 >= coverage_requirement 0.5


# =============================================================================
# Metric Tests
# =============================================================================

class TestCoverageRate:

    def test_full_coverage(self):
        from omninav.evaluation.metrics.inspection_metrics import CoverageRate

        m = CoverageRate()
        m.configure([[0, 0, 0], [1, 0, 0]], tolerance=0.5)
        m.update(_make_obs(pos=(0, 0, 0)))
        m.update(_make_obs(pos=(1, 0, 0)))
        assert m.compute() == 1.0

    def test_partial_coverage(self):
        from omninav.evaluation.metrics.inspection_metrics import CoverageRate

        m = CoverageRate()
        m.configure([[0, 0, 0], [10, 0, 0]], tolerance=0.5)
        m.update(_make_obs(pos=(0, 0, 0)))
        assert m.compute() == 0.5


class TestDetectionRate:

    def test_perfect_detection(self):
        from omninav.evaluation.metrics.inspection_metrics import DetectionRate

        m = DetectionRate()
        m.reset()
        m.record_detection(is_true_positive=True)
        m.record_detection(is_true_positive=True)
        assert m.compute() == 1.0

    def test_with_false_positives(self):
        from omninav.evaluation.metrics.inspection_metrics import DetectionRate

        m = DetectionRate()
        m.reset()
        m.record_detection(is_true_positive=True)
        m.record_detection(is_true_positive=False, is_false_positive=True)
        # precision = 1/2, recall = 1/1, F1 = 2 * 0.5 * 1 / 1.5 = 0.667
        assert m.compute() == pytest.approx(2/3, abs=0.01)


class TestInspectionTime:

    def test_efficient_completion(self):
        from omninav.evaluation.metrics.inspection_metrics import InspectionTime

        m = InspectionTime()
        m.configure(time_budget=100.0)
        m.reset()
        m.update(_make_obs(sim_time=0.0))
        m.update(_make_obs(sim_time=50.0))
        # efficiency = 100/50 = 2.0 → capped at 1.0
        assert m.compute() == 1.0


class TestSafetyScore:

    def test_no_collisions(self):
        from omninav.evaluation.metrics.inspection_metrics import SafetyScore

        m = SafetyScore()
        m.reset()
        assert m.compute() == 1.0

    def test_with_collisions(self):
        from omninav.evaluation.metrics.inspection_metrics import SafetyScore

        m = SafetyScore()
        m.reset()
        m.record_collision()
        m.record_collision()
        # 1.0 - 2 * 0.1 = 0.8
        assert m.compute() == pytest.approx(0.8)
