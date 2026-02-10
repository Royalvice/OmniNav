"""
Inspection-specific evaluation metrics.

- CoverageRate: Fraction of waypoints visited
- DetectionRate: Placeholder for anomaly detection accuracy
- InspectionTime: Normalized time efficiency
- SafetyScore: Safety metric based on collisions and near-misses
"""

from typing import Optional, TYPE_CHECKING
import numpy as np

from omninav.evaluation.base import MetricBase
from omninav.core.registry import METRIC_REGISTRY

if TYPE_CHECKING:
    from omninav.core.types import Observation, Action


@METRIC_REGISTRY.register("coverage_rate")
class CoverageRate(MetricBase):
    """
    Fraction of designated inspection waypoints successfully visited.

    Value range: [0.0, 1.0]
    """

    METRIC_NAME = "coverage_rate"

    def __init__(self):
        super().__init__()
        self._waypoints = []
        self._visited = []
        self._tolerance = 0.5

    def reset(self) -> None:
        self._visited = [False] * len(self._waypoints)

    def configure(self, waypoints, tolerance=0.5):
        """Configure with waypoint list."""
        self._waypoints = [np.array(wp, dtype=np.float32) for wp in waypoints]
        self._tolerance = tolerance
        self._visited = [False] * len(self._waypoints)

    def update(self, obs: "Observation", action: Optional["Action"] = None, **kwargs) -> None:
        robot_state = obs.get("robot_state", {})
        pos = np.asarray(robot_state.get("position", np.zeros((1, 3))))
        if pos.ndim == 2:
            pos = pos[0]

        for i, wp in enumerate(self._waypoints):
            if not self._visited[i]:
                if np.linalg.norm(wp[:2] - pos[:2]) < self._tolerance:
                    self._visited[i] = True

    def compute(self) -> float:
        if not self._waypoints:
            return 1.0
        return sum(self._visited) / len(self._waypoints)


@METRIC_REGISTRY.register("detection_rate")
class DetectionRate(MetricBase):
    """
    Anomaly detection accuracy metric.

    Tracks true positives, false positives, false negatives.
    Value: F1 score [0.0, 1.0]
    """

    METRIC_NAME = "detection_rate"

    def __init__(self):
        super().__init__()
        self._true_positives = 0
        self._false_positives = 0
        self._false_negatives = 0

    def reset(self) -> None:
        self._true_positives = 0
        self._false_positives = 0
        self._false_negatives = 0

    def record_detection(self, is_true_positive: bool, is_false_positive: bool = False):
        """Record a detection event."""
        if is_true_positive:
            self._true_positives += 1
        if is_false_positive:
            self._false_positives += 1

    def record_miss(self):
        """Record a missed anomaly (false negative)."""
        self._false_negatives += 1

    def update(self, obs: "Observation", action: Optional["Action"] = None, **kwargs) -> None:
        # Detection events are recorded externally via record_detection/record_miss
        pass

    def compute(self) -> float:
        """Compute F1 score."""
        precision_denom = self._true_positives + self._false_positives
        recall_denom = self._true_positives + self._false_negatives

        if precision_denom == 0 or recall_denom == 0:
            return 0.0

        precision = self._true_positives / precision_denom
        recall = self._true_positives / recall_denom

        if precision + recall == 0:
            return 0.0

        return 2 * precision * recall / (precision + recall)


@METRIC_REGISTRY.register("inspection_time")
class InspectionTime(MetricBase):
    """
    Time efficiency metric.

    Computes: time_budget / actual_time (capped at 1.0)
    Higher is better.
    """

    METRIC_NAME = "inspection_time"

    def __init__(self):
        super().__init__()
        self._start_time = 0.0
        self._end_time = 0.0
        self._time_budget = 120.0

    def reset(self) -> None:
        self._start_time = 0.0
        self._end_time = 0.0

    def configure(self, time_budget: float):
        """Set the time budget."""
        self._time_budget = time_budget

    def update(self, obs: "Observation", action: Optional["Action"] = None, **kwargs) -> None:
        sim_time = obs.get("sim_time", 0.0)
        if self._start_time == 0.0:
            self._start_time = sim_time
        self._end_time = sim_time

    def compute(self) -> float:
        elapsed = max(self._end_time - self._start_time, 1e-6)
        return min(self._time_budget / elapsed, 1.0)


@METRIC_REGISTRY.register("safety_score")
class SafetyScore(MetricBase):
    """
    Safety metric based on collision events.

    Computes: max(0, 1 - collision_count * penalty_per_collision)
    """

    METRIC_NAME = "safety_score"

    def __init__(self):
        super().__init__()
        self._collision_count = 0
        self._penalty_per_collision = 0.1
        self._near_miss_count = 0
        self._near_miss_penalty = 0.02
        self._min_clearance = float('inf')

    def reset(self) -> None:
        self._collision_count = 0
        self._near_miss_count = 0
        self._min_clearance = float('inf')

    def record_collision(self):
        """Record a collision event."""
        self._collision_count += 1

    def update(self, obs: "Observation", action: Optional["Action"] = None, **kwargs) -> None:
        # Check lidar for near-misses
        sensors = obs.get("sensors", {})
        for sensor_data in sensors.values():
            if "ranges" in sensor_data:
                ranges = np.asarray(sensor_data["ranges"])
                if ranges.ndim == 2:
                    ranges = ranges[0]
                valid = ranges[ranges > 0.01]
                if len(valid) > 0:
                    min_r = float(np.min(valid))
                    self._min_clearance = min(self._min_clearance, min_r)
                    if min_r < 0.2:  # near-miss threshold
                        self._near_miss_count += 1

    def compute(self) -> float:
        penalty = (
            self._collision_count * self._penalty_per_collision +
            self._near_miss_count * self._near_miss_penalty
        )
        return max(0.0, 1.0 - penalty)
