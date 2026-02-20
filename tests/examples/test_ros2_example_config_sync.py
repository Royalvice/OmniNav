"""Keep ROS2 package configs aligned with root configs."""

from __future__ import annotations

from pathlib import Path


def test_ros2_example_config_sync() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    package_cfg = repo_root / "examples" / "ros2" / "omninav_ros2_examples" / "configs"
    root_cfg = repo_root / "configs"

    mapping = [
        ("demo/ros2_rviz_sensor.yaml", "demo/ros2_rviz_sensor.yaml"),
        ("demo/ros2_nav2_full.yaml", "demo/ros2_nav2_full.yaml"),
        ("robot/go2w.yaml", "robot/go2w.yaml"),
        ("locomotion/wheel.yaml", "locomotion/wheel.yaml"),
        ("scene/complex_flat_obstacles.yaml", "scene/complex_flat_obstacles.yaml"),
        ("sensor/default.yaml", "sensor/default.yaml"),
    ]

    for pkg_rel, root_rel in mapping:
        pkg_file = package_cfg / pkg_rel
        root_file = root_cfg / root_rel
        assert pkg_file.exists(), f"Missing package config: {pkg_file}"
        assert root_file.exists(), f"Missing root config: {root_file}"
        assert pkg_file.read_text(encoding="utf-8") == root_file.read_text(encoding="utf-8"), (
            f"Config drift detected: {pkg_rel} != {root_rel}"
        )
