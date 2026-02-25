"""Contract tests for getting_started override composition."""

from types import SimpleNamespace

from examples.getting_started.run_getting_started import GettingStartedApp


def _args():
    return SimpleNamespace(test_mode=True, smoke_fast=True, max_steps=10, show_viewer=False)


def test_getting_started_build_overrides_contains_task_and_pipeline():
    app = GettingStartedApp(_args())
    app.cfg.task = "waypoint"
    app.cfg.global_planner = "global_sequential"
    app.cfg.local_planner = "dwa_planner"
    ov = app._build_overrides()
    assert "task=waypoint" in ov
    assert "algorithm=pipeline_default" in ov
    assert "algorithm.global_planner.type=global_sequential" in ov
    assert "algorithm.local_planner.type=dwa_planner" in ov


def test_getting_started_sensor_profile_overrides():
    app = GettingStartedApp(_args())
    app.cfg.sensor_profile = "none"
    ov = app._build_overrides()
    assert "sensor.depth_camera=null" in ov
    assert "sensor.lidar_2d=null" in ov
