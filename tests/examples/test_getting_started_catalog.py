"""Tests for getting_started catalog helpers."""

from examples.getting_started.catalog import list_genesis_candidates, list_omninav_ready


def test_catalog_lists_ready_components():
    ready = list_omninav_ready()
    assert "unitree_go2" in ready["robots"]
    assert "unitree_go2w" in ready["robots"]
    assert "global_sequential" in ready["algorithms"]
    assert "dwa_planner" in ready["algorithms"]
    assert "inspection" in ready["tasks"]
    assert "waypoint" in ready["tasks"]
    assert "waypoint_arena" in ready["scenes"]


def test_catalog_lists_genesis_candidates():
    candidates = list_genesis_candidates(max_items=20)
    assert "urdf_candidates" in candidates
    assert "mjcf_candidates" in candidates
    assert len(candidates["sensor_capabilities"]) >= 3
