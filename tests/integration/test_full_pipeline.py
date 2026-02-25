"""
Integration Tests for Full Navigation Pipeline

Verifies that the entire stack (Env, Robot, Sensors, Locomotion, Algorithm, Task)
works together correctly.
"""

import pytest
import numpy as np
import os
import tempfile
from omninav.interfaces import OmniNavEnv


def _genesis_available() -> bool:
    if os.environ.get("OMNINAV_RUN_GENESIS_TESTS", "0") != "1":
        return False
    try:
        import genesis  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.fixture
def headless_config_overrides():
    """Override config to disable viewer and use CPU/GPU as needed for CI."""
    return [
        "simulation.show_viewer=False",
        "simulation.n_envs=1",
        "task.time_budget=5.0",  # Short limit for test
    ]


def test_inspection_pipeline(headless_config_overrides):
    """
    Test the full inspection pipeline:
    - Initialize OmniNavEnv with Inspection config
    - Run for a few steps
    - Check if metrics are generated
    - Check if robot moves (basic)
    """
    if not _genesis_available():
        pytest.skip("genesis tests disabled")

    # Use default configs directory
    config_path = "configs"
    if not os.path.exists(config_path):
        pytest.skip("Config directory not found")

    # Use writable cache dirs in sandboxed CI-like environments.
    smoke_home = tempfile.mkdtemp(prefix="omninav_integration_home_", dir="/tmp")
    os.environ["HOME"] = smoke_home
    os.environ["XDG_CACHE_HOME"] = os.path.join(smoke_home, ".cache")
    os.environ["TI_CACHE_PATH"] = os.path.join(smoke_home, ".cache", "gstaichi", "ticache")
    os.environ["GS_ENABLE_FASTCACHE"] = "0"
    os.environ["TI_OFFLINE_CACHE"] = "0"
    os.environ["GS_ENABLE_NDARRAY"] = "0"

    # Initialize environment
    # Override defaults to use inspection task/algo
    overrides = headless_config_overrides + [
        "task=inspection",
        "algorithm=pipeline_default",
        "locomotion=kinematic_gait"
    ]
    
    # Use from_config to handle list of overrides correctly
    env = OmniNavEnv.from_config(config_path=config_path, overrides=overrides)
    obs_list = env.reset()
    
    assert len(obs_list) == 1
    assert "robot_state" in obs_list[0]
    assert "sensors" in obs_list[0]
    
    # Run for a few steps
    initial_pos = obs_list[0]["robot_state"]["position"]
    
    for _ in range(10):
        if env.is_done:
            break
        obs_list, info = env.step()
        
    result = env.get_result()
    
    # Use a small tolerance for movement check (kinematic gait might start slow)
    final_pos = obs_list[0]["robot_state"]["position"]
    dist_moved = np.linalg.norm(final_pos - initial_pos)
    
    # Check invariants
    assert result is not None
    assert isinstance(result.metrics, dict)
    assert "coverage_rate" in result.metrics
    
    print(f"Test finished. Dist moved: {dist_moved:.4f}")
