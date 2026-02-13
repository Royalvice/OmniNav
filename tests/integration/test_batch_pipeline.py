"""
Integration tests for Batch-First runtime semantics.
"""

from __future__ import annotations

import os
import numpy as np
import pytest

from omninav.interfaces import OmniNavEnv


def _genesis_available() -> bool:
    if os.environ.get("OMNINAV_RUN_GENESIS_TESTS", "0") != "1":
        return False
    try:
        import genesis  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.integration
def test_inspection_pipeline_n_envs_4():
    """The full pipeline should run with n_envs=4 and keep batch shapes."""
    if not _genesis_available():
        pytest.skip("genesis not installed")
    if not os.path.exists("configs"):
        pytest.skip("Config directory not found")

    overrides = [
        "simulation.show_viewer=False",
        "simulation.backend=cpu",
        "simulation.n_envs=4",
        "task=inspection",
        "algorithm=inspection",
        "locomotion=kinematic_gait",
        "task.time_budget=3.0",
    ]

    env = OmniNavEnv.from_config(config_path="configs", overrides=overrides)
    obs_list = env.reset()
    assert len(obs_list) == 1

    pos = obs_list[0]["robot_state"]["position"]
    assert pos.ndim == 2
    assert pos.shape[0] == 4
    assert pos.shape[1] == 3

    for _ in range(5):
        obs_list, info = env.step()
        assert "done_mask" in info
        done_mask = info["done_mask"]
        if done_mask is not None:
            done_mask = np.asarray(done_mask)
            assert done_mask.shape == (4,)

    result = env.get_result()
    assert result is not None
