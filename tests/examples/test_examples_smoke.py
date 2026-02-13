"""
Smoke tests for all scripts under examples/.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


EXAMPLES_DIR = Path("examples")
EXAMPLE_SCRIPTS = [
    "01_teleop_go2.py",
    "02_teleop_go2w.py",
    "03_lidar_visualization.py",
    "04_camera_visualization.py",
    "05_waypoint_navigation.py",
    "run_inspection.py",
]

EXAMPLE_TIMEOUT_SEC = {
    "05_waypoint_navigation.py": 600,
}


def _has_unhandled_traceback(output: str) -> bool:
    """
    Return True only for uncaught tracebacks.

    Python's logging module may print "Traceback (most recent call last)"
    blocks under "--- Logging error ---" while the process still exits
    successfully. Those are treated as non-fatal for smoke validation.
    """
    lines = output.splitlines()
    for idx, line in enumerate(lines):
        if "Traceback (most recent call last):" not in line:
            continue
        context = lines[max(0, idx - 3) : idx]
        if any("--- Logging error ---" in item for item in context):
            continue
        return True
    return False


@pytest.mark.integration
@pytest.mark.parametrize("script_name", EXAMPLE_SCRIPTS)
def test_example_script_smoke(script_name: str):
    """
    Every example script must start and exit cleanly in test mode.
    """
    script_path = EXAMPLES_DIR / script_name
    assert script_path.exists(), f"Missing example script: {script_path}"

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    repo_root = Path.cwd()
    pythonpath_items = [
        str(repo_root),
        str(repo_root / "external" / "Genesis"),
    ]
    if env.get("PYTHONPATH"):
        pythonpath_items.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_items)
    env["GS_ENABLE_FASTCACHE"] = "0"
    env["TI_OFFLINE_CACHE"] = "0"
    env["GS_ENABLE_NDARRAY"] = "0"
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    cmd = [
        sys.executable,
        str(script_path),
        "--test-mode",
        "--max-steps",
        "120",
        "--no-show-viewer",
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=EXAMPLE_TIMEOUT_SEC.get(script_name, 240),
        env=env,
    )

    combined = f"{result.stdout}\n{result.stderr}"
    assert result.returncode == 0, combined
    assert not _has_unhandled_traceback(combined), combined
