"""Focused smoke for getting_started GUI script."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path


def test_getting_started_smoke():
    script = Path("examples/getting_started/run_getting_started.py")
    assert script.exists()

    env = os.environ.copy()
    repo_root = Path.cwd()
    env["PYTHONPATH"] = os.pathsep.join([str(repo_root), str(repo_root / "external" / "Genesis")])
    smoke_home = Path(tempfile.mkdtemp(prefix="omninav_getting_started_home_", dir="/tmp"))
    env["HOME"] = str(smoke_home)
    env["XDG_CACHE_HOME"] = str(smoke_home / ".cache")
    env["TI_CACHE_PATH"] = str(smoke_home / ".cache" / "gstaichi" / "ticache")
    env["GS_ENABLE_FASTCACHE"] = "0"
    env["TI_OFFLINE_CACHE"] = "0"
    env["GS_ENABLE_NDARRAY"] = "0"
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    cmd = [
        sys.executable,
        str(script),
        "--test-mode",
        "--smoke-fast",
        "--max-steps",
        "20",
        "--no-show-viewer",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180, env=env)
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
