#!/usr/bin/env python3
"""ROS2 Demo B: OmniNav <-> Nav2 cmd_vel closed loop."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from omninav.interfaces import OmniNavEnv


def _resolve_config_dir() -> str:
    explicit = os.environ.get("OMNINAV_CONFIG_DIR", "")
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if path.exists():
            return str(path)

    repo_cfg = Path(__file__).resolve().parents[4] / "configs"
    if repo_cfg.exists():
        return str(repo_cfg)

    try:
        from ament_index_python.packages import get_package_share_directory

        pkg_share = Path(get_package_share_directory("omninav_ros2_examples"))
        installed_cfg = pkg_share / "omninav_configs"
        if installed_cfg.exists():
            return str(installed_cfg)
    except Exception:
        pass

    raise RuntimeError("Unable to locate OmniNav config directory. Set OMNINAV_CONFIG_DIR to <repo>/configs")


def _build_overrides(show_viewer: bool, test_mode: bool, smoke_fast: bool) -> list[str]:
    overrides = [f"simulation.show_viewer={str(show_viewer)}"]
    if test_mode or smoke_fast:
        overrides.extend([
            "simulation.backend=cpu",
            "simulation.substeps=1",
            "simulation.dt=0.02",
        ])
    if smoke_fast:
        overrides.extend(["ros2.enabled=false", "ros2.control_source=python"])
    return overrides


def main() -> int:
    parser = argparse.ArgumentParser(description="OmniNav ROS2/Nav2 bridge demo")
    parser.add_argument("--test-mode", action="store_true")
    parser.add_argument("--smoke-fast", action="store_true")
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--show-viewer", action=argparse.BooleanOptionalAction, default=False)
    args, _unknown = parser.parse_known_args()

    if args.smoke_fast and args.max_steps > 30:
        args.max_steps = 30

    overrides = _build_overrides(args.show_viewer, args.test_mode, args.smoke_fast)
    config_dir = _resolve_config_dir()

    with OmniNavEnv(config_path=config_dir, config_name="demo/ros2_nav2_full", overrides=overrides) as env:
        obs_list = env.reset()
        if not obs_list:
            return 1

        while True:
            _obs_list, info = env.step(actions=None)
            if env.is_done:
                break
            if args.test_mode and info["step"] >= args.max_steps:
                break

    return 0


if __name__ == "__main__":
    sys.exit(main())
