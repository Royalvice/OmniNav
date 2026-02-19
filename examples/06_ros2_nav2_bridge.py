#!/usr/bin/env python3
"""Demo 06: ROS2/Nav2 bridge with OmniNav-owned bridge implementation."""

from __future__ import annotations

import argparse
import sys

from omninav.interfaces import OmniNavEnv


def _build_overrides(show_viewer: bool, test_mode: bool, smoke_fast: bool) -> list[str]:
    overrides = [f"simulation.show_viewer={str(show_viewer)}"]
    if test_mode or smoke_fast:
        overrides.extend(
            [
                "simulation.backend=cpu",
                "simulation.substeps=1",
                "simulation.dt=0.02",
            ]
        )
    if smoke_fast:
        # Keep script path covered while reducing dependency on ROS daemon state.
        overrides.extend(["ros2.enabled=false", "ros2.control_source=python"])
    return overrides


def main() -> int:
    parser = argparse.ArgumentParser(description="ROS2/Nav2 bridge demo")
    parser.add_argument("--test-mode", action="store_true")
    parser.add_argument("--smoke-fast", action="store_true", help="Use lighter simulation settings for smoke tests")
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--show-viewer", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    if args.smoke_fast and args.max_steps > 30:
        args.max_steps = 30

    overrides = _build_overrides(args.show_viewer, args.test_mode, args.smoke_fast)

    with OmniNavEnv(config_path="configs", config_name="demo/nav2_bridge", overrides=overrides) as env:
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
