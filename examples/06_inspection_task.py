#!/usr/bin/env python3
"""Inspection task demo with config-driven module composition."""

from __future__ import annotations

import argparse
import logging
import sys

from omninav.interfaces import OmniNavEnv


def _build_overrides(show_viewer: bool, fast_mode: bool) -> list[str]:
    overrides = [f"simulation.show_viewer={str(show_viewer)}"]
    if fast_mode:
        overrides.extend(
            [
                "simulation.backend=cpu",
                "simulation.substeps=1",
                "simulation.dt=0.02",
                "task.time_budget=3.0",
            ]
        )
    return overrides


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspection pipeline demo")
    parser.add_argument("--test-mode", action="store_true")
    parser.add_argument("--smoke-fast", action="store_true", help="Use lighter simulation settings for smoke tests")
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--show-viewer", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.smoke_fast and args.max_steps > 30:
        args.max_steps = 30

    logging.basicConfig(level=logging.WARN)
    logging.getLogger("genesis").setLevel(logging.ERROR)

    overrides = _build_overrides(args.show_viewer, args.test_mode or args.smoke_fast)
    with OmniNavEnv(config_path="configs", config_name="demo/inspection", overrides=overrides) as env:
        obs_list = env.reset()
        if not obs_list:
            return 1

        while not env.is_done:
            _obs_list, info = env.step()
            if args.test_mode and info["step"] >= args.max_steps:
                break

        result = env.get_result()
        if result is None:
            return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
