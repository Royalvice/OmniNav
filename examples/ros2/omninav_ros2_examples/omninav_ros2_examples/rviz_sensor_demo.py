#!/usr/bin/env python3
"""ROS2 Demo A: RViz sensor visualization + viewer keyboard teleop."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from omninav.core.types import Action
from omninav.interfaces import OmniNavEnv


class _KeyboardInput:
    def __init__(self, allow_lateral: bool, linear_vel: float, lateral_vel: float, angular_vel: float):
        self.allow_lateral = allow_lateral
        self.linear_vel = linear_vel
        self.lateral_vel = lateral_vel
        self.angular_vel = angular_vel
        self.running = True
        self._cmd = np.zeros(3, dtype=np.float32)
        self._key_state: dict[str, bool] = {}
        self._listener = None
        self._keyboard = None

        try:
            from pynput import keyboard

            self._keyboard = keyboard
        except Exception as exc:
            raise RuntimeError("pynput is required for interactive teleop") from exc

    def start(self) -> None:
        def on_press(key):
            try:
                self._key_state[key.char.lower()] = True
            except Exception:
                if key == self._keyboard.Key.space:
                    self._key_state["space"] = True
                elif key == self._keyboard.Key.esc:
                    self._key_state["escape"] = True

        def on_release(key):
            try:
                self._key_state[key.char.lower()] = False
            except Exception:
                if key == self._keyboard.Key.space:
                    self._key_state["space"] = False
                elif key == self._keyboard.Key.esc:
                    self._key_state["escape"] = False

        self._listener = self._keyboard.Listener(on_press=on_press, on_release=on_release)
        self._listener.start()

    def read(self) -> np.ndarray:
        if self._key_state.get("escape", False):
            self.running = False
            return self._cmd

        if self._key_state.get("space", False):
            self._cmd.fill(0.0)
            return self._cmd

        self._cmd[0] = self.linear_vel if self._key_state.get("w", False) else (-self.linear_vel if self._key_state.get("s", False) else 0.0)
        self._cmd[1] = (
            self.lateral_vel if self._key_state.get("a", False) else (-self.lateral_vel if self._key_state.get("d", False) else 0.0)
        ) if self.allow_lateral else 0.0
        self._cmd[2] = self.angular_vel if self._key_state.get("q", False) else (-self.angular_vel if self._key_state.get("e", False) else 0.0)
        return self._cmd

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None


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


def _build_overrides(show_viewer: bool, fast_mode: bool, smoke_fast: bool) -> list[str]:
    overrides = [f"simulation.show_viewer={str(show_viewer)}"]
    if fast_mode:
        overrides.extend([
            "simulation.backend=cpu",
            "simulation.substeps=1",
            "simulation.dt=0.02",
        ])
    if smoke_fast:
        overrides.extend(["ros2.enabled=false", "ros2.control_source=python"])
    return overrides


def _scripted_cmd(step: int, max_steps: int, vx: float, wz: float) -> np.ndarray:
    cmd = np.zeros(3, dtype=np.float32)
    phase = max_steps // 3
    if step < phase:
        cmd[0] = vx
    elif step < 2 * phase:
        cmd[2] = wz
    else:
        cmd[0] = -vx
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description="OmniNav ROS2 RViz sensor demo")
    parser.add_argument("--test-mode", action="store_true")
    parser.add_argument("--smoke-fast", action="store_true")
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--show-viewer", action=argparse.BooleanOptionalAction, default=True)
    args, _unknown = parser.parse_known_args()

    if args.smoke_fast and args.max_steps > 30:
        args.max_steps = 30

    keyboard: Optional[_KeyboardInput] = None
    overrides = _build_overrides(args.show_viewer, args.test_mode or args.smoke_fast, args.smoke_fast)
    config_dir = _resolve_config_dir()

    with OmniNavEnv(config_path=config_dir, config_name="demo/ros2_rviz_sensor", overrides=overrides) as env:
        obs_list = env.reset()
        if not obs_list:
            return 1

        demo_cfg = env.cfg.get("demo", {})
        vx = float(demo_cfg.get("max_linear_vel", 0.6))
        vy = float(demo_cfg.get("max_lateral_vel", 0.0))
        wz = float(demo_cfg.get("max_angular_vel", 0.8))
        allow_lateral = bool(demo_cfg.get("allow_lateral", False))

        if not args.test_mode:
            keyboard = _KeyboardInput(allow_lateral=allow_lateral, linear_vel=vx, lateral_vel=vy, angular_vel=wz)
            keyboard.start()

        step = 0
        try:
            while True:
                if args.test_mode:
                    cmd = _scripted_cmd(step, args.max_steps, vx, wz)
                else:
                    cmd = keyboard.read() if keyboard is not None else np.zeros(3, dtype=np.float32)
                    if keyboard is not None and not keyboard.running:
                        break

                action = Action(cmd_vel=np.asarray(cmd, dtype=np.float32).reshape(1, 3))
                _obs_list, _info = env.step(actions=[action])

                step += 1
                if env.is_done:
                    break
                if args.test_mode and step >= args.max_steps:
                    break
        finally:
            if keyboard is not None:
                keyboard.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())
