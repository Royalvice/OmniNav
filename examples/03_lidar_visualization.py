#!/usr/bin/env python3
"""Demo 03: Go2w lidar visualization with keyboard teleop."""

from __future__ import annotations

import argparse
import math
import sys
from typing import Optional

import cv2
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


class _LidarWindow:
    def __init__(self, enable_window: bool, window_name: str = "Lidar View", size: int = 560):
        self.enable_window = bool(enable_window)
        self.window_name = window_name
        self.size = int(size)
        self.center = self.size // 2
        self.image = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        if self.enable_window:
            cv2.namedWindow(self.window_name)

    def _pick_lidar_data(self, obs: dict) -> Optional[dict]:
        sensors = obs.get("sensors", {}) if isinstance(obs, dict) else {}
        if not isinstance(sensors, dict):
            return None
        for value in sensors.values():
            if isinstance(value, dict) and "ranges" in value:
                return value
        return None

    def update(self, obs: dict) -> None:
        if not self.enable_window:
            return
        data = self._pick_lidar_data(obs)
        self.image.fill(20)

        # background grid circles
        for r in (60, 120, 180, 240):
            cv2.circle(self.image, (self.center, self.center), r, (55, 55, 55), 1)
        cv2.line(self.image, (self.center, 0), (self.center, self.size), (45, 45, 45), 1)
        cv2.line(self.image, (0, self.center), (self.size, self.center), (45, 45, 45), 1)

        if data is None:
            cv2.putText(self.image, "No lidar data", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 120, 255), 2)
            cv2.imshow(self.window_name, self.image)
            cv2.waitKey(1)
            return

        ranges = np.asarray(data.get("ranges", []), dtype=np.float32)
        if ranges.ndim == 2:
            ranges = ranges[0]
        if ranges.ndim != 1 or ranges.size == 0:
            cv2.putText(self.image, "Invalid lidar ranges", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 120, 255), 2)
            cv2.imshow(self.window_name, self.image)
            cv2.waitKey(1)
            return

        angle_min = float(data.get("angle_min", -math.pi))
        angle_increment = float(data.get("angle_increment", (2.0 * math.pi) / max(1, int(ranges.size))))
        max_range = float(np.nanmax(ranges)) if np.isfinite(np.nanmax(ranges)) else 1.0
        max_range = max(max_range, 1e-3)
        scale = (self.size * 0.46) / max_range

        valid = np.isfinite(ranges) & (ranges > 0.0)
        indices = np.nonzero(valid)[0]
        for i in indices:
            dist = float(ranges[i])
            a = angle_min + float(i) * angle_increment
            x = int(self.center + scale * dist * math.cos(a))
            y = int(self.center - scale * dist * math.sin(a))
            if 0 <= x < self.size and 0 <= y < self.size:
                self.image[y, x] = (0, 220, 255)

        cv2.circle(self.image, (self.center, self.center), 6, (0, 255, 0), -1)
        cv2.putText(
            self.image,
            f"rays={ranges.size} max={max_range:.2f}m",
            (16, self.size - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (220, 220, 220),
            1,
        )
        cv2.imshow(self.window_name, self.image)
        cv2.waitKey(1)

    def close(self) -> None:
        if self.enable_window:
            cv2.destroyWindow(self.window_name)


def _build_overrides(show_viewer: bool, fast_mode: bool) -> list[str]:
    overrides = [f"simulation.show_viewer={str(show_viewer)}"]
    if fast_mode:
        overrides.extend([
            "simulation.backend=cpu",
            "simulation.substeps=1",
            "simulation.dt=0.02",
        ])
    return overrides


def main() -> int:
    parser = argparse.ArgumentParser(description="Go2w lidar visualization demo")
    parser.add_argument("--test-mode", action="store_true")
    parser.add_argument("--smoke-fast", action="store_true", help="Use lighter simulation settings for smoke tests")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--show-viewer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-sensor-window", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.smoke_fast and args.max_steps > 30:
        args.max_steps = 30

    keyboard: Optional[_KeyboardInput] = None
    lidar_window: Optional[_LidarWindow] = None
    overrides = _build_overrides(args.show_viewer, args.test_mode or args.smoke_fast)

    with OmniNavEnv(config_path="configs", config_name="demo/lidar_visualization", overrides=overrides) as env:
        obs_list = env.reset()
        if not obs_list:
            return 1

        demo_cfg = env.cfg.get("demo", {})
        vx = float(demo_cfg.get("max_linear_vel", 0.5))
        vy = float(demo_cfg.get("max_lateral_vel", 0.0))
        wz = float(demo_cfg.get("max_angular_vel", 0.5))
        allow_lateral = bool(demo_cfg.get("allow_lateral", False))

        if not args.test_mode:
            keyboard = _KeyboardInput(allow_lateral=allow_lateral, linear_vel=vx, lateral_vel=vy, angular_vel=wz)
            keyboard.start()
        lidar_window = _LidarWindow(enable_window=(args.show_sensor_window and (not args.test_mode)))

        step = 0
        try:
            while True:
                if args.test_mode:
                    cmd = np.array([vx, 0.0, 0.0], dtype=np.float32) if step < args.max_steps // 2 else np.array([0.0, 0.0, wz], dtype=np.float32)
                else:
                    cmd = keyboard.read() if keyboard is not None else np.zeros(3, dtype=np.float32)
                    if keyboard is not None and not keyboard.running:
                        break

                action = Action(cmd_vel=np.asarray(cmd, dtype=np.float32).reshape(1, 3))
                obs_list, _info = env.step(actions=[action])
                if lidar_window is not None and obs_list:
                    lidar_window.update(obs_list[0])

                step += 1
                if env.is_done:
                    break
                if args.test_mode and step >= args.max_steps:
                    break
        finally:
            if keyboard is not None:
                keyboard.stop()
            if lidar_window is not None:
                lidar_window.close()
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
