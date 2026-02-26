#!/usr/bin/env python3
"""Demo 04: Go2w camera visualization with keyboard teleop."""

from __future__ import annotations

import argparse
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


class _CameraWindow:
    def __init__(self, enable_window: bool, window_name: str = "Camera View"):
        self.enable_window = bool(enable_window)
        self.window_name = window_name
        if self.enable_window:
            cv2.namedWindow(self.window_name)

    def _pick_camera_data(self, obs: dict) -> Optional[dict]:
        sensors = obs.get("sensors", {}) if isinstance(obs, dict) else {}
        if not isinstance(sensors, dict):
            return None
        for value in sensors.values():
            if not isinstance(value, dict):
                continue
            if ("rgb" in value) or ("depth" in value):
                return value
        return None

    @staticmethod
    def _depth_to_bgr(depth: np.ndarray) -> np.ndarray:
        arr = np.asarray(depth, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[0]
        finite = np.isfinite(arr)
        if not np.any(finite):
            return np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)
        dmin = float(np.nanmin(arr[finite]))
        dmax = float(np.nanmax(arr[finite]))
        if dmax - dmin < 1e-6:
            norm = np.zeros_like(arr, dtype=np.uint8)
        else:
            norm = np.clip((arr - dmin) / (dmax - dmin), 0.0, 1.0)
            norm = (norm * 255.0).astype(np.uint8)
        return cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)

    def update(self, obs: dict) -> None:
        if not self.enable_window:
            return
        data = self._pick_camera_data(obs)
        if data is None:
            frame = np.zeros((320, 480, 3), dtype=np.uint8)
            cv2.putText(frame, "No camera data", (24, 46), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 140, 255), 2)
            cv2.imshow(self.window_name, frame)
            cv2.waitKey(1)
            return

        panes: list[np.ndarray] = []
        if "rgb" in data:
            rgb = np.asarray(data["rgb"])
            if rgb.ndim == 4:
                rgb = rgb[0]
            if rgb.ndim == 3 and rgb.shape[2] == 3:
                panes.append(cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR))
        if "depth" in data:
            panes.append(self._depth_to_bgr(np.asarray(data["depth"])))

        if not panes:
            frame = np.zeros((320, 480, 3), dtype=np.uint8)
            cv2.putText(frame, "Camera output empty", (24, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 140, 255), 2)
            cv2.imshow(self.window_name, frame)
            cv2.waitKey(1)
            return

        if len(panes) == 1:
            frame = panes[0]
        else:
            h = min(panes[0].shape[0], panes[1].shape[0])
            w0 = int(panes[0].shape[1] * (h / panes[0].shape[0]))
            w1 = int(panes[1].shape[1] * (h / panes[1].shape[0]))
            p0 = cv2.resize(panes[0], (w0, h), interpolation=cv2.INTER_LINEAR)
            p1 = cv2.resize(panes[1], (w1, h), interpolation=cv2.INTER_LINEAR)
            frame = np.hstack([p0, p1])

        cv2.imshow(self.window_name, frame)
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
    parser = argparse.ArgumentParser(description="Go2w camera visualization demo")
    parser.add_argument("--test-mode", action="store_true")
    parser.add_argument("--smoke-fast", action="store_true", help="Use lighter simulation settings for smoke tests")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--show-viewer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-sensor-window", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.smoke_fast and args.max_steps > 30:
        args.max_steps = 30

    keyboard: Optional[_KeyboardInput] = None
    camera_window: Optional[_CameraWindow] = None
    overrides = _build_overrides(args.show_viewer, args.test_mode or args.smoke_fast)

    with OmniNavEnv(config_path="configs", config_name="demo/camera_visualization", overrides=overrides) as env:
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
        camera_window = _CameraWindow(enable_window=(args.show_sensor_window and (not args.test_mode)))

        step = 0
        try:
            while True:
                if args.test_mode:
                    cmd = np.array([vx, 0.0, 0.0], dtype=np.float32) if step < args.max_steps // 2 else np.array([0.0, 0.0, -wz], dtype=np.float32)
                else:
                    cmd = keyboard.read() if keyboard is not None else np.zeros(3, dtype=np.float32)
                    if keyboard is not None and not keyboard.running:
                        break

                action = Action(cmd_vel=np.asarray(cmd, dtype=np.float32).reshape(1, 3))
                obs_list, _info = env.step(actions=[action])
                if camera_window is not None and obs_list:
                    camera_window.update(obs_list[0])

                step += 1
                if env.is_done:
                    break
                if args.test_mode and step >= args.max_steps:
                    break
        finally:
            if keyboard is not None:
                keyboard.stop()
            if camera_window is not None:
                camera_window.close()
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
