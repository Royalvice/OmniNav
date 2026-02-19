#!/usr/bin/env python3
"""Demo 05: Go2w waypoint navigation with minimap click target."""

from __future__ import annotations

import argparse
import math
import sys
from collections import deque
from typing import Optional

import cv2
import numpy as np

from omninav.core.types import Action
from omninav.interfaces import OmniNavEnv


def _quat_to_yaw_wxyz(quat: np.ndarray) -> float:
    siny_cosp = 2.0 * (quat[0] * quat[3] + quat[1] * quat[2])
    cosy_cosp = 1.0 - 2.0 * (quat[2] * quat[2] + quat[3] * quat[3])
    return float(math.atan2(siny_cosp, cosy_cosp))


def _normalize_angle(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


class _Minimap:
    def __init__(self, enable_window: bool, map_size: int = 500, map_scale: float = 50.0):
        self.enable_window = enable_window
        self.map_size = map_size
        self.map_scale = map_scale
        self.center = map_size // 2
        self.window_name = "Minimap: Click to Navigate"
        self.image = np.zeros((map_size, map_size, 3), dtype=np.uint8)
        self.trajectory = deque(maxlen=1000)
        self.target: Optional[np.ndarray] = None

        if self.enable_window:
            cv2.namedWindow(self.window_name)
            cv2.setMouseCallback(self.window_name, self._mouse_callback)

    def _world_to_map(self, x: float, y: float) -> tuple[int, int]:
        return int(self.center + x * self.map_scale), int(self.center - y * self.map_scale)

    def _map_to_world(self, u: int, v: int) -> tuple[float, float]:
        return (u - self.center) / self.map_scale, (self.center - v) / self.map_scale

    def _mouse_callback(self, event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            wx, wy = self._map_to_world(x, y)
            self.target = np.array([wx, wy], dtype=np.float32)

    def update(self, robot_pos: np.ndarray, yaw: float) -> None:
        if not self.enable_window:
            return

        self.image.fill(240)
        for i in range(-5, 6):
            u, _ = self._world_to_map(float(i), 0.0)
            cv2.line(self.image, (u, 0), (u, self.map_size), (200, 200, 200), 1)
            _, v = self._world_to_map(0.0, float(i))
            cv2.line(self.image, (0, v), (self.map_size, v), (200, 200, 200), 1)

        self.trajectory.append(robot_pos[:2].copy())
        if len(self.trajectory) > 1:
            pts = [self._world_to_map(float(p[0]), float(p[1])) for p in self.trajectory]
            cv2.polylines(self.image, [np.array(pts)], False, (0, 0, 255), 2)

        if self.target is not None:
            tx, ty = self._world_to_map(float(self.target[0]), float(self.target[1]))
            cv2.drawMarker(self.image, (tx, ty), (0, 100, 0), cv2.MARKER_CROSS, 20, 2)
            cv2.circle(self.image, (tx, ty), 5, (0, 100, 0), -1)

        rx, ry = self._world_to_map(float(robot_pos[0]), float(robot_pos[1]))
        cv2.circle(self.image, (rx, ry), 8, (0, 0, 0), -1)
        hx = int(rx + 15 * math.cos(yaw))
        hy = int(ry - 15 * math.sin(yaw))
        cv2.line(self.image, (rx, ry), (hx, hy), (0, 0, 0), 2)
        cv2.imshow(self.window_name, self.image)
        cv2.waitKey(1)

    def close(self) -> None:
        if self.enable_window:
            cv2.destroyWindow(self.window_name)


class _WaypointController:
    def __init__(self, goal_tolerance: float, angle_tolerance: float):
        self.goal_tolerance = goal_tolerance
        self.angle_tolerance = angle_tolerance
        self.target: Optional[np.ndarray] = None
        self._state = "IDLE"
        self._brake_steps = 0

    def set_target(self, xy: Optional[np.ndarray]) -> None:
        self.target = None if xy is None else np.asarray(xy, dtype=np.float32)
        if self.target is not None:
            self._state = "BRAKE_ALIGN"
            self._brake_steps = 10

    def step(self, pos: np.ndarray, yaw: float, max_linear: float, max_angular: float) -> np.ndarray:
        cmd = np.zeros(3, dtype=np.float32)
        if self.target is None:
            return cmd

        delta = self.target - pos[:2]
        dist = float(np.linalg.norm(delta))
        heading = math.atan2(float(delta[1]), float(delta[0]))
        err = _normalize_angle(heading - yaw)

        if self._state.startswith("BRAKE"):
            self._brake_steps -= 1
            if self._brake_steps <= 0:
                self._state = "ALIGN" if self._state == "BRAKE_ALIGN" else "MOVE"
            return cmd

        if self._state == "ALIGN":
            if abs(err) < self.angle_tolerance:
                self._state = "BRAKE_MOVE"
                self._brake_steps = 10
                return cmd
            cmd[2] = float(np.clip(err * 2.0, -max_angular, max_angular))
            return cmd

        if dist < self.goal_tolerance:
            self.target = None
            self._state = "IDLE"
            return cmd

        if abs(err) > 0.5:
            self._state = "BRAKE_ALIGN"
            self._brake_steps = 10
            return cmd

        cmd[0] = float(np.clip(dist, -max_linear, max_linear))
        cmd[2] = float(np.clip(err * 1.5, -max_angular, max_angular))
        self._state = "MOVE"
        return cmd


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
    parser = argparse.ArgumentParser(description="Go2w waypoint navigation demo")
    parser.add_argument("--test-mode", action="store_true")
    parser.add_argument("--smoke-fast", action="store_true", help="Use lighter simulation settings for smoke tests")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--show-viewer", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.smoke_fast and args.max_steps > 40:
        args.max_steps = 40

    overrides = _build_overrides(args.show_viewer, args.test_mode or args.smoke_fast)

    with OmniNavEnv(config_path="configs", config_name="demo/waypoint_navigation", overrides=overrides) as env:
        obs_list = env.reset()
        if not obs_list:
            return 1

        demo_cfg = env.cfg.get("demo", {})
        max_linear = float(demo_cfg.get("max_linear_vel", 0.5))
        max_angular = float(demo_cfg.get("max_angular_vel", 0.5))

        minimap = _Minimap(enable_window=not args.test_mode)
        controller = _WaypointController(
            goal_tolerance=float(demo_cfg.get("goal_tolerance", 0.15)),
            angle_tolerance=float(demo_cfg.get("angle_tolerance", 0.05)),
        )

        init_target = demo_cfg.get("initial_target", None)
        if init_target is not None:
            target = np.asarray(init_target, dtype=np.float32)
            controller.set_target(target)
            minimap.target = target.copy()

        step = 0
        try:
            while True:
                state = obs_list[0]["robot_state"]
                pos = np.asarray(state["position"])[0]
                quat = np.asarray(state["orientation"])[0]
                yaw = _quat_to_yaw_wxyz(quat)

                if minimap.target is not None:
                    if controller.target is None or np.linalg.norm(minimap.target - controller.target) > 1e-2:
                        controller.set_target(minimap.target)

                if args.test_mode and step in (0, args.max_steps // 3, (2 * args.max_steps) // 3):
                    scripted_targets = demo_cfg.get("scripted_targets", [[2.0, 0.0], [2.0, 2.0], [0.0, 1.0]])
                    idx = 0 if step == 0 else (1 if step == args.max_steps // 3 else 2)
                    target = np.asarray(scripted_targets[idx], dtype=np.float32)
                    controller.set_target(target)
                    minimap.target = target.copy()

                cmd = controller.step(pos, yaw, max_linear=max_linear, max_angular=max_angular)
                action = Action(cmd_vel=np.asarray(cmd, dtype=np.float32).reshape(1, 3))

                minimap.update(pos, yaw)
                obs_list, _info = env.step(actions=[action])
                step += 1

                if env.is_done:
                    break
                if args.test_mode and step >= args.max_steps:
                    break
        finally:
            minimap.close()
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
