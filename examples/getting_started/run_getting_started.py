#!/usr/bin/env python3
"""Getting Started GUI for OmniNav (Task + Global + Local planners)."""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from omninav.interfaces import OmniNavEnv

# Support both launch styles:
# 1) `python examples/getting_started/run_getting_started.py`
# 2) `python -m examples.getting_started.run_getting_started`
try:
    from examples.getting_started.catalog import list_genesis_candidates, list_omninav_ready
except ModuleNotFoundError:
    from catalog import list_genesis_candidates, list_omninav_ready


@dataclass
class UIConfig:
    task: str = "waypoint"
    global_planner: str = "global_sequential"
    local_planner: str = "dwa_planner"
    robot: str = "go2w"
    scene: str = "complex_flat_obstacles"
    sensor_profile: str = "lidar_camera"
    backend: str = "gpu"
    show_viewer: bool = True
    smoke_fast: bool = False


class Minimap:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.size = 560
        self.scale = 48.0
        self.center = self.size // 2
        self.window = "OmniNav Getting Started Minimap"
        self.image = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        self.path: List[np.ndarray] = []
        self.route: List[np.ndarray] = []
        self.pending_add: Optional[np.ndarray] = None
        self.pending_remove_last = False
        self.pending_clear = False

        if self.enabled:
            cv2.namedWindow(self.window)
            cv2.setMouseCallback(self.window, self._mouse_cb)

    def _mouse_cb(self, event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            wx, wy = self.map_to_world(x, y)
            self.pending_add = np.array([wx, wy], dtype=np.float32)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.pending_remove_last = True
        elif event == cv2.EVENT_MBUTTONDOWN:
            self.pending_clear = True

    def world_to_map(self, x: float, y: float) -> tuple[int, int]:
        return int(self.center + x * self.scale), int(self.center - y * self.scale)

    def map_to_world(self, u: int, v: int) -> tuple[float, float]:
        return (u - self.center) / self.scale, (self.center - v) / self.scale

    def draw(self, pos_xy: Optional[np.ndarray], yaw: float, selected_idx: int) -> None:
        if not self.enabled:
            return
        self.image.fill(245)
        for i in range(-6, 7):
            u, _ = self.world_to_map(float(i), 0.0)
            cv2.line(self.image, (u, 0), (u, self.size), (220, 220, 220), 1)
            _, v = self.world_to_map(0.0, float(i))
            cv2.line(self.image, (0, v), (self.size, v), (220, 220, 220), 1)

        if len(self.path) > 1:
            pts = [self.world_to_map(float(p[0]), float(p[1])) for p in self.path]
            cv2.polylines(self.image, [np.array(pts, dtype=np.int32)], False, (40, 60, 220), 2)

        for i, wp in enumerate(self.route):
            u, v = self.world_to_map(float(wp[0]), float(wp[1]))
            color = (0, 160, 0) if i != selected_idx else (0, 60, 255)
            cv2.circle(self.image, (u, v), 5, color, -1)
            cv2.putText(self.image, str(i), (u + 6, v - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 20, 20), 1)

        if pos_xy is not None:
            u, v = self.world_to_map(float(pos_xy[0]), float(pos_xy[1]))
            cv2.circle(self.image, (u, v), 8, (20, 20, 20), -1)
            hx = int(u + 16 * math.cos(yaw))
            hy = int(v - 16 * math.sin(yaw))
            cv2.line(self.image, (u, v), (hx, hy), (20, 20, 20), 2)

        cv2.putText(
            self.image,
            "L:add waypoint  R:remove last  M:clear route",
            (8, self.size - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (20, 20, 20),
            1,
        )
        cv2.imshow(self.window, self.image)
        cv2.waitKey(1)

    def close(self) -> None:
        if self.enabled:
            cv2.destroyWindow(self.window)


class GettingStartedApp:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.cfg = UIConfig(
            task="waypoint",
            global_planner="global_sequential",
            local_planner="dwa_planner",
            robot="go2w",
            scene="complex_flat_obstacles",
            sensor_profile="lidar_camera",
            backend="cpu" if (args.test_mode or args.smoke_fast) else "gpu",
            show_viewer=bool(getattr(args, "show_viewer", True)),
            smoke_fast=bool(args.smoke_fast),
        )

        self.ready = list_omninav_ready()
        self.candidates = list_genesis_candidates()
        self.env: Optional[OmniNavEnv] = None
        self.obs_list: List[Dict[str, Any]] = []
        self.info: Dict[str, Any] = {}
        self.running = False
        self.paused = False
        self.max_steps = int(args.max_steps)

        self.last_tick = None
        self.fps = 0.0
        self._sim_tick_ms = max(1, int(getattr(args, "tick_ms", 1)))
        self._ui_refresh_interval_s = max(0.05, float(getattr(args, "ui_refresh_ms", 200)) / 1000.0)
        self._result_refresh_interval_s = max(0.1, float(getattr(args, "result_refresh_ms", 500)) / 1000.0)
        self._next_ui_refresh_ts = 0.0
        self._last_result_refresh_ts = 0.0
        self._cached_result = None

        self.route: List[np.ndarray] = [
            np.array([2.0, 0.0], dtype=np.float32),
            np.array([2.0, 2.0], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
        ]
        self.route_dirty = True
        self.active_wp_idx = 0

        self.minimap = Minimap(enabled=not args.test_mode)
        self.root = None
        self.info_text = None
        self.vars: Dict[str, Any] = {}

    @staticmethod
    def _quat_to_yaw(quat_wxyz: np.ndarray) -> float:
        siny_cosp = 2.0 * (quat_wxyz[0] * quat_wxyz[3] + quat_wxyz[1] * quat_wxyz[2])
        cosy_cosp = 1.0 - 2.0 * (quat_wxyz[2] * quat_wxyz[2] + quat_wxyz[3] * quat_wxyz[3])
        return float(math.atan2(siny_cosp, cosy_cosp))

    def _route_to_text(self) -> str:
        if not self.route:
            return "[]"
        return "[" + ",".join(f"[{float(w[0]):.4f},{float(w[1]):.4f},0.0]" for w in self.route) + "]"

    def _build_overrides(self) -> List[str]:
        locomotion_type = "kinematic_wheel_position" if self.cfg.robot == "go2w" else "kinematic_gait"
        o = [
            f"task={self.cfg.task}",
            "algorithm=pipeline_default",
            f"algorithm.global_planner.type={self.cfg.global_planner}",
            f"algorithm.local_planner.type={self.cfg.local_planner}",
            f"robot={self.cfg.robot}",
            f"locomotion.type={locomotion_type}",
            f"scene={self.cfg.scene}",
            f"simulation.backend={self.cfg.backend}",
            f"simulation.show_viewer={str(self.cfg.show_viewer)}",
        ]
        if self.cfg.smoke_fast:
            o.extend(["simulation.substeps=1", "simulation.dt=0.02"])
        if self.cfg.task in ("waypoint", "inspection"):
            o.append(f"task.waypoints={self._route_to_text()}")

        if self.cfg.sensor_profile == "none":
            o.extend(["sensor.depth_camera=null", "sensor.lidar_2d=null"])
        elif self.cfg.sensor_profile == "lidar_only":
            o.extend([
                "sensor.depth_camera=null",
                "sensor.lidar_2d.num_rays=180",
                "sensor.lidar_2d.max_range=10.0",
                "+sensor.lidar_2d.update_every_n_steps=2",
            ])
        elif self.cfg.sensor_profile == "camera_only":
            o.extend([
                "sensor.lidar_2d=null",
                "sensor.depth_camera.width=256",
                "sensor.depth_camera.height=144",
                "sensor.depth_camera.camera_types=[rgb]",
                "+sensor.depth_camera.update_every_n_steps=3",
            ])
        elif self.cfg.sensor_profile == "inspection_minimal":
            o.extend([
                "sensor.depth_camera=null",
                "sensor.lidar_2d.num_rays=180",
                "sensor.lidar_2d.max_range=10.0",
                "+sensor.lidar_2d.update_every_n_steps=2",
            ])
        else:
            # Keep getting_started responsive by default without changing core sensor configs.
            o.extend([
                "sensor.depth_camera.width=256",
                "sensor.depth_camera.height=144",
                "sensor.depth_camera.camera_types=[rgb]",
                "sensor.lidar_2d.num_rays=180",
                "sensor.lidar_2d.max_range=10.0",
                "+sensor.depth_camera.update_every_n_steps=2",
                "+sensor.lidar_2d.update_every_n_steps=2",
            ])
        return o

    def _sync_from_ui(self) -> None:
        if not self.vars:
            return
        self.cfg.task = self.vars["task"].get()
        self.cfg.global_planner = self.vars["global_planner"].get()
        self.cfg.local_planner = self.vars["local_planner"].get()
        self.cfg.robot = self.vars["robot"].get()
        self.cfg.scene = self.vars["scene"].get()
        self.cfg.sensor_profile = self.vars["sensor_profile"].get()
        self.cfg.backend = self.vars["backend"].get()
        self.cfg.show_viewer = bool(self.vars["show_viewer"].get())

    def rebuild(self) -> None:
        self.close_env()
        overrides = self._build_overrides()
        self.env = OmniNavEnv(config_path="configs", config_name="demo/getting_started", overrides=overrides)
        self.obs_list = self.env.reset()
        self.info = {"step": 0, "sim_time": 0.0, "done_mask": None}
        self.running = True
        self.paused = False
        self.last_tick = None
        self.fps = 0.0
        self._next_ui_refresh_ts = 0.0
        self._last_result_refresh_ts = 0.0
        self._cached_result = None
        self.active_wp_idx = 0
        self.route_dirty = False

    def close_env(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None

    def _apply_minimap_events(self) -> None:
        if self.minimap.pending_add is not None:
            self.route.append(self.minimap.pending_add.copy())
            self.minimap.pending_add = None
            self.route_dirty = True
        if self.minimap.pending_remove_last:
            self.minimap.pending_remove_last = False
            if self.route:
                self.route.pop()
                self.route_dirty = True
        if self.minimap.pending_clear:
            self.minimap.pending_clear = False
            self.route = []
            self.route_dirty = True

    def step_once(self) -> None:
        if self.env is None or not self.running or self.paused:
            return
        self._apply_minimap_events()
        if self.route_dirty:
            self.rebuild()
            return

        self.obs_list, self.info = self.env.step()
        now = time.time()
        if self.last_tick is not None:
            self.fps = 1.0 / max(now - self.last_tick, 1e-6)
        self.last_tick = now
        if now - self._last_result_refresh_ts >= self._result_refresh_interval_s:
            self._cached_result = self.env.get_result() if self.env is not None else None
            self._last_result_refresh_ts = now
        if self.env.is_done:
            self.running = False
        if self.args.test_mode and self.info.get("step", 0) >= self.max_steps:
            self.running = False

    def _sensor_meta(self) -> List[str]:
        if self.env is None:
            return []
        out: List[str] = []
        sensor_cfg = self.env.cfg.get("sensor", {})
        if not sensor_cfg:
            return ["- sensors: none configured"]
        for name, scfg in sensor_cfg.items():
            if scfg is None:
                continue
            stype = scfg.get("type", "unknown")
            line = f"- {name} [{stype}] link={scfg.get('link_name', 'n/a')}"
            if stype == "camera":
                line += f" res={scfg.get('width', '?')}x{scfg.get('height', '?')} fov={scfg.get('fov', '?')}"
                line += f" types={list(scfg.get('camera_types', []))}"
            elif stype == "lidar_2d":
                nr = scfg.get("num_rays", "?")
                hfov = scfg.get("horizontal_fov", 360.0)
                line += f" rays={nr} hfov={hfov} range=[{scfg.get('min_range', '?')},{scfg.get('max_range', '?')}]"
            elif stype == "raycaster_depth":
                line += f" res={scfg.get('width', '?')}x{scfg.get('height', '?')}"
            elif stype == "raycaster":
                line += f" size={scfg.get('size', '?')} resolution={scfg.get('resolution', '?')}"
            out.append(line)
        return out if out else ["- sensors: none configured"]

    def _sensor_runtime(self) -> List[str]:
        if not self.obs_list:
            return ["- no sensor runtime data"]
        out: List[str] = []
        sensors = self.obs_list[0].get("sensors", {})
        for name, data in sensors.items():
            if "rgb" in data:
                arr = np.asarray(data["rgb"])
                out.append(f"- {name}.rgb shape={tuple(arr.shape)} dtype={arr.dtype}")
            if "depth" in data:
                arr = np.asarray(data["depth"])
                out.append(f"- {name}.depth shape={tuple(arr.shape)} dtype={arr.dtype}")
            if "ranges" in data:
                rng = np.asarray(data["ranges"], dtype=np.float32)
                if rng.ndim == 2:
                    rng = rng[0]
                valid = rng[np.isfinite(rng) & (rng > 1e-3)]
                if valid.size:
                    out.append(f"- {name}.ranges n={rng.size} min={float(valid.min()):.3f} mean={float(valid.mean()):.3f} max={float(valid.max()):.3f}")
                else:
                    out.append(f"- {name}.ranges n={rng.size} no-valid")
        return out if out else ["- no sensor runtime data"]

    def _info_dump(self) -> str:
        lines: List[str] = []
        step = int(self.info.get("step", 0))
        sim_t = float(self.info.get("sim_time", 0.0))
        lines.append("=== Runtime ===")
        lines.append(f"step={step} sim_time={sim_t:.3f} fps={self.fps:.1f} running={self.running} paused={self.paused}")
        lines.append(f"done_mask={self.info.get('done_mask', None)}")
        lines.append("")

        lines.append("=== Task + Planner ===")
        lines.append(f"task={self.cfg.task} global={self.cfg.global_planner} local={self.cfg.local_planner}")
        if self.env is not None and hasattr(self.env, "algorithm") and self.env.algorithm is not None:
            info = self.env.algorithm.info
            gp = info.get("global_planner", {})
            lp = info.get("local_planner", {})
            lines.append(f"global_info={gp}")
            lines.append(f"local_info={lp}")
            if isinstance(gp, dict) and "current_goal_index" in gp:
                self.active_wp_idx = int(gp["current_goal_index"])
        lines.append("")

        lines.append("=== Robot ===")
        if self.obs_list:
            rs = self.obs_list[0].get("robot_state", {})
            pos = np.asarray(rs.get("position", np.zeros((1, 3))), dtype=np.float32)[0]
            quat = np.asarray(rs.get("orientation", np.array([[1, 0, 0, 0]])), dtype=np.float32)[0]
            lv = np.asarray(rs.get("linear_velocity", np.zeros((1, 3))), dtype=np.float32)[0]
            av = np.asarray(rs.get("angular_velocity", np.zeros((1, 3))), dtype=np.float32)[0]
            lines.append(f"pos=({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}) yaw={self._quat_to_yaw(quat):.3f}")
            lines.append(f"lin_vel=({lv[0]:.3f},{lv[1]:.3f},{lv[2]:.3f}) ang_vel=({av[0]:.3f},{av[1]:.3f},{av[2]:.3f})")
        lines.append("")

        lines.append("=== Task Result Snapshot ===")
        result = self._cached_result
        if result is not None:
            lines.append(f"success={result.success} episode_length={result.episode_length}")
            for k, v in sorted(result.metrics.items()):
                lines.append(f"metric.{k}={v}")
        else:
            lines.append("task result unavailable")
        lines.append("")

        lines.append("=== Sensor Metadata ===")
        lines.extend(self._sensor_meta())
        lines.append("")
        lines.append("=== Sensor Runtime Summary ===")
        lines.extend(self._sensor_runtime())
        lines.append("")

        lines.append("=== Route ===")
        lines.append(f"count={len(self.route)} active_idx={self.active_wp_idx}")
        max_route_lines = 20
        for i, wp in enumerate(self.route[:max_route_lines]):
            lines.append(f"[{i}] ({float(wp[0]):.3f}, {float(wp[1]):.3f})")
        if len(self.route) > max_route_lines:
            lines.append(f"... truncated {len(self.route) - max_route_lines} points ...")
        lines.append("")

        lines.append("=== OmniNav Ready ===")
        lines.append(f"tasks={self.ready['tasks']}")
        lines.append(f"algorithms={self.ready['algorithms']}")
        lines.append("")
        lines.append("=== Genesis Candidate Snapshot ===")
        lines.append(f"urdf(sample)={len(self.candidates['urdf_candidates'])} mjcf(sample)={len(self.candidates['mjcf_candidates'])}")
        lines.append(f"sensors={self.candidates['sensor_capabilities']}")
        lines.append("")
        lines.append("=== Overrides ===")
        for x in self._build_overrides():
            lines.append(x)
        return "\n".join(lines)

    def _refresh(self) -> None:
        if self.info_text is not None:
            self.info_text.configure(state="normal")
            self.info_text.delete("1.0", "end")
            info = self._info_dump()
            for line in info.splitlines():
                if line.startswith("==="):
                    tag = "section"
                elif line.startswith("metric."):
                    tag = "metric"
                elif line.startswith("step=") or "fps=" in line:
                    tag = "runtime"
                elif line.startswith("task=") or line.startswith("global_info=") or line.startswith("local_info="):
                    tag = "planner"
                elif line.startswith("- "):
                    tag = "bullet"
                elif line.startswith("success="):
                    tag = "result"
                else:
                    tag = "normal"
                self.info_text.insert("end", line + "\n", (tag,))
            self.info_text.configure(state="disabled")

        pos_xy = None
        yaw = 0.0
        if self.obs_list:
            rs = self.obs_list[0].get("robot_state", {})
            pos = np.asarray(rs.get("position", np.zeros((1, 3))), dtype=np.float32)[0]
            quat = np.asarray(rs.get("orientation", np.array([[1, 0, 0, 0]])), dtype=np.float32)[0]
            pos_xy = pos[:2]
            yaw = self._quat_to_yaw(quat)
            self.minimap.path.append(pos_xy.copy())
            if len(self.minimap.path) > 1200:
                self.minimap.path = self.minimap.path[-1200:]
        self.minimap.route = [x.copy() for x in self.route]
        self.minimap.draw(pos_xy, yaw, self.active_wp_idx)

    def _build_gui(self) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.root = tk.Tk()
        self.root.title("OmniNav Getting Started")
        self.root.geometry("1220x780")
        self.root.configure(bg="#0b1220")

        left = ttk.Frame(self.root, padding=8)
        left.pack(side="left", fill="y")
        right = ttk.Frame(self.root, padding=8)
        right.pack(side="right", fill="both", expand=True)

        self.vars["task"] = tk.StringVar(value=self.cfg.task)
        self.vars["global_planner"] = tk.StringVar(value=self.cfg.global_planner)
        self.vars["local_planner"] = tk.StringVar(value=self.cfg.local_planner)
        self.vars["robot"] = tk.StringVar(value=self.cfg.robot)
        self.vars["scene"] = tk.StringVar(value=self.cfg.scene)
        self.vars["sensor_profile"] = tk.StringVar(value=self.cfg.sensor_profile)
        self.vars["backend"] = tk.StringVar(value=self.cfg.backend)
        self.vars["show_viewer"] = tk.BooleanVar(value=self.cfg.show_viewer)

        def add_row(label: str, var, values: List[str]) -> None:
            ttk.Label(left, text=label).pack(anchor="w")
            ttk.OptionMenu(left, var, var.get(), *values).pack(fill="x", pady=3)

        add_row("Task", self.vars["task"], ["waypoint", "inspection"])
        add_row("Global Planner", self.vars["global_planner"], ["global_sequential", "global_route_opt"])
        add_row("Local Planner", self.vars["local_planner"], ["dwa_planner"])
        add_row("Robot", self.vars["robot"], ["go2", "go2w"])
        add_row("Scene", self.vars["scene"], self.ready["scenes"])
        add_row("Sensor Profile", self.vars["sensor_profile"], ["none", "lidar_only", "camera_only", "lidar_camera", "inspection_minimal"])
        add_row("Backend", self.vars["backend"], ["gpu", "cpu"])
        ttk.Checkbutton(left, text="Show Viewer", variable=self.vars["show_viewer"]).pack(anchor="w", pady=4)

        ttk.Button(left, text="Start / Rebuild", command=lambda: [self._sync_from_ui(), self.rebuild()]).pack(fill="x", pady=6)
        ttk.Button(left, text="Pause / Resume", command=lambda: setattr(self, "paused", not self.paused)).pack(fill="x", pady=2)
        ttk.Button(left, text="Reset Env", command=lambda: self.env.reset() if self.env is not None else None).pack(fill="x", pady=2)
        ttk.Button(left, text="Stop", command=lambda: setattr(self, "running", False)).pack(fill="x", pady=2)
        ttk.Button(left, text="Clear Route", command=self._clear_route).pack(fill="x", pady=2)

        ttk.Label(
            left,
            text="Minimap:\nL-click add route point\nR-click remove last\nM-click clear all",
            justify="left",
        ).pack(anchor="w", pady=10)

        self.info_text = tk.Text(right, wrap="none")
        self.info_text.pack(fill="both", expand=True)
        self.info_text.configure(
            state="disabled",
            background="#0f172a",
            foreground="#e2e8f0",
            insertbackground="#e2e8f0",
            font=("Consolas", 10),
            padx=8,
            pady=8,
        )
        self.info_text.tag_configure("section", foreground="#93c5fd", font=("Consolas", 13, "bold"), spacing1=6, spacing3=4)
        self.info_text.tag_configure("runtime", foreground="#fbbf24", font=("Consolas", 11, "bold"))
        self.info_text.tag_configure("planner", foreground="#86efac", font=("Consolas", 10, "bold"))
        self.info_text.tag_configure("result", foreground="#fda4af", font=("Consolas", 10, "bold"))
        self.info_text.tag_configure("metric", foreground="#c4b5fd", font=("Consolas", 10))
        self.info_text.tag_configure("bullet", foreground="#67e8f9", font=("Consolas", 10))
        self.info_text.tag_configure("normal", foreground="#e2e8f0", font=("Consolas", 10))

        def on_close():
            self.running = False
            self.close_env()
            self.minimap.close()
            cv2.destroyAllWindows()
            self.root.destroy()

        self.root.protocol("WM_DELETE_WINDOW", on_close)

    def _clear_route(self) -> None:
        self.route = []
        self.route_dirty = True
        self.active_wp_idx = 0

    def _tick(self) -> None:
        tick_start = time.time()
        self.step_once()
        now = time.time()
        if now >= self._next_ui_refresh_ts:
            self._refresh()
            self._next_ui_refresh_ts = now + self._ui_refresh_interval_s
        if self.root is not None:
            elapsed_ms = int((time.time() - tick_start) * 1000)
            delay_ms = max(1, self._sim_tick_ms - elapsed_ms)
            self.root.after(delay_ms, self._tick)

    def run_gui(self) -> int:
        self._build_gui()
        self.rebuild()
        self._tick()
        self.root.mainloop()
        return 0

    def run_test_mode(self) -> int:
        self.rebuild()
        for _ in range(self.max_steps):
            if not self.running:
                break
            self.step_once()
            self._refresh()
        rc = 0 if self.env is not None else 1
        self.close_env()
        self.minimap.close()
        cv2.destroyAllWindows()
        return rc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OmniNav Getting Started GUI")
    parser.add_argument("--test-mode", action="store_true")
    parser.add_argument("--smoke-fast", action="store_true")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--show-viewer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tick-ms", type=int, default=1, help="Main loop tick interval in ms.")
    parser.add_argument("--ui-refresh-ms", type=int, default=200, help="GUI refresh interval in ms.")
    parser.add_argument("--result-refresh-ms", type=int, default=500, help="Task result refresh interval in ms.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    app = GettingStartedApp(args)
    if args.test_mode:
        return app.run_test_mode()
    return app.run_gui()


if __name__ == "__main__":
    sys.exit(main())
