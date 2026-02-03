#!/usr/bin/env python3
"""
Demo 02: Go2w (Wheeled) Teleoperation (Refactored)

This demo showcases:
1. Loading Unitree Go2w (wheeled) robot
2. Controlling mecanum wheels via WheelController
"""

import sys
import numpy as np
from omegaconf import OmegaConf

# OmniNav imports
from omninav.core import GenesisSimulationManager
from omninav.robots import Go2wRobot
from omninav.locomotion import WheelController

# Velocity command state
current_cmd = np.zeros(3)  # [vx, vy, wz]
running = True

# Movement speeds (higher for wheeled robot)
LINEAR_VEL = 1.0   # m/s
LATERAL_VEL = 0.5  # m/s
ANGULAR_VEL = 1.5  # rad/s

# -----------------------------------------------------------------------------
# Keyboard Input Handling
# -----------------------------------------------------------------------------
_USE_POLLING = sys.platform == "win32"

if _USE_POLLING:
    import ctypes
    _user32 = ctypes.windll.user32
    _VK = {
        "w": 0x57, "a": 0x41, "s": 0x53, "d": 0x44,
        "q": 0x51, "e": 0x45,
        "space": 0x20, "escape": 0x1B,
    }

    def poll_keyboard():
        global current_cmd, running
        def down(name):
            return (_user32.GetAsyncKeyState(_VK[name]) & 0x8000) != 0
        if down("escape"):
            running = False
            return
        if down("space"):
            current_cmd[:] = 0
            return
        current_cmd[0] = LINEAR_VEL if down("w") else (-LINEAR_VEL if down("s") else 0.0)
        current_cmd[1] = LATERAL_VEL if down("a") else (-LATERAL_VEL if down("d") else 0.0)
        current_cmd[2] = ANGULAR_VEL if down("q") else (-ANGULAR_VEL if down("e") else 0.0)
else:
    from pynput import keyboard
    _listener = None
    
    def on_press(key):
        global current_cmd, running
        try:
            if key.char == "w": current_cmd[0] = LINEAR_VEL
            elif key.char == "s": current_cmd[0] = -LINEAR_VEL
            elif key.char == "a": current_cmd[1] = LATERAL_VEL
            elif key.char == "d": current_cmd[1] = -LATERAL_VEL
            elif key.char == "q": current_cmd[2] = ANGULAR_VEL
            elif key.char == "e": current_cmd[2] = -ANGULAR_VEL
        except AttributeError:
            if key == keyboard.Key.space: current_cmd[:] = 0
            elif key == keyboard.Key.esc: running = False
    
    def on_release(key):
        global current_cmd
        try:
            if key.char in ["w", "s"]: current_cmd[0] = 0.0
            elif key.char in ["a", "d"]: current_cmd[1] = 0.0
            elif key.char in ["q", "e"]: current_cmd[2] = 0.0
        except AttributeError: pass


def main():
    global running

    print("=" * 60)
    print("  OmniNav Demo 02: Go2w Teleop")
    print("=" * 60)
    # 1. Configuration
    cfg = OmegaConf.create({
        "simulation": {
            "dt": 0.01,
            "backend": "gpu",
            "show_viewer": True,
            "camera_pos": [2.0, 2.0, 1.5],
            "camera_lookat": [0.0, 0.0, 0.3],
        },
        "scene": {
            "ground_plane": {"enabled": True},
        },
        "robot": OmegaConf.load("configs/robot/go2w.yaml"),
        "control": OmegaConf.load("configs/locomotion/wheel.yaml"),
    })

    # 2. Init Simulation
    sim = GenesisSimulationManager()
    sim.initialize(cfg)
    
    # 3. Setup Robot (Go2w)
    robot = Go2wRobot(cfg.robot, sim.scene)
    sim.add_robot(robot)
    
    # 4. Build
    sim.load_scene(cfg.scene)
    sim.build()
    
    # 5. Setup Wheel Controller
    controller = WheelController(cfg.control, robot)
    controller.reset()

    # 6. Start Input
    if not _USE_POLLING:
        _listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        _listener.start()

    print("\nSimulation started.\n")

    # 7. Loop
    try:
        while running:
            if _USE_POLLING:
                poll_keyboard()

            controller.step(current_cmd)
            sim.step()
            
    finally:
        if not _USE_POLLING and _listener is not None:
            _listener.stop()
    
    print("\nDemo finished.")


if __name__ == "__main__":
    main()
