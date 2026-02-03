#!/usr/bin/env python3
"""
Demo 02: Go2w (Wheeled) Teleoperation (Refactored)

This demo showcases:
1. Loading Unitree Go2w (wheeled) robot
2. Controlling mecanum wheels via WheelController
"""

import sys
import warnings

# Suppress annoying pygltflib warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pygltflib")

import numpy as np
from omegaconf import OmegaConf

# OmniNav imports
from omninav.core import GenesisSimulationManager
from omninav.robots import Go2wRobot
from omninav.locomotion import WheelController

# Velocity command state
current_cmd = np.zeros(3)  # [vx, vy, wz]
running = True

# Movement speeds (reduced for stability)
LINEAR_VEL = 0.5   # m/s
# LATERAL_VEL removed (not using omni-directional)
ANGULAR_VEL = 0.5  # rad/s

# -----------------------------------------------------------------------------
# Keyboard Input Handling
# -----------------------------------------------------------------------------
_USE_POLLING = sys.platform == "win32"

if _USE_POLLING:
    import ctypes
    _user32 = ctypes.windll.user32
    _VK = {
        "w": 0x57, "s": 0x53,
        "q": 0x51, "e": 0x45,
        "space": 0x20, "escape": 0x1B,
    }

    # State tracking for edge detection
    _was_turning = False

    def poll_keyboard(controller_ref):
        global current_cmd, running, _was_turning
        def down(name):
            return (_user32.GetAsyncKeyState(_VK[name]) & 0x8000) != 0
        if down("escape"):
            running = False
            return
        if down("space"):
            current_cmd[:] = 0
            return
            
        current_cmd[0] = LINEAR_VEL if down("w") else (-LINEAR_VEL if down("s") else 0.0)
        # current_cmd[1] unused (lateral)
        
        is_turning = down("q") or down("e")
        current_cmd[2] = ANGULAR_VEL if down("q") else (-ANGULAR_VEL if down("e") else 0.0)
        
        if _was_turning and not is_turning:
            # Released q or e
            if controller_ref:
                controller_ref.force_snap_legs_to_default()
        
        _was_turning = is_turning

else:
    from pynput import keyboard
    _listener = None
    _controller_ref = None # Hack to access controller in callback
    
    def on_press(key):
        global current_cmd, running
        try:
            if key.char == "w": current_cmd[0] = LINEAR_VEL
            elif key.char == "s": current_cmd[0] = -LINEAR_VEL
            elif key.char == "q": current_cmd[2] = ANGULAR_VEL
            elif key.char == "e": current_cmd[2] = -ANGULAR_VEL
        except AttributeError:
            if key == keyboard.Key.space: current_cmd[:] = 0
            elif key == keyboard.Key.esc: running = False
    
    def on_release(key):
        global current_cmd
        try:
            if key.char in ["w", "s"]: current_cmd[0] = 0.0
            elif key.char in ["q", "e"]: 
                current_cmd[2] = 0.0
                if _controller_ref:
                    _controller_ref.force_snap_legs_to_default()
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
            "disable_keyboard_shortcuts": True,
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
    
    # Reset robot to apply initial standing pose
    robot.reset()
    
    # 5. Setup Wheel Controller
    controller = WheelController(cfg.control, robot)
    controller.reset()

    # 6. Start Input
    if not _USE_POLLING:
        # Pass controller to pynput callbacks
        global _controller_ref
        _controller_ref = controller
        
        _listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        _listener.start()

    print("\nSimulation started.\n")

    # 7. Loop
    try:
        while running:
            if _USE_POLLING:
                poll_keyboard(controller)

            controller.step(current_cmd)
            sim.step()
            
    finally:
        if not _USE_POLLING and _listener is not None:
            _listener.stop()
    
    print("\nDemo finished.")


if __name__ == "__main__":
    main()
