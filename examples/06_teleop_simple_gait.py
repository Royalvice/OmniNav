#!/usr/bin/env python3
"""
Demo 06: Go2 Teleoperation (SimpleGaitController - Kinematic Mode)

Features tested:
- Kinematic animation-style control (won't fall)
- Collision detection (won't pass through the obstacle cube)
- Terrain sensing (moves over the ground)
"""

import sys
import numpy as np
from omegaconf import OmegaConf

from omninav.core import GenesisSimulationManager
from omninav.robots import Go2Robot
from omninav.locomotion import SimpleGaitController

# =============================================================================
# Keyboard Input (Adapted from Demo 01)
# =============================================================================
current_cmd = np.zeros(3)  # [vx, vy, wz]
running = True

LINEAR_VEL = 0.5
LATERAL_VEL = 0.3
ANGULAR_VEL = 1.0

_USE_POLLING = sys.platform == "win32"

if _USE_POLLING:
    import ctypes
    _user32 = ctypes.windll.user32
    _VK = {"w": 0x57, "a": 0x41, "s": 0x53, "d": 0x44,
           "q": 0x51, "e": 0x45, "space": 0x20, "escape": 0x1B}

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
    print("  OmniNav Demo 06: Go2 Teleop (SimpleGait Kinematic)")
    print("=" * 60)
    print("Controls: WASD=move, Q/E=rotate, Space=stop, Esc=exit")
    print("=" * 60)
    
    # 1. Configuration
    cfg = OmegaConf.create({
        "simulation": {
            "dt": 0.01,
            "substeps": 2,
            "backend": "gpu",
            "show_viewer": True,
            "camera_pos": [2.5, 2.5, 2.0],
            "camera_lookat": [1.0, 0.0, 0.3],
            "camera_fov": 40,
            "disable_keyboard_shortcuts": True,
        },
        "scene": {
            "ground_plane": {"enabled": True},
        },
    })
    
    robot_cfg = OmegaConf.load("configs/robot/go2.yaml")
    # Force gravity_compensation for kinematic mode
    robot_cfg.gravity_compensation = 1.0
    
    loco_cfg = OmegaConf.load("configs/locomotion/simple_gait.yaml")
    
    # 2. Initialize Simulation
    sim = GenesisSimulationManager()
    sim.initialize(cfg)
    
    # 3. Add Robot
    robot = Go2Robot(robot_cfg, sim.scene)
    sim.add_robot(robot)
    
    # 4. Add Obstacle for collision testing
    import genesis as gs
    sim.scene.add_entity(
        morph=gs.morphs.Box(
            pos=(1.5, 0.0, 0.25),
            size=(0.5, 0.5, 0.5),
        ),
        surface=gs.surfaces.Default(color=(0.8, 0.2, 0.2, 1.0)),
    )
    
    # 5. Build Scene
    sim.load_scene(cfg.scene)
    sim.build()
    
    # 6. Create Locomotion Controller
    controller = SimpleGaitController(loco_cfg, robot)
    controller.reset()
    
    # 7. Start Keyboard Listener
    if not _USE_POLLING:
        _listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        _listener.start()
    
    print("\nSimulation started. Try moving towards the red cube at (1.5, 0.0).\n")
    
    step_count = 0
    try:
        while running:
            if _USE_POLLING: poll_keyboard()
            
            # Step locomotion controller
            controller.step(current_cmd)
            
            # Step simulation
            sim.step()
            
            step_count += 1
            if step_count % 100 == 0:
                pos, _ = controller._get_base_pose()
                print(f"Step {step_count}: Pos=[{pos[0]:.2f}, {pos[1]:.2f}], Height={pos[2]:.3f}")
    
    except KeyboardInterrupt:
        pass
    finally:
        if not _USE_POLLING: _listener.stop()
    
    print("\nDemo finished.")


if __name__ == "__main__":
    main()
