#!/usr/bin/env python3
"""
Demo 04: Camera Visualization (Enhanced + Locomotion)

This demo showcases:
1. Mounting a CameraSensor (RGB-D) to Go2
2. Complex obstacle environment and visible ground
3. Physics-based locomotion control (WASD/QE) using IKController
4. Real-time vision using OpenCV
"""

import sys
import numpy as np
import cv2
from omegaconf import OmegaConf

from omninav.core import GenesisSimulationManager
from omninav.robots import Go2wRobot
from omninav.sensors import CameraSensor
from omninav.locomotion import WheelController

# =============================================================================
# Keyboard Input (Consolidated from Demo 01)
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
           "q": 0x51, "e": 0x45, "r": 0x52, "space": 0x20, "escape": 0x1B}

    def poll_keyboard(controller=None):
        global current_cmd, running
        def down(name):
            return (_user32.GetAsyncKeyState(_VK[name]) & 0x8000) != 0
        if down("escape"):
            running = False
            return
        if down("space") or down("r"):
            current_cmd[:] = 0
            if controller: controller.force_snap_legs_to_default()
            return
        current_cmd[0] = LINEAR_VEL if down("w") else (-LINEAR_VEL if down("s") else 0.0)
        # Go2w is mecanum, can strafe
        current_cmd[1] = 0.0 # Disable strafing for stability per Demo 02
        current_cmd[2] = ANGULAR_VEL if down("q") else (-ANGULAR_VEL if down("e") else 0.0)
else:
    from pynput import keyboard
    _listener = None
    def on_press(key):
        global current_cmd, running
        try:
            if key.char == "w": current_cmd[0] = LINEAR_VEL
            elif key.char == "s": current_cmd[0] = -LINEAR_VEL
            # elif key.char == "a": current_cmd[1] = LATERAL_VEL
            # elif key.char == "d": current_cmd[1] = -LATERAL_VEL
            elif key.char == "q": current_cmd[2] = ANGULAR_VEL
            elif key.char == "e": current_cmd[2] = -ANGULAR_VEL
        except AttributeError:
            if key == keyboard.Key.space: current_cmd[:] = 0
            elif key == keyboard.Key.esc: running = False
    def on_release(key):
        global current_cmd
        try:
            if key.char in ["w", "s"]: current_cmd[0] = 0.0
            # elif key.char in ["a", "d"]: current_cmd[1] = 0.0
            elif key.char in ["q", "e"]: current_cmd[2] = 0.0
        except AttributeError: pass


def main():
    global running
    print("=" * 60)
    print("  OmniNav Demo 04: Camera Visualization (Go2w)")
    print("  Controls: WASD=move, Q/E=rotate, Space=stop, Esc=exit")
    print("=" * 60)

    # 1. Config
    cfg = OmegaConf.create({
        "simulation": {
            "dt": 0.01,
            "backend": "gpu", 
            "show_viewer": True,
            "disable_keyboard_shortcuts": True,
        },
        "scene": {
            "ground_plane": {"enabled": True},
        },
        "robot": OmegaConf.load("configs/robot/go2w.yaml"),
        "sensor": OmegaConf.load("configs/sensor/camera_rgbd.yaml"),
        "control": OmegaConf.load("configs/locomotion/wheel.yaml"),
    })

    # 2. Init Simulation
    sim = GenesisSimulationManager()
    sim.initialize(cfg)
    
    # 3. Add Obstacles (Ring pattern)
    import genesis as gs
    num_cylinders = 8
    for i in range(num_cylinders):
        angle = 2 * np.pi * i / num_cylinders
        x = 3.0 * np.cos(angle)
        y = 3.0 * np.sin(angle)
        sim.scene.add_entity(gs.morphs.Cylinder(height=1.5, radius=0.3, pos=(x, y, 0.75), fixed=True))

    num_boxes = 6
    for i in range(num_boxes):
        angle = 2 * np.pi * i / num_boxes + np.pi / 6
        x = 5.0 * np.cos(angle)
        y = 5.0 * np.sin(angle)
        sim.scene.add_entity(gs.morphs.Box(size=(0.5, 0.5, 2.0 * (i + 1) / num_boxes), pos=(x, y, 1.0), fixed=True))
    
    # 4. Robot + Sensor + Controller
    robot = Go2wRobot(cfg.robot, sim.scene)
    sim.add_robot(robot)
    
    controller = WheelController(cfg.control, robot)
    
    camera = CameraSensor(cfg.sensor, sim.scene, robot)
    camera.attach(link_name="base", position=[0.45, 0.0, 0.2], orientation=[90, 0, -90])
    camera.create()
    
    # 5. Load Scene Assets (Ground plane etc)
    sim.load_scene(cfg.scene)
    
    # 6. Build
    sim.build()
    robot.reset()
    controller.reset()
    
    # 7. Start Teleop
    if not _USE_POLLING:
        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()

    print("Simulation started. Press 'q' in OpenCV window or 'Esc' to exit.")

    # 8. Loop
    try:
        while running:
            if _USE_POLLING:
                poll_keyboard()
            
            # Control
            controller.step(current_cmd)
            sim.step()
            
            # View
            data = camera.get_data()
            rgb = data.get("rgb")[0] if data.get("rgb") is not None else None
            depth = data.get("depth")[0] if data.get("depth") is not None else None
            
            if rgb is not None:
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                cv2.imshow("Robot RGB Eye", bgr)
            
            if depth is not None:
                depth_vis = np.clip(depth, 0, 5.0) / 5.0
                cv2.imshow("Robot Depth Eye", depth_vis)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
