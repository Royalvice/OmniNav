#!/usr/bin/env python3
"""
Demo 07: IKController Terrain Adaptation (Stair Climbing)

Verifies:
- Physics-based locomotion with foot IK.
- Terrain sensing via downward raycasters.
- Adaptive foot placement on a stair-like obstacle.
"""

import sys
import numpy as np
from omegaconf import OmegaConf

from omninav.core import GenesisSimulationManager
from omninav.robots import Go2Robot
from omninav.locomotion import IKController

# =============================================================================
# Keyboard Input
# =============================================================================
current_cmd = np.zeros(3)
running = True

LINEAR_VEL = 0.4
ANGULAR_VEL = 0.5

_USE_POLLING = sys.platform == "win32"

if _USE_POLLING:
    import ctypes
    _user32 = ctypes.windll.user32
    _VK = {"w": 0x57, "a": 0x41, "s": 0x53, "d": 0x44, "q": 0x51, "e": 0x45, "space": 0x20, "escape": 0x1B}
    def poll_keyboard():
        global current_cmd, running
        def down(name): return (_user32.GetAsyncKeyState(_VK[name]) & 0x8000) != 0
        if down("escape"): running = False; return
        if down("space"): current_cmd[:] = 0; return
        current_cmd[0] = LINEAR_VEL if down("w") else (-LINEAR_VEL if down("s") else 0.0)
        current_cmd[2] = ANGULAR_VEL if down("q") else (-ANGULAR_VEL if down("e") else 0.0)
else:
    from pynput import keyboard
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

def main():
    global running
    print("=" * 60)
    print("  OmniNav Demo 07: IK Terrain Adaptation (Stairs)")
    print("=" * 60)
    print("Controls: W/S=move, Q/E=rotate, Space=stop, Esc=exit")
    print("=" * 60)
    
    cfg = OmegaConf.create({
        "simulation": {
            "dt": 0.01,
            "substeps": 2,
            "backend": "gpu",
            "show_viewer": True,
            "camera_pos": [3.0, 3.0, 1.5],
            "camera_lookat": [1.5, 0.0, 0.2],
        },
        "scene": {
            "ground_plane": {"enabled": True},
        },
    })
    
    robot_cfg = OmegaConf.load("configs/robot/go2.yaml")
    loco_cfg = OmegaConf.load("configs/locomotion/ik_gait.yaml")
    
    sim = GenesisSimulationManager()
    sim.initialize(cfg)
    
    robot = Go2Robot(robot_cfg, sim.scene)
    sim.add_robot(robot)
    
    # 4. Add Stairs (small boxes)
    import genesis as gs
    for i in range(3):
        h = 0.05 * (i + 1)
        sim.scene.add_entity(
            morph=gs.morphs.Box(
                pos=(1.5 + i * 0.3, 0.0, h/2),
                size=(0.4, 1.0, h),
                fixed=True,
            ),
            surface=gs.surfaces.Default(color=(0.5, 0.5, 0.5, 1.0)),
        )
    
    sim.load_scene(cfg.scene)
    sim.build()
    
    controller = IKController(loco_cfg, robot)
    controller.reset()
    
    if not _USE_POLLING:
        _listener = keyboard.Listener(on_press=on_press).start()
    
    # Auto-move forward
    current_cmd[0] = LINEAR_VEL
    
    step_count = 0
    try:
        while running and step_count < 3000:
            if _USE_POLLING: poll_keyboard()
            controller.step(current_cmd)
            sim.step()
            step_count += 1
            if step_count % 100 == 0:
                pos = robot.get_state().position
                if pos.ndim > 1: pos = pos[0]
                print(f"Step {step_count}: Pos=[{pos[0]:.2f}, {pos[1]:.2f}], Height={pos[2]:.3f}")
    except KeyboardInterrupt: pass
    print("\nDemo finished.")

if __name__ == "__main__":
    main()
