#!/usr/bin/env python3
"""
Demo 01: Go2 Quadruped Teleoperation (Kinematic Controller)

A game-quality demonstration of quadruped locomotion:
- Pure kinematic control (won't fall)
- Procedural animation with foot locking
- Terrain adaptation (stairs)
- Obstacle avoidance

Scene includes:
- Ground plane
- 3-step staircase
- Obstacle cube

Controls:
    WASD  - Move forward/backward/left/right
    Q/E   - Rotate left/right
    Space - Stop
    Esc   - Exit
"""

import sys
import numpy as np
from omegaconf import OmegaConf

from omninav.core import GenesisSimulationManager
from omninav.robots import Go2Robot
from omninav.locomotion import KinematicController

# =============================================================================
# Keyboard Input
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


# =============================================================================
# Scene Building
# =============================================================================
def build_scene(sim):
    """Add stairs and obstacle to scene."""
    import genesis as gs
    
    # Staircase: 3 steps, ascending height
    step_depths = [0.4, 0.4, 0.4]
    step_heights = [0.05, 0.10, 0.15]
    stair_start_x = 1.5
    stair_width = 1.2
    
    for i, (depth, height) in enumerate(zip(step_depths, step_heights)):
        x = stair_start_x + i * depth
        sim.scene.add_entity(
            morph=gs.morphs.Box(
                pos=(x, 0.0, height / 2),
                size=(depth, stair_width, height),
                fixed=True,
            ),
            surface=gs.surfaces.Default(color=(0.6, 0.6, 0.6, 1.0)),
        )
    
    # Obstacle cube
    sim.scene.add_entity(
        morph=gs.morphs.Box(
            pos=(-1.0, 1.5, 0.25),
            size=(0.5, 0.5, 0.5),
            fixed=True,
        ),
        surface=gs.surfaces.Default(color=(0.8, 0.2, 0.2, 1.0)),
    )


# =============================================================================
# Main
# =============================================================================
def main():
    global running, _listener
    
    print("=" * 60)
    print("  OmniNav Demo 01: Go2 Teleop (Kinematic Controller)")
    print("=" * 60)
    print("Features: Procedural animation, foot locking, stair climbing")
    print("Controls: WASD=move, Q/E=rotate, Space=stop, Esc=exit")
    print("=" * 60)
    
    # -------------------------------------------------------------------------
    # 1. Configuration
    # -------------------------------------------------------------------------
    cfg = OmegaConf.create({
        "simulation": {
            "dt": 0.01,
            "substeps": 2,
            "backend": "gpu",
            "show_viewer": True,
            "camera_pos": [3.0, 3.0, 2.0],
            "camera_lookat": [1.0, 0.0, 0.3],
            "camera_fov": 40,
            "disable_keyboard_shortcuts": True,
        },
        "scene": {
            "ground_plane": {"enabled": True},
        },
    })
    
    robot_cfg = OmegaConf.load("configs/robot/go2.yaml")
    loco_cfg = OmegaConf.load("configs/locomotion/kinematic_gait.yaml")
    
    # -------------------------------------------------------------------------
    # 2. Initialize Simulation
    # -------------------------------------------------------------------------
    sim = GenesisSimulationManager()
    sim.initialize(cfg)
    
    # -------------------------------------------------------------------------
    # 3. Create and Add Robot
    # -------------------------------------------------------------------------
    robot = Go2Robot(robot_cfg, sim.scene)
    sim.add_robot(robot)
    
    # -------------------------------------------------------------------------
    # 4. Build Scene with Stairs and Obstacles
    # -------------------------------------------------------------------------
    build_scene(sim)
    sim.load_scene(cfg.scene) # Restore ground plane
    # -------------------------------------------------------------------------
    # 5. Create Locomotion Controller (Before build to add sensors)
    # -------------------------------------------------------------------------
    controller = KinematicController(loco_cfg, robot)
    
    # NEW: Add controller-specific sensors (Raycasters for terrain)
    # This must be done before sim.build()!
    if hasattr(controller, "recover_cursor_lock"): # Dummy check, real method is add_sensors
        pass 
    try:
        controller.add_sensors(sim.scene)
    except AttributeError:
        # Fallback if method not defined yet (during refactor)
        print("Controller does not support add_sensors yet.")

    # -------------------------------------------------------------------------
    # 6. Build Scene
    # -------------------------------------------------------------------------
    sim.build()
    
    controller.reset()
    
    # -------------------------------------------------------------------------
    # 6. Start Keyboard Listener
    # -------------------------------------------------------------------------
    if not _USE_POLLING:
        _listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        _listener.start()
    
    print("\nSimulation started. Walk towards the stairs at X=1.5.")
    print("The robot should climb them without falling.\n")
    
    # -------------------------------------------------------------------------
    # 7. Initial Stabilization
    # -------------------------------------------------------------------------
    print("Stabilizing...")
    for _ in range(50):
        controller.step(np.zeros(3))
        sim.step()
    print("Ready!\n")
    
    # -------------------------------------------------------------------------
    # 8. Main Simulation Loop
    # -------------------------------------------------------------------------
    step_count = 0
    try:
        while running:
            if _USE_POLLING:
                poll_keyboard()
            
            controller.step(current_cmd)
            sim.step()
            
            step_count += 1
            if step_count % 200 == 0:
                state = robot.get_state()
                pos = state.position
                if pos.ndim > 1:
                    pos = pos[0]
                print(f"Step {step_count}: Pos=[{pos[0]:.2f}, {pos[1]:.2f}], Height={pos[2]:.3f}")
    
    except KeyboardInterrupt:
        pass
    finally:
        if not _USE_POLLING and _listener is not None:
            _listener.stop()
    
    print("\nDemo finished.")


if __name__ == "__main__":
    main()
