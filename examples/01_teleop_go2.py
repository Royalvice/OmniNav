#!/usr/bin/env python3
"""
Demo 01: Go2 Quadruped Keyboard Teleoperation (OmniNav Framework)

Demonstrates OmniNav framework usage:
- GenesisSimulationManager for simulation
- Go2Robot for robot
- SimpleGaitController for locomotion

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
from omninav.locomotion import SimpleGaitController, IKController

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
# Main
# =============================================================================
def main():
    global running, _listener
    
    print("=" * 60)
    print("  OmniNav Demo 01: Go2 Teleop (IK Controller)")
    print("=" * 60)
    print("Controls: WASD=move, Q/E=rotate, Space=stop, Esc=exit")
    print("=" * 60)
    
    # -------------------------------------------------------------------------
    # 1. Configuration (using OmegaConf)
    # -------------------------------------------------------------------------
    cfg = OmegaConf.create({
        "simulation": {
            "dt": 0.01,
            "substeps": 2,
            "backend": "gpu",
            "show_viewer": True,
            "camera_pos": [2.5, 2.5, 2.0],
            "camera_lookat": [0.0, 0.0, 0.3],
            "camera_fov": 40,
            "disable_keyboard_shortcuts": True,  # Allow WASD for teleop
        },
        "scene": {
            "ground_plane": {"enabled": True},
        },
    })
    
    # Load robot config from file
    robot_cfg = OmegaConf.load("configs/robot/go2.yaml")
    
    # Load locomotion config (Use IK config)
    loco_cfg = OmegaConf.load("configs/locomotion/ik_gait.yaml")
    
    # -------------------------------------------------------------------------
    # 2. Initialize Simulation (OmniNav SimulationManager)
    # -------------------------------------------------------------------------
    sim = GenesisSimulationManager()
    sim.initialize(cfg)
    
    # -------------------------------------------------------------------------
    # 3. Create and Add Robot (OmniNav Go2Robot)
    # -------------------------------------------------------------------------
    robot = Go2Robot(robot_cfg, sim.scene)
    sim.add_robot(robot)
    
    # -------------------------------------------------------------------------
    # 4. Load Scene Assets
    # -------------------------------------------------------------------------
    sim.load_scene(cfg.scene)
    
    # -------------------------------------------------------------------------
    # 5. Build Scene (triggers robot.post_build for joint initialization)
    # -------------------------------------------------------------------------
    sim.build()
    
    # -------------------------------------------------------------------------
    # 6. Create Locomotion Controller (OmniNav IKController)
    # -------------------------------------------------------------------------
    controller = IKController(loco_cfg, robot)
    controller.reset()
    
    # -------------------------------------------------------------------------
    # 7. Start Keyboard Listener
    # -------------------------------------------------------------------------
    if not _USE_POLLING:
        _listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        _listener.start()
    
    print("\nSimulation started. Focus on viewer window and use WASD/QE.\n")
    
    # -------------------------------------------------------------------------
    # 8. Initial Stabilization (let robot settle into standing pose)
    # -------------------------------------------------------------------------
    print("Stabilizing robot standing pose...")
    for _ in range(100):
        # Keep controlling to default pose during stabilization
        controller.step(np.zeros(3))
        sim.step()
    print("Stabilization complete. Ready for teleop.\n")
    
    # -------------------------------------------------------------------------
    # 9. Main Simulation Loop
    # -------------------------------------------------------------------------
    step_count = 0
    last_cmd = np.zeros(3)
    try:
        while running:
            if _USE_POLLING:
                poll_keyboard()
            
            # Debug: Print when command changes
            if not np.allclose(current_cmd, last_cmd):
                print(f"[DEBUG] cmd_vel = [{current_cmd[0]:.2f}, {current_cmd[1]:.2f}, {current_cmd[2]:.2f}]")
                last_cmd = current_cmd.copy()
            
            # Step locomotion controller
            controller.step(current_cmd)
            
            # Step simulation
            sim.step()
            
            # Status output (every 100 steps for more frequent feedback)
            step_count += 1
            if step_count % 100 == 0:
                state = robot.get_state()
                pos = state.position
                if pos.ndim > 1:
                    pos = pos[0]
                # Also print Z to see if robot is standing
                # Debug: Print first 3 joint angles to check for jitter
                qpos = robot.entity.get_qpos()
                if hasattr(qpos, 'cpu'): qpos = qpos.cpu().numpy()
                if qpos.ndim > 1: qpos = qpos[0]
                joints = qpos[7:10] # FL Leg joints
                
                print(f"Step {step_count}: Pos=[{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}] | FL_Joints={np.array2string(joints, precision=4)}")
    
    except KeyboardInterrupt:
        pass
    finally:
        if not _USE_POLLING and _listener is not None:
            _listener.stop()
    
    print("\nDemo finished.")


if __name__ == "__main__":
    main()
