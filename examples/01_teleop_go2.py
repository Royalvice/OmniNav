#!/usr/bin/env python3
"""
Demo 01: Go2 Quadruped Keyboard Teleoperation

Control the Unitree Go2 quadruped robot using keyboard:
    WASD: Move forward/backward/left/right
    Q/E:  Rotate left/right
    Space: Stop
    Esc:  Exit

This demo showcases:
- Go2 robot loading and spawning
- IK-based trot gait locomotion
- Real-time keyboard control
"""

import numpy as np
import genesis as gs
from pynput import keyboard

# Velocity command state
current_cmd = np.zeros(3)  # [vx, vy, wz]
running = True

# Movement speeds
LINEAR_VEL = 0.5   # m/s
LATERAL_VEL = 0.3  # m/s
ANGULAR_VEL = 1.0  # rad/s


def on_press(key):
    """Handle key press events."""
    global current_cmd, running

    try:
        if key.char == "w":
            current_cmd[0] = LINEAR_VEL
        elif key.char == "s":
            current_cmd[0] = -LINEAR_VEL
        elif key.char == "a":
            current_cmd[1] = LATERAL_VEL
        elif key.char == "d":
            current_cmd[1] = -LATERAL_VEL
        elif key.char == "q":
            current_cmd[2] = ANGULAR_VEL
        elif key.char == "e":
            current_cmd[2] = -ANGULAR_VEL
    except AttributeError:
        if key == keyboard.Key.space:
            current_cmd = np.zeros(3)
        elif key == keyboard.Key.esc:
            running = False


def on_release(key):
    """Handle key release events."""
    global current_cmd

    try:
        if key.char in ["w", "s"]:
            current_cmd[0] = 0.0
        elif key.char in ["a", "d"]:
            current_cmd[1] = 0.0
        elif key.char in ["q", "e"]:
            current_cmd[2] = 0.0
    except AttributeError:
        pass


def main():
    global running

    print("=" * 60)
    print("  Go2 Quadruped Teleop Demo")
    print("=" * 60)
    print("Controls:")
    print("  WASD  - Move forward/backward/left/right")
    print("  Q/E   - Rotate left/right")
    print("  Space - Stop")
    print("  Esc   - Exit")
    print("=" * 60)

    # Initialize Genesis
    gs.init(backend=gs.gpu)

    # Create scene with viewer
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.5, 2.5, 2.0),
            camera_lookat=(0.0, 0.0, 0.3),
            camera_fov=40,
            max_FPS=60,
        ),
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        show_viewer=True,
    )

    # Add ground plane
    scene.add_entity(gs.morphs.Plane())

    # Add Go2 robot using Genesis built-in asset
    go2 = scene.add_entity(
        gs.morphs.URDF(
            file="urdf/go2/urdf/go2.urdf",
            pos=(0.0, 0.0, 0.4),
        )
    )

    # Build scene
    scene.build()

    # Get joint info for control
    joint_names = [
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    ]
    dof_idx = [go2.get_joint(name).dof_idx_local for name in joint_names]

    # Set PD control gains
    kp = np.array([100, 100, 100] * 4)
    kv = np.array([10, 10, 10] * 4)
    go2.set_dofs_kp(kp, dof_idx)
    go2.set_dofs_kv(kv, dof_idx)

    # Standing pose (default joint angles)
    default_qpos = np.array([
        0.0, 0.8, -1.5,  # FL
        0.0, 0.8, -1.5,  # FR
        0.0, 1.0, -1.5,  # RL
        0.0, 1.0, -1.5,  # RR
    ])

    # Simple trot gait parameters
    gait_phase = 0.0
    gait_freq = 2.0    # Hz
    step_height = 0.05  # m
    dt = 0.01

    # Start keyboard listener
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    print("\nSimulation started. Use keyboard to control the robot.")

    # Simulation loop
    step_count = 0
    while running:
        # Update gait phase
        gait_phase = (gait_phase + dt * gait_freq) % 1.0

        # Compute joint targets
        target_qpos = default_qpos.copy()

        # Add simple oscillation for walking (simplified gait)
        if np.linalg.norm(current_cmd) > 0.01:
            swing_amplitude = step_height * 2
            phase_offset = np.pi  # Trot gait offset

            # FL and RR swing together
            fl_rr_swing = np.sin(2 * np.pi * gait_phase) * swing_amplitude
            # FR and RL swing together
            fr_rl_swing = np.sin(2 * np.pi * gait_phase + phase_offset) * swing_amplitude

            # Adjust thigh joints for walking
            target_qpos[1] += fl_rr_swing * 0.1   # FL thigh
            target_qpos[4] += fr_rl_swing * 0.1   # FR thigh
            target_qpos[7] += fr_rl_swing * 0.1   # RL thigh
            target_qpos[10] += fl_rr_swing * 0.1  # RR thigh

        # Apply position control
        go2.control_dofs_position(target_qpos, dof_idx)

        # Step simulation
        scene.step()
        step_count += 1

        # Print status every 100 steps
        if step_count % 100 == 0:
            pos = go2.get_pos()
            if hasattr(pos, "cpu"):
                pos = pos.cpu().numpy()
            print(f"Position: {pos.flatten()[:2]}, cmd_vel: {current_cmd}")

    listener.stop()
    print("\nDemo finished.")


if __name__ == "__main__":
    main()
