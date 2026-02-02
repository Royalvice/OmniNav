#!/usr/bin/env python3
"""
Demo 02: Go2w Wheeled Robot Keyboard Teleoperation

Control the Unitree Go2w wheeled robot using keyboard:
    WASD: Move forward/backward/left/right (omnidirectional)
    Q/E:  Rotate left/right
    Space: Stop
    Esc:  Exit

This demo showcases:
- Go2w wheeled robot loading
- Mecanum wheel inverse kinematics
- Smooth omnidirectional motion
"""

import numpy as np
import genesis as gs
from pynput import keyboard

# Velocity command state
current_cmd = np.zeros(3)  # [vx, vy, wz]
running = True

# Movement speeds
LINEAR_VEL = 1.0   # m/s (faster than quadruped)
LATERAL_VEL = 0.8  # m/s
ANGULAR_VEL = 2.0  # rad/s

# Wheel parameters (Mecanum configuration)
WHEEL_RADIUS = 0.05
WHEEL_BASE = 0.4    # front-rear distance
TRACK_WIDTH = 0.3   # left-right distance


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


def compute_wheel_velocities(vx: float, vy: float, wz: float) -> np.ndarray:
    """
    Compute Mecanum wheel velocities from body velocity.

    Args:
        vx: Forward velocity (m/s)
        vy: Lateral velocity (m/s)
        wz: Angular velocity (rad/s)

    Returns:
        Wheel angular velocities [FL, FR, RL, RR] (rad/s)
    """
    R = WHEEL_RADIUS
    L = WHEEL_BASE / 2
    W = TRACK_WIDTH / 2

    # Mecanum wheel inverse kinematics
    v_FL = (vx - vy - (L + W) * wz) / R
    v_FR = (vx + vy + (L + W) * wz) / R
    v_RL = (vx + vy - (L + W) * wz) / R
    v_RR = (vx - vy + (L + W) * wz) / R

    return np.array([v_FL, v_FR, v_RL, v_RR], dtype=np.float32)


def main():
    global running

    print("=" * 60)
    print("  Go2w Wheeled Robot Teleop Demo")
    print("=" * 60)
    print("Controls:")
    print("  WASD  - Omnidirectional movement")
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
            camera_lookat=(0.0, 0.0, 0.2),
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

    # For this demo, using a simple box with 4 wheels as placeholder
    # In production, load Go2w URDF from project assets
    print("\nNote: Using simplified wheeled robot model")
    print("For full Go2w, place URDF at assets/robots/unitree/go2w/\n")

    # Create simple 4-wheeled robot body
    body = scene.add_entity(
        gs.morphs.Box(
            size=(0.4, 0.3, 0.1),
            pos=(0.0, 0.0, 0.15),
        )
    )

    # Add 4 cylinder wheels (simplified)
    wheel_positions = [
        (0.15, 0.18, 0.05),   # FL
        (0.15, -0.18, 0.05),  # FR
        (-0.15, 0.18, 0.05),  # RL
        (-0.15, -0.18, 0.05),  # RR
    ]

    wheels = []
    for i, wpos in enumerate(wheel_positions):
        wheel = scene.add_entity(
            gs.morphs.Cylinder(
                radius=0.05,
                height=0.03,
                pos=wpos,
                euler=(90, 0, 0),
            )
        )
        wheels.append(wheel)

    # Build scene
    scene.build()

    # Start keyboard listener
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    print("Simulation started. Use keyboard to control the robot.")
    print("(Using simplified model - actual wheel control requires URDF)")

    # Simulation loop
    step_count = 0
    while running:
        # Compute wheel velocities
        wheel_vels = compute_wheel_velocities(
            current_cmd[0], current_cmd[1], current_cmd[2]
        )

        # In a full implementation, apply velocities to wheel joints:
        # go2w.control_dofs_velocity(wheel_vels, wheel_dof_idx)

        # For this demo, just show the computed values
        if step_count % 100 == 0 and np.linalg.norm(current_cmd) > 0.01:
            print(f"cmd_vel: [{current_cmd[0]:.2f}, {current_cmd[1]:.2f}, "
                  f"{current_cmd[2]:.2f}] -> wheel_vel: {wheel_vels}")

        # Step simulation
        scene.step()
        step_count += 1

    listener.stop()
    print("\nDemo finished.")


if __name__ == "__main__":
    main()
