#!/usr/bin/env python3
"""
Demo 03: 2D Lidar Visualization

Visualize 2D Lidar scanning with obstacles in the scene:
    - Go2 robot with front-mounted 2D Lidar
    - Multiple cylindrical and box obstacles
    - Real-time Lidar hit point visualization

Controls:
    WASD:  Move robot
    Q/E:   Rotate robot
    Space: Stop
    Esc:   Exit

This demo showcases:
- Lidar sensor creation and attachment
- Genesis Lidar with SphericalPattern
- Real-time scan visualization
"""

import numpy as np
import genesis as gs
from pynput import keyboard
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Velocity command state
current_cmd = np.zeros(3)
running = True


def on_press(key):
    global current_cmd, running
    try:
        if key.char == "w":
            current_cmd[0] = 0.5
        elif key.char == "s":
            current_cmd[0] = -0.5
        elif key.char == "a":
            current_cmd[1] = 0.3
        elif key.char == "d":
            current_cmd[1] = -0.3
        elif key.char == "q":
            current_cmd[2] = 1.0
        elif key.char == "e":
            current_cmd[2] = -1.0
    except AttributeError:
        if key == keyboard.Key.space:
            current_cmd = np.zeros(3)
        elif key == keyboard.Key.esc:
            running = False


def on_release(key):
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
    print("  2D Lidar Visualization Demo")
    print("=" * 60)
    print("Controls: WASD to move, Q/E to rotate, Space to stop, Esc to exit")
    print("=" * 60)

    # Initialize Genesis
    gs.init(backend=gs.gpu)

    # Create scene
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.0, 0.0, 8.0),
            camera_lookat=(0.0, 0.0, 0.0),
            camera_fov=60,
            max_FPS=60,
        ),
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        show_viewer=True,
    )

    # Add ground plane
    scene.add_entity(gs.morphs.Plane())

    # Add Go2 robot
    go2 = scene.add_entity(
        gs.morphs.URDF(
            file="urdf/go2/urdf/go2.urdf",
            pos=(0.0, 0.0, 0.4),
        )
    )

    # Add obstacles around the robot
    obstacles = []

    # Cylindrical obstacles in a circle
    for i in range(8):
        angle = i * np.pi / 4
        x = 3.0 * np.cos(angle)
        y = 3.0 * np.sin(angle)
        obs = scene.add_entity(
            gs.morphs.Cylinder(
                radius=0.3,
                height=1.0,
                pos=(x, y, 0.5),
                fixed=True,
            )
        )
        obstacles.append(obs)

    # Box obstacles
    box_positions = [(1.5, 0.0, 0.25), (-1.5, 1.0, 0.25), (0.0, -1.5, 0.25)]
    for pos in box_positions:
        obs = scene.add_entity(
            gs.morphs.Box(
                size=(0.5, 0.5, 0.5),
                pos=pos,
                fixed=True,
            )
        )
        obstacles.append(obs)

    # Add 2D Lidar sensor using SphericalPattern
    lidar_pattern = gs.sensors.SphericalPattern(
        fov=(270.0, 0.5),  # (horizontal, vertical) degrees
        n_points=(270, 1),  # (horizontal, vertical) rays
    )

    lidar = scene.add_sensor(
        gs.sensors.Lidar(
            pattern=lidar_pattern,
            entity_idx=go2.idx,
            link_idx_local=go2.get_link("base").idx_local,
            pos_offset=(0.3, 0.0, 0.1),
            euler_offset=(0.0, 0.0, 0.0),
            min_range=0.1,
            max_range=10.0,
            return_world_frame=False,
            draw_debug=True,  # Visualize hits in Genesis viewer
        )
    )

    # Build scene
    scene.build()

    # Set up control gains for Go2
    joint_names = [
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    ]
    dof_idx = [go2.get_joint(name).dof_idx_local for name in joint_names]
    go2.set_dofs_kp(np.array([100.0] * 12), dof_idx)
    go2.set_dofs_kv(np.array([10.0] * 12), dof_idx)

    # Standing pose
    stand_qpos = np.array([0.0, 0.8, -1.5] * 4)

    # Set up matplotlib for scan visualization
    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})
    ax.set_title("2D Lidar Scan")
    ax.set_ylim(0, 10)

    # Initialize plot
    angles = np.linspace(-np.deg2rad(135), np.deg2rad(135), 270)
    line, = ax.plot(angles, np.zeros(270), "b.", markersize=2)

    # Start keyboard listener
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    print("\nSimulation started. Watch the Lidar visualization window.")

    step_count = 0
    while running:
        # Apply standing pose control
        go2.control_dofs_position(stand_qpos, dof_idx)

        # Step simulation
        scene.step()
        step_count += 1

        # Update Lidar visualization every 10 steps
        if step_count % 10 == 0:
            data = lidar.read()
            hit_dist = data.hit_dist
            if hasattr(hit_dist, "cpu"):
                hit_dist = hit_dist.cpu().numpy()

            # Handle dimensions
            ranges = np.array(hit_dist).flatten()
            if len(ranges) >= 270:
                line.set_ydata(ranges[:270])
                fig.canvas.draw_idle()
                fig.canvas.flush_events()

    listener.stop()
    plt.close()
    print("\nDemo finished.")


if __name__ == "__main__":
    main()
