#!/usr/bin/env python3
"""
Demo 04: RGB-D Camera Visualization

Visualize RGB and Depth images from a camera mounted on Go2:
    - Split-screen display: RGB | Depth
    - Real-time camera rendering
    - Keyboard control for robot movement

Controls:
    WASD:  Move robot
    Q/E:   Rotate robot
    Space: Stop
    Esc:   Exit

This demo showcases:
- Genesis RasterizerCameraOptions
- Camera attachment to robot links
- RGB and depth image extraction
"""

import numpy as np
import genesis as gs
from pynput import keyboard
import cv2

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


def depth_to_colormap(depth: np.ndarray, max_depth: float = 10.0) -> np.ndarray:
    """Convert depth image to colormap for visualization."""
    # Normalize depth to 0-255
    depth_normalized = np.clip(depth / max_depth, 0, 1) * 255
    depth_uint8 = depth_normalized.astype(np.uint8)
    # Apply colormap
    return cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)


def main():
    global running

    print("=" * 60)
    print("  RGB-D Camera Visualization Demo")
    print("=" * 60)
    print("Controls: WASD to move, Q/E to rotate, Space to stop, Esc to exit")
    print("=" * 60)

    # Initialize Genesis
    gs.init(backend=gs.gpu)

    # Create scene
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.0, 3.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.3),
            camera_fov=40,
            max_FPS=60,
        ),
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        show_viewer=True,
    )

    # Add ground plane with texture
    scene.add_entity(gs.morphs.Plane())

    # Add Go2 robot
    go2 = scene.add_entity(
        gs.morphs.URDF(
            file="urdf/go2/urdf/go2.urdf",
            pos=(0.0, 0.0, 0.4),
        )
    )

    # Add colorful objects in the scene
    scene.add_entity(
        gs.morphs.Box(
            size=(0.2, 0.2, 0.4),
            pos=(1.0, 0.0, 0.2),
            fixed=True,
        )
    )
    scene.add_entity(
        gs.morphs.Sphere(
            radius=0.15,
            pos=(0.8, 0.5, 0.15),
            fixed=True,
        )
    )
    scene.add_entity(
        gs.morphs.Cylinder(
            radius=0.1,
            height=0.5,
            pos=(0.6, -0.4, 0.25),
            fixed=True,
        )
    )

    # Add RGB-D camera attached to robot head
    head_link = go2.get_link("Head_upper")

    camera = scene.add_sensor(
        gs.sensors.RasterizerCameraOptions(
            res=(640, 480),
            pos=(0.05, 0.0, 0.02),
            lookat=(1.0, 0.0, 0.02),  # Look forward
            up=(0.0, 0.0, 1.0),
            fov=60.0,
            near=0.1,
            far=100.0,
            entity_idx=go2.idx,
            link_idx_local=head_link.idx_local,
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

    # Start keyboard listener
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    print("\nSimulation started. Watch the camera visualization window.")
    print("Press Esc in either window to exit.")

    cv2.namedWindow("RGB-D Camera", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("RGB-D Camera", 1280, 480)

    step_count = 0
    while running:
        # Apply standing pose control
        go2.control_dofs_position(stand_qpos, dof_idx)

        # Step simulation
        scene.step()
        step_count += 1

        # Update camera visualization every 5 steps
        if step_count % 5 == 0:
            data = camera.read()

            # Get RGB image
            rgb = data.rgb
            if hasattr(rgb, "cpu"):
                rgb = rgb.cpu().numpy()
            rgb = np.array(rgb)
            if rgb.ndim == 4:
                rgb = rgb[0]  # Remove batch dimension

            # Get depth image
            depth = data.depth
            if hasattr(depth, "cpu"):
                depth = depth.cpu().numpy()
            depth = np.array(depth)
            if depth.ndim == 3:
                depth = depth[0]

            # Convert depth to colormap
            depth_colored = depth_to_colormap(depth)

            # Convert RGB to BGR for OpenCV
            rgb_bgr = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)

            # Create side-by-side display
            combined = np.hstack([rgb_bgr, depth_colored])

            # Add labels
            cv2.putText(combined, "RGB", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(combined, "Depth", (650, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow("RGB-D Camera", combined)

            # Check for key press in OpenCV window
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # Esc key
                running = False

    listener.stop()
    cv2.destroyAllWindows()
    print("\nDemo finished.")


if __name__ == "__main__":
    main()
