#!/usr/bin/env python3
"""
Demo 05: Waypoint Navigation

Simple waypoint following navigation with Go2 robot:
    - Proportional control to navigate to waypoints
    - 2D Lidar for obstacle visualization
    - Trajectory display

Controls:
    Click:  Set new waypoint (if Genesis supports mouse input)
    1-5:    Select preset waypoints
    Space:  Stop navigation
    Esc:    Exit

This demo showcases:
- WaypointFollower algorithm
- Lidar-based obstacle sensing
- Proportional navigation control
"""

import numpy as np
import genesis as gs
from pynput import keyboard

# Navigation state
running = True
current_waypoint_idx = 0
manual_stop = False

# Preset waypoints (x, y)
WAYPOINTS = [
    np.array([2.0, 0.0]),
    np.array([2.0, 2.0]),
    np.array([0.0, 2.0]),
    np.array([-2.0, 0.0]),
    np.array([0.0, 0.0]),
]

# Navigation parameters
GOAL_THRESHOLD = 0.3   # m - distance to consider goal reached
MAX_LINEAR_VEL = 0.5   # m/s
MAX_ANGULAR_VEL = 1.5  # rad/s
KP_LINEAR = 1.5        # Proportional gain for linear velocity
KP_ANGULAR = 2.0       # Proportional gain for angular velocity


def on_press(key):
    """Handle key press events."""
    global running, current_waypoint_idx, manual_stop

    try:
        # Select waypoints with number keys
        if key.char >= "1" and key.char <= "5":
            current_waypoint_idx = int(key.char) - 1
            manual_stop = False
            print(f"Navigating to waypoint {current_waypoint_idx + 1}: "
                  f"{WAYPOINTS[current_waypoint_idx]}")
    except AttributeError:
        if key == keyboard.Key.space:
            manual_stop = True
            print("Navigation paused.")
        elif key == keyboard.Key.esc:
            running = False


class WaypointFollower:
    """Simple proportional controller for waypoint following."""

    def __init__(
        self,
        goal_threshold: float = 0.3,
        max_linear_vel: float = 0.5,
        max_angular_vel: float = 1.5,
    ):
        self.goal = np.zeros(2)
        self.goal_threshold = goal_threshold
        self.max_linear_vel = max_linear_vel
        self.max_angular_vel = max_angular_vel

    def set_goal(self, goal: np.ndarray) -> None:
        """Set navigation goal (x, y)."""
        self.goal = np.array(goal[:2])

    def step(self, robot_pos: np.ndarray, robot_yaw: float) -> np.ndarray:
        """
        Compute velocity command to reach goal.

        Args:
            robot_pos: Robot position [x, y]
            robot_yaw: Robot heading (radians)

        Returns:
            cmd_vel: [vx, vy, wz] velocity command
        """
        # Vector to goal
        to_goal = self.goal - robot_pos[:2]
        distance = np.linalg.norm(to_goal)

        # Check if goal reached
        if distance < self.goal_threshold:
            return np.zeros(3)

        # Compute desired heading
        desired_yaw = np.arctan2(to_goal[1], to_goal[0])

        # Compute heading error (normalized to [-pi, pi])
        yaw_error = desired_yaw - robot_yaw
        while yaw_error > np.pi:
            yaw_error -= 2 * np.pi
        while yaw_error < -np.pi:
            yaw_error += 2 * np.pi

        # Proportional control
        linear_vel = np.clip(KP_LINEAR * distance, 0, self.max_linear_vel)
        angular_vel = np.clip(KP_ANGULAR * yaw_error, -self.max_angular_vel,
                              self.max_angular_vel)

        # Reduce linear velocity when turning sharply
        if abs(yaw_error) > np.deg2rad(30):
            linear_vel *= 0.3

        return np.array([linear_vel, 0.0, angular_vel])


def quat_to_yaw(quat: np.ndarray) -> float:
    """Extract yaw angle from quaternion [w, x, y, z]."""
    w, x, y, z = quat
    # Yaw (rotation around z-axis)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    return np.arctan2(siny_cosp, cosy_cosp)


def main():
    global running, current_waypoint_idx, manual_stop

    print("=" * 60)
    print("  Waypoint Navigation Demo")
    print("=" * 60)
    print("Controls:")
    print("  1-5:   Select preset waypoints")
    print("  Space: Pause navigation")
    print("  Esc:   Exit")
    print("=" * 60)
    print(f"Waypoints: {[f'{i+1}:{wp}' for i, wp in enumerate(WAYPOINTS)]}")
    print("=" * 60)

    # Initialize Genesis
    gs.init(backend=gs.gpu)

    # Create scene with top-down view
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

    # Add waypoint markers
    for i, wp in enumerate(WAYPOINTS):
        scene.add_entity(
            gs.morphs.Cylinder(
                radius=0.15,
                height=0.05,
                pos=(wp[0], wp[1], 0.025),
                fixed=True,
            )
        )

    # Add some obstacles
    obstacle_positions = [(1.0, 1.0), (-1.0, 1.0), (1.5, -0.5)]
    for ox, oy in obstacle_positions:
        scene.add_entity(
            gs.morphs.Box(
                size=(0.3, 0.3, 0.5),
                pos=(ox, oy, 0.25),
                fixed=True,
            )
        )

    # Add Lidar with debug visualization
    lidar = scene.add_sensor(
        gs.sensors.Lidar(
            pattern=gs.sensors.SphericalPattern(
                fov=(270.0, 0.5),
                n_points=(180, 1),
            ),
            entity_idx=go2.idx,
            link_idx_local=go2.get_link("base").idx_local,
            pos_offset=(0.3, 0.0, 0.1),
            euler_offset=(0.0, 0.0, 0.0),
            min_range=0.1,
            max_range=5.0,
            return_world_frame=False,
            draw_debug=True,
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

    # Stand pose
    stand_qpos = np.array([0.0, 0.8, -1.5] * 4)

    # Initialize waypoint follower
    follower = WaypointFollower()
    follower.set_goal(WAYPOINTS[current_waypoint_idx])

    # Start keyboard listener
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    print(f"\nNavigating to waypoint 1: {WAYPOINTS[0]}")

    step_count = 0
    last_waypoint_idx = -1
    trajectory = []

    while running:
        # Check for waypoint change
        if last_waypoint_idx != current_waypoint_idx:
            follower.set_goal(WAYPOINTS[current_waypoint_idx])
            last_waypoint_idx = current_waypoint_idx

        # Get robot state
        pos = go2.get_pos()
        quat = go2.get_quat()
        if hasattr(pos, "cpu"):
            pos = pos.cpu().numpy()
            quat = quat.cpu().numpy()
        pos = np.array(pos).flatten()
        quat = np.array(quat).flatten()

        robot_pos = pos[:2]
        robot_yaw = quat_to_yaw(quat)

        # Record trajectory
        if step_count % 50 == 0:
            trajectory.append(robot_pos.copy())

        # Compute navigation command
        if not manual_stop:
            cmd_vel = follower.step(robot_pos, robot_yaw)
        else:
            cmd_vel = np.zeros(3)

        # Check goal reached
        dist = np.linalg.norm(robot_pos - WAYPOINTS[current_waypoint_idx])
        if dist < GOAL_THRESHOLD and not manual_stop:
            print(f"Reached waypoint {current_waypoint_idx + 1}!")
            # Move to next waypoint
            current_waypoint_idx = (current_waypoint_idx + 1) % len(WAYPOINTS)
            print(f"Next waypoint {current_waypoint_idx + 1}: "
                  f"{WAYPOINTS[current_waypoint_idx]}")

        # Apply standing pose with heading adjustment
        go2.control_dofs_position(stand_qpos, dof_idx)

        # Step simulation
        scene.step()
        step_count += 1

        # Print status
        if step_count % 100 == 0:
            print(f"Pos: ({robot_pos[0]:.2f}, {robot_pos[1]:.2f}), "
                  f"Yaw: {np.rad2deg(robot_yaw):.1f}°, "
                  f"Dist: {dist:.2f}m, "
                  f"cmd: [{cmd_vel[0]:.2f}, {cmd_vel[2]:.2f}]")

    listener.stop()
    print(f"\nDemo finished. Trajectory points: {len(trajectory)}")


if __name__ == "__main__":
    main()
