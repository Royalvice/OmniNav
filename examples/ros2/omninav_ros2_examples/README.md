# OmniNav ROS2 Quick Start (Humble + Nav2, Ubuntu 22.04)

This package provides two install-space demos for first-time OmniNav + Nav2 users.

- No Gazebo is used.
- OmniNav is the simulator; Nav2 only provides the navigation stack.

## 1. Environment

Use:
- Ubuntu 22.04
- ROS2 Humble
- Nav2

Install OmniNav Python environment by following the repository root guide:
- `INSTALL.md`

Then install Nav2:

```bash
sudo apt update
sudo apt install -y ros-humble-navigation2 ros-humble-nav2-bringup
```

Activate runtime environment before running demos:

```bash
source ~/omninav_ros_env/bin/activate
source /opt/ros/humble/setup.bash
```

## 2. Build ROS2 package (install space)

Use this repository as a ROS2 workspace root (or link this package into your own workspace).
Important: build with the **venv Python interpreter** to keep `ros2 run` entrypoints on the same dependency set.

```bash
cd /home/vice/code/ros/OmniNav
~/omninav_ros_env/bin/python -m colcon build --symlink-install --packages-select omninav_ros2_examples
source install/setup.bash
```

Do not use a system `colcon build` here, otherwise installed scripts may use `/usr/bin/python3`
and miss OmniNav/Genesis dependencies from `~/omninav_ros_env`.

If your OmniNav configs are not under this repo path at runtime, set:

```bash
export OMNINAV_CONFIG_DIR=/path/to/OmniNav/configs
```

## 3. Demo A: RViz sensors + viewer teleop

Launch (recommended):

```bash
ros2 launch omninav_ros2_examples rviz_sensor_demo.launch.py
```

Or run OmniNav only:

```bash
ros2 run omninav_ros2_examples rviz_sensor_demo
```

What you get:
- RViz shows `/camera/rgb/image_raw`, `/camera/depth/image_raw`, `/scan`.
- Genesis viewer keyboard teleop for Go2w:
  - `w/s`: forward/backward
  - `q/e`: turn left/right
  - `space`: stop
  - `esc`: exit

## 4. Demo B: full Nav2 closed loop

Launch full chain:

```bash
ros2 launch omninav_ros2_examples nav2_full_stack.launch.py
```

Or run OmniNav bridge process only:

```bash
ros2 run omninav_ros2_examples nav2_bridge_demo
```

In RViz:
1. Use `2D Pose Estimate` once.
2. Use `2D Goal Pose` to send goals.

Nav2 publishes `/cmd_vel`; OmniNav consumes it and drives Go2w in the OmniNav scene.

## 5. Quick checks

```bash
ros2 topic echo /clock
ros2 topic echo /scan --once
ros2 topic echo /camera/rgb/image_raw --once
ros2 topic echo /camera/depth/image_raw --once
ros2 topic echo /cmd_vel --once
```

TF chain should include:
- `map -> odom -> base -> lidar_frame`
- `base -> camera_rgb_frame`
- `base -> camera_depth_frame`
