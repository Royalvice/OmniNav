# ROS2 / Nav2 Demos

Source guide:
- [examples/ros2/omninav_ros2_examples/README.md](https://github.com/Royalvice/OmniNav/blob/main/examples/ros2/omninav_ros2_examples/README.md)

## 1. Activate environment

```bash
source ~/omninav_ros_env/bin/activate
source /opt/ros/humble/setup.bash
```

## 2. Build ROS2 package

```bash
cd /home/vice/code/ros/OmniNav
~/omninav_ros_env/bin/python -m colcon build --symlink-install --packages-select omninav_ros2_examples
source install/setup.bash
```

## 3. Demo A: RViz sensor visualization

```bash
ros2 launch omninav_ros2_examples rviz_sensor_demo.launch.py
```

Code:
- [examples/ros2/omninav_ros2_examples/omninav_ros2_examples/rviz_sensor_demo.py](https://github.com/Royalvice/OmniNav/blob/main/examples/ros2/omninav_ros2_examples/omninav_ros2_examples/rviz_sensor_demo.py)

## 4. Demo B: Nav2 closed loop

```bash
ros2 launch omninav_ros2_examples nav2_full_stack.launch.py
```

Code:
- [examples/ros2/omninav_ros2_examples/omninav_ros2_examples/nav2_bridge_demo.py](https://github.com/Royalvice/OmniNav/blob/main/examples/ros2/omninav_ros2_examples/omninav_ros2_examples/nav2_bridge_demo.py)

## 5. Optional map export

```bash
OMNINAV_EXPORT_NAV2_MAP=1 ros2 run omninav_ros2_examples nav2_bridge_demo --test-mode --no-show-viewer
```
