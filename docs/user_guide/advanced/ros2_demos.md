# ROS2 / Nav2 Demos

<div class="lang-zh">

本页对应 [`examples/ros2/omninav_ros2_examples/README.md`](https://github.com/Royalvice/OmniNav/blob/main/examples/ros2/omninav_ros2_examples/README.md)，用于快速跑通 ROS2 + Nav2。

## 1. 环境

1. Ubuntu 22.04
2. ROS2 Humble
3. Nav2

激活环境：

```bash
source ~/omninav_ros_env/bin/activate
source /opt/ros/humble/setup.bash
```

## 2. 构建 ROS2 示例包

```bash
cd /home/vice/code/ros/OmniNav
~/omninav_ros_env/bin/python -m colcon build --symlink-install --packages-select omninav_ros2_examples
source install/setup.bash
```

如配置路径不在默认位置，可设置：

```bash
export OMNINAV_CONFIG_DIR=/path/to/OmniNav/configs
```

## 3. Demo A：RViz 传感器可视化 + viewer 遥操作

```bash
ros2 launch omninav_ros2_examples rviz_sensor_demo.launch.py
```

核心脚本：
- [`examples/ros2/omninav_ros2_examples/omninav_ros2_examples/rviz_sensor_demo.py`](https://github.com/Royalvice/OmniNav/blob/main/examples/ros2/omninav_ros2_examples/omninav_ros2_examples/rviz_sensor_demo.py)

## 4. Demo B：Nav2 闭环

```bash
ros2 launch omninav_ros2_examples nav2_full_stack.launch.py
```

核心脚本：
- [`examples/ros2/omninav_ros2_examples/omninav_ros2_examples/nav2_bridge_demo.py`](https://github.com/Royalvice/OmniNav/blob/main/examples/ros2/omninav_ros2_examples/omninav_ros2_examples/nav2_bridge_demo.py)

在 RViz 中：
1. 先设置 `2D Pose Estimate`
2. 再发送 `2D Goal Pose`

## 5. 可选：导出静态地图给 Nav2

```bash
OMNINAV_EXPORT_NAV2_MAP=1 ros2 run omninav_ros2_examples nav2_bridge_demo --test-mode --no-show-viewer
```

导出结果：
- [`examples/ros2/omninav_ros2_examples/maps/nav_open_space.pgm`](https://github.com/Royalvice/OmniNav/blob/main/examples/ros2/omninav_ros2_examples/maps/nav_open_space.pgm)
- [`examples/ros2/omninav_ros2_examples/maps/nav_open_space.yaml`](https://github.com/Royalvice/OmniNav/blob/main/examples/ros2/omninav_ros2_examples/maps/nav_open_space.yaml)

## 6. 快速检查

```bash
ros2 topic echo /clock
ros2 topic echo /scan --once
ros2 topic echo /camera/rgb/image_raw --once
ros2 topic echo /camera/depth/image_raw --once
ros2 topic echo /cmd_vel --once
```

</div>

<div class="lang-en">

This page maps to [`examples/ros2/omninav_ros2_examples/README.md`](https://github.com/Royalvice/OmniNav/blob/main/examples/ros2/omninav_ros2_examples/README.md) for quick ROS2 + Nav2 bring-up.

## 1. Environment

1. Ubuntu 22.04
2. ROS2 Humble
3. Nav2

Activate runtime environment:

```bash
source ~/omninav_ros_env/bin/activate
source /opt/ros/humble/setup.bash
```

## 2. Build ROS2 example package

```bash
cd /home/vice/code/ros/OmniNav
~/omninav_ros_env/bin/python -m colcon build --symlink-install --packages-select omninav_ros2_examples
source install/setup.bash
```

If configs are not in default path:

```bash
export OMNINAV_CONFIG_DIR=/path/to/OmniNav/configs
```

## 3. Demo A: RViz sensor visualization + viewer teleop

```bash
ros2 launch omninav_ros2_examples rviz_sensor_demo.launch.py
```

Core script:
- [`examples/ros2/omninav_ros2_examples/omninav_ros2_examples/rviz_sensor_demo.py`](https://github.com/Royalvice/OmniNav/blob/main/examples/ros2/omninav_ros2_examples/omninav_ros2_examples/rviz_sensor_demo.py)

## 4. Demo B: Nav2 closed loop

```bash
ros2 launch omninav_ros2_examples nav2_full_stack.launch.py
```

Core script:
- [`examples/ros2/omninav_ros2_examples/omninav_ros2_examples/nav2_bridge_demo.py`](https://github.com/Royalvice/OmniNav/blob/main/examples/ros2/omninav_ros2_examples/omninav_ros2_examples/nav2_bridge_demo.py)

In RViz:
1. Set `2D Pose Estimate`
2. Send `2D Goal Pose`

## 5. Optional: export static map for Nav2

```bash
OMNINAV_EXPORT_NAV2_MAP=1 ros2 run omninav_ros2_examples nav2_bridge_demo --test-mode --no-show-viewer
```

Exported files:
- [`examples/ros2/omninav_ros2_examples/maps/nav_open_space.pgm`](https://github.com/Royalvice/OmniNav/blob/main/examples/ros2/omninav_ros2_examples/maps/nav_open_space.pgm)
- [`examples/ros2/omninav_ros2_examples/maps/nav_open_space.yaml`](https://github.com/Royalvice/OmniNav/blob/main/examples/ros2/omninav_ros2_examples/maps/nav_open_space.yaml)

## 6. Quick checks

```bash
ros2 topic echo /clock
ros2 topic echo /scan --once
ros2 topic echo /camera/rgb/image_raw --once
ros2 topic echo /camera/depth/image_raw --once
ros2 topic echo /cmd_vel --once
```

</div>
