# ROS2 / Nav2 Demos

参考源：
- [examples/ros2/omninav_ros2_examples/README.md](https://github.com/Royalvice/OmniNav/blob/main/examples/ros2/omninav_ros2_examples/README.md)

## 1. 激活环境

```bash
source ~/omninav_ros_env/bin/activate
source /opt/ros/humble/setup.bash
```

## 2. 构建 ROS2 包

```bash
cd /home/vice/code/ros/OmniNav
~/omninav_ros_env/bin/python -m colcon build --symlink-install --packages-select omninav_ros2_examples
source install/setup.bash
```

## 3. Demo A：RViz 传感器可视化

```bash
ros2 launch omninav_ros2_examples rviz_sensor_demo.launch.py
```

源码：
- [examples/ros2/omninav_ros2_examples/omninav_ros2_examples/rviz_sensor_demo.py](https://github.com/Royalvice/OmniNav/blob/main/examples/ros2/omninav_ros2_examples/omninav_ros2_examples/rviz_sensor_demo.py)

## 4. Demo B：Nav2 闭环

```bash
ros2 launch omninav_ros2_examples nav2_full_stack.launch.py
```

源码：
- [examples/ros2/omninav_ros2_examples/omninav_ros2_examples/nav2_bridge_demo.py](https://github.com/Royalvice/OmniNav/blob/main/examples/ros2/omninav_ros2_examples/omninav_ros2_examples/nav2_bridge_demo.py)

## 5. 可选：导出静态地图

```bash
OMNINAV_EXPORT_NAV2_MAP=1 ros2 run omninav_ros2_examples nav2_bridge_demo --test-mode --no-show-viewer
```
