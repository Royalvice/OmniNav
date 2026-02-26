# ROS2 / Nav2 Demos（演示）

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

## 6. 故障排查（高频问题）

### Nav2 报 `map` frame 超时

若日志出现：
- `Timed out waiting for transform from base_link to map`

请按顺序检查：
1. RViz 启动后是否正确发送了 `2D Pose Estimate`。
2. 当前终端是否执行了 `source install/setup.bash`。
3. TF 树是否存在 `map -> odom -> base_link` 链路。

### RViz queue full / 传感器消息丢弃

若日志出现：
- `Message Filter dropping message ... queue is full`

建议：
1. 降低 demo 配置中的发布频率（`scan_every_n_steps`、`rgb_every_n_steps`、`depth_every_n_steps`）。
2. 先确保仿真 FPS 稳定，再逐步提高传感器发布频率。
3. 降低相机分辨率或关闭非关键数据流。
