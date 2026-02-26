# Interfaces API

## 覆盖范围

- [omninav/interfaces/python_api.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/interfaces/python_api.py)
- [omninav/interfaces/gym_wrapper.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/interfaces/gym_wrapper.py)
- [omninav/interfaces/ros2/bridge.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/interfaces/ros2/bridge.py)
- [omninav/interfaces/ros2/components.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/interfaces/ros2/components.py)
- [omninav/interfaces/ros2/adapter.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/interfaces/ros2/adapter.py)

## `OmniNavEnv`

源码： [python_api.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/interfaces/python_api.py)

核心生命周期：
- `__init__(cfg | config_path/config_name/overrides)`
- `reset()`
- `step(actions=None)`
- `get_result()`
- `close()`

常用属性：
- `is_done`
- `step_count`
- `sim_time`
- `map_service`

## `OmniNavGymWrapper`

源码： [gym_wrapper.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/interfaces/gym_wrapper.py)

用途：
- 基于 `OmniNavEnv` 的 Gymnasium 兼容封装

## ROS2 桥接

主桥接类：
- `ROS2Bridge` in [bridge.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/interfaces/ros2/bridge.py)

组件：
- 发布/订阅组件： [components.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/interfaces/ros2/components.py)
- 数据适配： [adapter.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/interfaces/ros2/adapter.py)

相关示例：
- [rviz_sensor_demo.py](https://github.com/Royalvice/OmniNav/blob/main/examples/ros2/omninav_ros2_examples/omninav_ros2_examples/rviz_sensor_demo.py)
- [nav2_bridge_demo.py](https://github.com/Royalvice/OmniNav/blob/main/examples/ros2/omninav_ros2_examples/omninav_ros2_examples/nav2_bridge_demo.py)
