# Interfaces API

## Scope

- [omninav/interfaces/python_api.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/interfaces/python_api.py)
- [omninav/interfaces/gym_wrapper.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/interfaces/gym_wrapper.py)
- [omninav/interfaces/ros2/bridge.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/interfaces/ros2/bridge.py)
- [omninav/interfaces/ros2/components.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/interfaces/ros2/components.py)
- [omninav/interfaces/ros2/adapter.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/interfaces/ros2/adapter.py)

## `OmniNavEnv`

Source: [python_api.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/interfaces/python_api.py)

Core lifecycle:
- `__init__(cfg | config_path/config_name/overrides)`
- `reset()`
- `step(actions=None)`
- `get_result()`
- `close()`

Useful properties:
- `is_done`
- `step_count`
- `sim_time`
- `map_service`

## `OmniNavGymWrapper`

Source: [gym_wrapper.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/interfaces/gym_wrapper.py)

Usage:
- Gymnasium-compatible wrapper over `OmniNavEnv`

## ROS2 bridge

Main bridge:
- `ROS2Bridge` in [bridge.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/interfaces/ros2/bridge.py)

Components:
- publishers/subscribers in [components.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/interfaces/ros2/components.py)
- conversion helpers in [adapter.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/interfaces/ros2/adapter.py)

Related demos:
- [rviz_sensor_demo.py](https://github.com/Royalvice/OmniNav/blob/main/examples/ros2/omninav_ros2_examples/omninav_ros2_examples/rviz_sensor_demo.py)
- [nav2_bridge_demo.py](https://github.com/Royalvice/OmniNav/blob/main/examples/ros2/omninav_ros2_examples/omninav_ros2_examples/nav2_bridge_demo.py)
