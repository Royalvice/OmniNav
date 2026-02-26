# Robots / Sensors / Locomotion API（机器人/传感器/运动控制）

## Robots

- [omninav/robots/base.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/robots/base.py)
- [omninav/robots/go2.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/robots/go2.py)
- [omninav/robots/go2w.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/robots/go2w.py)

导出：
- `RobotBase`
- `Go2Robot`
- `Go2wRobot`

## Sensors

- [omninav/sensors/base.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/sensors/base.py)
- [omninav/sensors/lidar.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/sensors/lidar.py)
- [omninav/sensors/camera.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/sensors/camera.py)
- [omninav/sensors/raycaster.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/sensors/raycaster.py)
- [omninav/sensors/raycaster_depth.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/sensors/raycaster_depth.py)

导出：
- `SensorBase`
- `Lidar2DSensor`
- `CameraSensor`
- `RaycasterSensor`
- `RaycasterDepthSensor`

## Locomotion

- [omninav/locomotion/base.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/locomotion/base.py)
- [omninav/locomotion/wheel_controller.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/locomotion/wheel_controller.py)
- [omninav/locomotion/kinematic_controller.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/locomotion/kinematic_controller.py)
- [omninav/locomotion/rl_controller.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/locomotion/rl_controller.py)

导出：
- `LocomotionControllerBase`
- `WheelController`
- `KinematicController`
- `KinematicWheelPositionController`
- `RLController`

## 配置参考

- [configs/robot](https://github.com/Royalvice/OmniNav/tree/main/configs/robot)
- [configs/sensor](https://github.com/Royalvice/OmniNav/tree/main/configs/sensor)
- [configs/locomotion](https://github.com/Royalvice/OmniNav/tree/main/configs/locomotion)
