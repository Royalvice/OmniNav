# OmniNav 开发任务

## 阶段一：架构设计与规划
- [x] 确认需求与架构决策 <!-- id: arch-decisions -->
- [x] 撰写详细实现计划 (`implementation_plan.md`) <!-- id: impl-plan -->
- [x] 用户审核与反馈 <!-- id: user-review -->

## 阶段二：核心框架搭建 (Current)

### 2.1 Registry 机制
- [x] 实现 `omninav/core/registry.py` <!-- id: registry-impl -->
- [ ] 创建 `tests/core/test_registry.py` <!-- id: core-test -->

### 2.2 Sensor Layer
- [x] 重构 `SensorBase` 到 `omninav/sensors/base.py` <!-- id: sensor-base -->
- [x] 实现 `Lidar2DSensor` (`omninav/sensors/lidar.py`) <!-- id: lidar-impl -->
- [x] 实现 `CameraSensor` (`omninav/sensors/camera.py`) <!-- id: camera-impl -->
- [x] 创建 `configs/sensor/lidar_2d.yaml` 和 `camera_rgbd.yaml` <!-- id: sensor-config -->
- [ ] 创建 `tests/sensors/test_sensors.py` 测试用例 <!-- id: sensor-test -->

### 2.3 Locomotion Layer
- [x] 实现 `WheelController` (`omninav/locomotion/wheel_controller.py`) - Go2w <!-- id: wheel-impl -->
- [x] 实现 `IKController` (`omninav/locomotion/ik_controller.py`) - Go2 IK 步态 <!-- id: ik-impl -->
- [x] 创建 `RLController` 接口占位 (`omninav/locomotion/rl_controller.py`) <!-- id: rl-placeholder -->
- [x] 创建 `configs/locomotion/wheel.yaml` 和 `ik_gait.yaml` <!-- id: loco-config -->
- [ ] 创建 `tests/locomotion/test_locomotion.py` 测试用例 <!-- id: loco-test -->

### 2.4 ROS2 Interface
- [x] 实现 `Ros2Bridge` (`omninav/interfaces/ros2/bridge.py`) <!-- id: ros2-bridge -->
- [x] 实现 `LidarPublisher`, `CameraPublisher`, `OdomPublisher` <!-- id: ros2-publishers -->
- [x] 实现 `CmdVelSubscriber` <!-- id: ros2-subscriber -->
- [ ] 创建 `tests/interfaces/test_ros2_bridge.py` <!-- id: ros2-test -->

### 2.5 Integration
- [ ] 更新 `Go2Robot` / `Go2wRobot` 以支持 Sensor 挂载 <!-- id: robot-sensor -->
- [ ] 创建 `tests/robots/test_robot_sensor_integration.py` <!-- id: integration-test -->
- [ ] 创建 `tests/conftest.py` (pytest fixtures) <!-- id: conftest -->

## 阶段三：算法与验证
- [ ] 算法层 (Algorithm Layer) 基础实现 (Waypoint Follower) <!-- id: algo-layer -->
- [ ] 评测层 (Evaluation Layer) 框架 <!-- id: eval-layer -->
- [ ] 全流程验证 (Sim2Real 预备) <!-- id: validation -->

## 阶段四：文档与示例
- [ ] 编写用户文档 (docs/) <!-- id: docs -->
- [ ] 创建示例脚本 <!-- id: examples -->
