# OmniNav 开发任务

## 阶段一：架构设计与规划
- [x] 确认需求与架构决策 <!-- id: arch-decisions -->
- [x] 撰写详细需求文档 (`dev_docs/requirements.md`) <!-- id: requirements-doc -->
- [x] 撰写详细实现计划 (`dev_docs/implementation_plan.md`) <!-- id: impl-plan -->
- [x] 用户审核与反馈 <!-- id: user-review -->

## 阶段二：核心框架搭建 (Phase 2 & 2.5 Complete)

### 2.1 Registry & Core
- [x] 实现 `omninav/core/registry.py` <!-- id: registry-impl -->
- [x] 创建 `tests/core/test_registry.py` <!-- id: core-test -->

### 2.2 Robot & Sensor Layer
- [x] Robot/Sensor Base Refactoring (del duplicate SensorBase, add `mount_sensors`) <!-- id: base-refactor -->
- [x] `Lidar2DSensor` & `CameraSensor` 实现 (Genesis API 适配) <!-- id: sensor-impl -->
- [x] `Go2Robot` & `Go2wRobot` 实现 <!-- id: robot-impl -->
- [x] Configs: `go2.yaml`, `go2w.yaml`, `sensor/*.yaml` <!-- id: config-files -->

### 2.3 Locomotion Layer
- [x] `WheelController` (Mecanum IK) <!-- id: wheel-impl -->
- [x] `IKController` (Go2 Gait) <!-- id: ik-impl -->
- [x] `configs/locomotion/*.yaml` <!-- id: loco-config -->

### 2.4 ROS2 Interface (Basic)
- [x] `Ros2Bridge` Basic Impl (`/scan`, `/odom`, `/image`) <!-- id: ros2-basic -->

### 2.5 集成与Demo
- [x] 5 Interactive Examples (`examples/`) <!-- id: demos -->

## 阶段三：算法与API标准化 (Phase 3 - Current)

### 3.1 API 重构 (Batch-First & TypedDict)
- [ ] 定义 Core Types (`omninav/core/types.py`) <!-- id: types-def -->
  - `Observation`, `Action`, `RobotState` (Batch-First)
- [ ] 更新 `OmniNavEnv` 支持 Batch 维度 <!-- id: env-batch -->
- [ ] 重构 `RobotBase` / `SensorBase` 数据接口 <!-- id: component-batch -->

### 3.2 导航算法实现
- [ ] 实现 `WaypointFollower` (支持 Batch 输入) <!-- id: waypoint-algo -->
- [ ] 定义 VLA 接口规范 (Observation with language) <!-- id: vla-interface -->

### 3.3 运动层适配
- [ ] `LocomotionController` 适配 Batch 输入 <!-- id: loco-batch -->
- [ ] `RLLocomotionController` 占位实现 (End-to-End 支持) <!-- id: rl-loco -->

## 阶段四：评测系统 (Evaluation)
- [ ] 评测任务基类 (`TaskBase`) 升级 <!-- id: task-base -->
- [ ] 实现 `PointNavTask` (SR, SPL) <!-- id: pointnav -->
- [ ] 实现 `ObjectNavTask` <!-- id: objectnav -->

## 阶段五：资产与高级特性
- [ ] 场景生成器基类 (`SceneGeneratorBase`) <!-- id: scene-gen-base -->
- [ ] 随机障碍物生成器 (`RandomObstacleGenerator`) <!-- id: rand-obstacle -->
- [ ] 完善 ROS2 `/tf` 树 (map->odom->base) <!-- id: ros2-adv -->
- [ ] 场景重建与轨迹重放 (P2优先级) <!-- id: sim2real-adv -->
