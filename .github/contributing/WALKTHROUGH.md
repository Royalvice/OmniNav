# OmniNav 项目进展概览 (Walkthrough)

> 本文档记录项目当前状态、已完成的核心功能以及下一步开发计划。

## ✅ 已完成功能 (Phase 1 - Phase 2.5)

### 1. 核心架构
*   **分层架构**: Core / Robot / Sensor / Locomotion / Algorithm / Interface 分层清晰。
*   **注册机制**: 基于 `omninav.core.registry` 实现组件的动态注册与配置加载。
*   **配置管理**: 使用 Hydra/OmegaConf 管理所有配置 (`configs/`)。

### 2. 机器人与传感器
*   **基类设计**:
    *   `RobotBase`: 支持 `spawn()` 和 `mount_sensors()` 生命周期分离。
    *   `SensorBase`: 统一定义 `attach()` 和 `get_data()` 接口。
*   **具体实现**:
    *   **Go2 (四足)**: 基础支持。
    *   **Go2w (轮式)**: 完整的 Mecanum 轮运动学支持。
    *   **Lidar2D**: 基于 Genesis Spherical Pattern 实现 2D 激光雷达仿真。
    *   **Camera**: 基于 Genesis Rasterizer 实现 RGB-D 相机仿真。

### 3. 可执行 Demos
位于 `examples/` 目录下：
1.  `01_teleop_go2.py`: 四足机器人键盘遥控。
2.  `02_teleop_go2w.py`: 轮式机器人全向移动遥控。
3.  `03_lidar_visualization.py`: 2D Lidar 实时数据可视化。
4.  `04_camera_visualization.py`: RGB-D 相机分屏显示。
5.  `05_waypoint_navigation.py`: 基础航点导航与避障演示。

### 4. 文档建设
*   `dev_docs/requirements.md`: 详细需求规格说明书。
*   `dev_docs/implementation_plan.md`: 详细实现架构与 API 设计（Batch-First, API 标准化）。

---

## 🚧 下一步计划 (Phase 3: 算法与 API 标准化)

当前重点是 **API 重构**，为支持大规模并行训练 (RL) 和复杂的 VLA 任务打基础。

### 1. 核心数据结构定义 (`omninav/core/types.py`)
*   定义 **Batch-First** 的 `TypedDict`:
    *   `Observation`: 包含 `robot_state`, `sensor_data`, `task_info`。
    *   `Action`: 标准化 `cmd_vel`。
    *   `RobotState`: 包含位置、姿态、速度等信息的 Batched Tensor。

### 2. 批量化支持 (Batch Support)
*   升级 `OmniNavEnv` 以处理 `(num_envs, ...)` 数据流。
*   升级 `RobotBase` 和 `SensorBase` 处理并行环境数据。

### 3. 先进算法接入
*   实现支持 Batch 输入的 `WaypointFollower`。
*   设计 VLA (Vision-Language-Action) 接口，在 Observation 中预留语言指令字段。

---

## 📚 常用指令

### 运行 Demo
```bash
# 激活环境 (假设已安装 genesis/pynput/opencv)
python examples/05_waypoint_navigation.py
```

### 运行测试
```bash
pytest tests/
```
