# OmniNav 实现计划

> **项目定位**: 面向上游应用的具身智能仿真平台，基于 Genesis 物理引擎，初版聚焦宇树 Go2w (轮式) 导航仿真。

## 目录
1. [架构总览](#1-架构总览)
2. [目录结构](#2-目录结构)
3. [核心机制 (Registry)](#3-核心机制)
4. [各层详细设计](#4-各层详细设计)
5. [配置管理](#5-配置管理)
6. [验证计划](#6-验证计划)

---

## 0. 编码规范

### 语言规范

| 规则 | 说明 |
|------|------|
| **源码注释** | **必须使用英文** - 所有 Python 代码中的 docstring 和注释必须使用英文 |
| **配置文件** | YAML 配置文件中的注释可以使用中文或英文 |
| **文档** | `docs/` 目录下的用户文档使用中文，`dev_docs/` 使用中文 |
| **提交信息** | Git commit message 使用英文 |

### 代码风格

- **格式化**: 使用 `black`
- **导入排序**: 使用 `isort`
- **类型检查**: 使用 `mypy`

## 1. 架构总览

```
┌─────────────────────────────────────────────────────────────────┐
│                      用户层 (User Layer)                         │
│              Python 脚本 / Notebook / ROS2 节点                   │
└─────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────┐
│                     接口层 (Interface Layer)                      │
│      Python API (OmniNavEnv)  │  ROS2 Bridge (Adapters)         │
│                               │  (发布 /scan, /odom, /image)    │
└─────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────┐
│                算法层 (Algorithm Layer - 可插拔)                   │
│         导航算法 (Navigation) + 感知算法 (Perception)             │
└─────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────┐
│                    运动层 (Locomotion Layer)                      │
│     LocomotionController (cmd_vel -> joint_targets)             │
│       ├── WheelController (Go2w)                                │
│       └── GaitController/RL (Go2 - Future)                      │
└─────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────┐
│                    机器人层 (Robot Layer)                         │
│           RobotBase (硬件抽象: set_dofs, get_state)              │
│       ├── Go2wRobot (轮式)                                      │
│       └── Go2Robot (四足)                                       │
│           └── Sensors (Lidar, Camera)                           │
└─────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────┐
│                      核心层 (Core Layer)                          │
│              SimulationManager + Component Registry               │
│              Genesis Scene Wrapper (load_scene)                   │
└─────────────────────────────────────────────────────────────────┘
```

### 核心设计原则

1.  **Registry 驱动**: 所有可插拔组件 (Robot, Sensor, Algorithm, Task) 通过 `omninav.core.registry` 注册，由 Config `type` 字段动态构建。
2.  **Locomotion 分离**: `Robot` 只负责底层的关节控制 (Joint Level)，`Locomotion` 负责将高层指令 (`cmd_vel`) 转换为关节指令。
3.  **ROS2 适配器模式**: 不在 Core 中硬编码 ROS2，而是通过 `Interface Layer` 的 Adapter 将仿真数据转换为 ROS2 消息。逻辑借鉴 `external/genesis_ros` 但不直接使用其主循环。

---

## 2. 目录结构

```
OmniNav/
├── external/genesis_ros/           # 参考/借鉴源码
├── omninav/
│   ├── core/
│   │   ├── registry.py             # [NEW] 注册器实现
│   │   └── simulation_manager.py
│   ├── assets/                     # 路径解析, USD 加载
│   ├── robots/                     # Robot 实现
│   ├── sensors/                    # [NEW] 传感器实现 (Lidar, Camera)
│   ├── locomotion/                 # 运动控制器
│   ├── algorithms/                 # 导航算法
│   └── interfaces/
│       └── ros2/                   # [NEW] ROS2 Bridge 实现
│           ├── publishers/         # 借鉴 genesis_ros 的发布逻辑
│           └── bridge.py           # ROS2 节点封装
│
├── configs/                        # Hydra 配置
│   ├── robot/
│   │   ├── go2w.yaml               # type: unitree_go2w
│   │   └── go2.yaml                # type: unitree_go2
│   ├── sensor/
│   │   ├── lidar_2d.yaml           # type: lidar_2d
│   │   └── depth_camera.yaml       # type: camera
...
```

---

## 3. 核心机制 (Registry)

实现一个分层注册器 `omninav.core.registry`，支持通过配置动态实例化对象。

```python
# omninav/core/registry.py
class Registry:
    def register(self, name=None): ...
    def build(self, cfg, **kwargs): ...

ROBOT_REGISTRY = Registry("robot")
SENSOR_REGISTRY = Registry("sensor")
LOCOMOTION_REGISTRY = Registry("locomotion")
ALGORITHM_REGISTRY = Registry("algorithm")
```

---

## 4. 各层详细设计

### 4.1 机器人与传感器 (Robot & Sensor)

**SensorBase**:
*   `create(scene)`: 调用 `scene.add_sensor` 或 `scene.add_camera`。
*   `get_data()`: 返回 NumPy 格式数据 (RGB, Depth, Points)。

**具体实现**:
1.  **Lidar2D (`type: lidar_2d`)**:
    *   使用 `gs.sensors.Lidar` + `gs.sensors.SphericalPattern` (设置 flat FOV, e.g., vertical < 1 deg)。
    *   输出模拟 LaserScan 数据。
2.  **Camera (`type: camera`)**:
    *   使用 `gs.sensors.Camera` (或 `vis.camera`)。
    *   支持 RGB 和 Depth。

### 4.2 运动控制 (Locomotion)

**LocomotionControllerBase**:
```python
class LocomotionControllerBase(ABC):
    def compute_action(self, cmd_vel: np.ndarray, obs: Dict) -> np.ndarray:
        """输入速度指令，输出关节目标 (positions/velocities)"""
        pass
```

*   **WheelController** (for Go2w): 解析 $v_x, v_y, \omega_z$ 为轮子转速。
*   **GaitController** (for Go2 - Future): IK 或 RL 策略。

### 4.3 ROS2 接口 (Interface)

**ROS2 Bridge** (`omninav.interfaces.ros2`):
*   独立于 Genesis 主循环，但在 `step()` 后被调用。
*   **Adapter Pattern**:
    *   `LidarPublisher`: `Genesis Lidar Data -> sensor_msgs/LaserScan`
    *   `CameraPublisher`: `Genesis Image -> sensor_msgs/Image` (使用 `cv_bridge` 或纯 numpy)
    *   `OdomPublisher`: `Robot State -> nav_msgs/Odometry`
*   复用 `genesis_ros` 的数据转换逻辑 (如 `raycaster_to_laser_scan_msg`)，但不直接继承其类。

---

## 5. 配置管理

**Registry 配合 Config**:

```yaml
# configs/robot/go2w.yaml
type: unitree_go2w  # ROBOT_REGISTRY key
control:
  controller_type: wheel_controller # LOCOMOTION_REGISTRY key

mounts:
  - sensor: lidar_2d  # sensor config file name
    link: base
```

---

## 6. 验证计划

| 阶段 | 验证内容 | 状态 |
|------|---------|------|
| **Step 1** | 实现 Registry 和各类 Base Class | 待开始 |
| **Step 2** | 实现 Lidar 和 Camera (在 Viewer 中可视化) | 待开始 |
| **Step 3** | 实现 Go2w 的 Locomotion (键盘控制移动) | 待开始 |
| **Step 4** | 实现 ROS2 Bridge (发布 /scan, /image) | 待开始 |
| **Step 5** | 集成测试: 场景 + 机器人 + 传感器 + ROS2 发布 | 待开始 |
