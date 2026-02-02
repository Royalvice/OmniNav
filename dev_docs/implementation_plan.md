# OmniNav 实现计划

> **项目定位**: 面向上游应用的具身智能仿真平台，基于 Genesis 物理引擎，初版聚焦宇树 Go2 机器狗导航仿真。

## 目录
1. [架构总览](#1-架构总览)
2. [目录结构](#2-目录结构)
3. [各层详细设计](#3-各层详细设计)
4. [配置管理 (Hydra/OmegaConf)](#4-配置管理)
5. [编码规范](#5-编码规范)
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

- **格式化**: 使用 `black` 进行代码格式化
- **导入排序**: 使用 `isort` 进行导入排序  
- **类型检查**: 使用 `mypy` 进行静态类型检查
- **Lint**: 使用 `ruff` 进行代码检查

## 1. 架构总览

```
┌─────────────────────────────────────────────────────────────────┐
│                      用户层 (User Layer)                         │
│              Python 脚本 / Notebook / ROS2 节点                   │
└─────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────┐
│                     接口层 (Interface Layer)                      │
│                  Python API (OmniNavEnv)                         │
│                  ROS2 Bridge (可选)                               │
└─────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────┐
│                     评测层 (Evaluation Layer)                     │
│              任务定义 (Tasks) + 评价指标 (Metrics)                 │
└─────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────┐
│                算法层 (Algorithm Layer - 可插拔)                   │
│         导航算法 (Navigation) + 感知算法 (Perception)             │
└─────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────┐
│                    运动层 (Locomotion Layer)                      │
│           运动学控制 (Kinematic) / RL 策略 (预留)                  │
└─────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────┐
│                    机器人层 (Robot Layer)                         │
│              Robot 基类 + Go2/Go2w 实现 + 传感器                   │
└─────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────┐
│                     资产层 (Asset Layer)                          │
│              场景加载器 (USD/GLB/Mesh)                            │
└─────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────┐
│                      核心层 (Core Layer)                          │
│              SimulationManager + Genesis Scene Wrapper            │
└─────────────────────────────────────────────────────────────────┘
```

### 核心设计原则

| 原则 | 说明 |
|------|------|
| **分层解耦** | 每层只依赖其下层，上层可独立替换 |
| **接口优先** | 所有模块定义抽象基类 (ABC)，具体实现通过注册机制发现 |
| **配置驱动** | 使用 Hydra/OmegaConf 统一管理所有配置 |
| **可选依赖** | ROS2 相关功能通过配置开关控制，不影响纯 Python 使用 |

---

## 2. 目录结构

```
OmniNav/
├── external/
│   ├── Genesis/                    # git submodule
│   └── genesis_ros/                # git submodule: humble 分支
│
├── src/omninav/
│   ├── core/                       # 核心层
│   ├── assets/                     # 资产层
│   ├── robots/                     # 机器人层
│   ├── locomotion/                 # 运动层
│   ├── algorithms/                 # 算法层 (可插拔)
│   ├── evaluation/                 # 评测层
│   └── interfaces/                 # 接口层
│
├── configs/                        # Hydra 配置
│   ├── config.yaml                 # 主配置入口
│   ├── robot/                      # 机器人配置
│   ├── sensor/                     # 传感器配置
│   ├── scene/                      # 场景配置
│   ├── locomotion/                 # 运动控制配置
│   ├── algorithm/                  # 算法配置
│   └── task/                       # 评测任务配置
│
├── assets/                         # 本地资产文件
│   ├── robots/unitree/
│   └── scenes/
│
├── docs/                           # GitHub Pages 用户文档 (MkDocs)
├── dev_docs/                       # 开发文档 (不入 git)
├── scripts/                        # 示例入口脚本
├── tests/                          # 测试目录
└── pyproject.toml
```

---

## 3. 各层详细设计

### 3.1 核心层 - 关键接口

```python
class SimulationManagerBase(ABC):
    def initialize(self, cfg: DictConfig) -> None: ...
    def build(self) -> None: ...
    def step(self) -> None: ...
    def reset(self) -> None: ...
    def get_sim_time(self) -> float: ...
    def add_robot(self, robot: RobotBase) -> None: ...
    def load_scene(self, scene_cfg: DictConfig) -> None: ...
```

### 3.2 机器人层 - 关键接口

```python
class RobotBase(ABC):
    def spawn(self) -> None: ...
    def get_state(self) -> RobotState: ...
    def apply_command(self, cmd_vel: np.ndarray) -> None: ...
    def mount_sensor(self, mount: SensorMount, sensor: SensorBase) -> None: ...
    def get_observations(self) -> Dict[str, Any]: ...

class SensorBase(ABC):
    def create(self, scene: gs.Scene) -> None: ...
    def get_data(self) -> Any: ...
    def attach_to_robot(self, robot, link_name, position, orientation) -> None: ...
```

### 3.3 算法层 - 关键接口

```python
class AlgorithmBase(ABC):
    def reset(self, task_info: Dict[str, Any]) -> None: ...
    def step(self, observation: Dict[str, Any]) -> np.ndarray: ...  # 返回 cmd_vel
    @property
    def is_done(self) -> bool: ...
```

### 3.4 评测层 - 关键接口

```python
class TaskBase(ABC):
    def reset(self) -> Dict[str, Any]: ...  # 返回 task_info
    def step(self, robot_state, action) -> None: ...
    def is_terminated(self, robot_state) -> bool: ...
    def compute_result(self) -> TaskResult: ...

class MetricBase(ABC):
    def reset(self) -> None: ...
    def update(self, **kwargs) -> None: ...
    def compute(self) -> float: ...
```

---

## 4. 配置管理

### 主配置入口 (`configs/config.yaml`)

```yaml
defaults:
  - robot: go2
  - locomotion: kinematic
  - algorithm: waypoint
  - task: point_nav
  - scene: warehouse
  - _self_

simulation:
  backend: "gpu"
  dt: 0.01
  substeps: 4
  show_viewer: true

ros2:
  enabled: false
```

### 注册机制

所有可插拔模块使用装饰器注册：

```python
@ROBOT_REGISTRY.register("unitree_go2")
class UnitreeGo2(RobotBase): ...

@ALGORITHM_REGISTRY.register("waypoint_follower")
class WaypointFollower(AlgorithmBase): ...
```

---

## 5. 验证计划

| 阶段 | 验证内容 | 状态 |
|------|---------|------|
| 单元测试 | Registry, Robot, Algorithm, Task | 待实现 |
| 集成测试 | 完整导航流程 (场景加载 → 算法运行 → 指标计算) | 待实现 |
| 手动验证 | Viewer 可视化、ROS2 Topic 发布 | 待实现 |
