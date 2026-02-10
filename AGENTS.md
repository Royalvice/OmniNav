# AGENTS.md - OmniNav AI Agent 指南

本指南面向协助开发 OmniNav 智能巡检导航的仿真平台的 AI 编码助手。

## 1. 上下文同步协议 (Context Sync Routine)

**在执行任何编码任务前**，你必须按照以下顺序同步上下文，以确保与项目最新状态对齐：

1.  **理解愿景 (`.github/contributing/REQUIREMENTS.md`)**: 理解 P0-P2 优先级，确保新功能符合系统架构。
2.  **检查进度 (`.github/contributing/TASK.md`)**: 确认当前处于哪个 Phase，识别标记为 `[ ]` 的待办任务。
3.  **对齐设计 (`.github/contributing/IMPLEMENTATION_PLAN.md`)**: **CRITICAL**。查看 Mermaid 图表、数据流规范（Batch-First）和 API 签名。
4.  **确认功能 (`.github/contributing/WALKTHROUGH.md`)**: 了解已实现的功能和 Demo，避免重复造轮子。
5.  **查询物理引擎接口 (`external/Genesis/doc/source/api_reference`)**: **MANDATORY**。涉及任何对 Genesis 物理引擎接口的对齐、调用或修改，**必须**查询此目录下的官方 API 引用文档。

## 2. 核心架构原则与规范

### 2.1 架构原则
*   **Batch-First Everything**: 所有数据（Observation, Action, State）必须支持 `(num_envs, ...)` 维度。单一环境请使用 `(1, ...)`。
*   **Strongly Typed**: 必须使用 `omninav.core.types` 中定义的 `TypedDict` 进行数据交换。
*   **Registry-Driven**: 不要直接实例化组件，应通过注册器发现，例如 `ROBOT_REGISTRY.build(cfg, ...)`。
*   **Lifecycle Awareness**: 组件必须继承 `LifecycleMixin` 并严格遵循 `CREATED -> SPAWNED -> BUILT -> READY` 状态流转。

### 2.2 核心编码规范

#### 使用 LifecycleMixin 管理初始化时序
```python
class MyComponent(LifecycleMixin):
    def __init__(self, cfg):
        super().__init__()
        # ... 初始化逻辑 ...
        self._transition_to(LifecycleState.CREATED)

    def spawn(self):
        self._require_state(LifecycleState.CREATED)
        # ... 生成逻辑 ...
        self._transition_to(LifecycleState.SPAWNED)
```

#### 使用 Registry 进行插件化扩展
```python
from omninav.core.registry import ALGORITHM_REGISTRY

@ALGORITHM_REGISTRY.register("my_awesome_algo")
class MyAlgo(AlgorithmBase):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg)
```

#### Hydra 配置覆盖
优先使用 `OmniNavEnv.from_config()`，它正确处理了 Hydra 组合逻辑。
```python
env = OmniNavEnv.from_config(overrides=["robot=go2w", "task=inspection"])
```

## 3. 目录结构映射

```text
OmniNav/
├── configs/                        # Hydra 分层配置
├── docs/                           # 用户文档与 API 参考
├── omninav/                        # 核心源码包
│   ├── algorithms/                 # 规划与导航算法 (Plugin-based)
│   ├── assets/                     # 场景生成与资产加载
│   ├── core/                       # 核心层：Runtime, Registry, Lifecycle, Hooks
│   ├── evaluation/                 # 评测层：Task, Metrics
│   ├── interfaces/                 # 接口层：Env, Gym, ROS2
│   ├── locomotion/                 # 运动层：Kinematic/RL 控制器
│   ├── robots/                     # 机器人层：Go2/Go2w
│   └── sensors/                    # 传感器层：Batch 可视化
├── tests/                          # 核心/接口及全流程集成测试
└── examples/                       # 巡检自动化等全流程 Demo
```

## 4. 关键参考文档

| 文档                                                                  | 作用                                           |
| --------------------------------------------------------------------- | ---------------------------------------------- |
| [REQUIREMENTS.md](.github/contributing/REQUIREMENTS.md)               | 愿景与优先级 (Why)                             |
| [IMPLEMENTATION_PLAN.md](.github/contributing/IMPLEMENTATION_PLAN.md) | 架构图、数据流、API 规范 (How)                 |
| [TASK.md](.github/contributing/TASK.md)                               | 详细任务清单 (What's Next)                     |
| [WALKTHROUGH.md](.github/contributing/WALKTHROUGH.md)                 | 进展记录、Demo 说明、技术亮点 (How it's built) |

## 5. 常见错误排查建议

*   **Genesis API**: 严禁依赖陈旧的训练数据。必须在执行前通过 `view_file` 查阅 `external/Genesis/doc/source/api_reference` 中的最新 API 定义以及 `external/Genesis/examples` 中的官方示例。
*   **Hydra Overrides**: 如果覆盖无效，检查是否在 `python_api.py` 的 `from_config` 中正确传递了 `overrides` 列表给 `hydra.compose`。
*   **Batch Dimension**: 处理 Observation 时，始终检查 `len(obs)`。
