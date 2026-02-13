# AGENTS.md - OmniNav AI Agent 工程指南

本指南用于约束 AI 编码助手在 OmniNav 仓库中的协作方式，目标是保证实现与需求、架构、测试三者一致。

## 1. 上下文同步协议 (必须执行)

在执行任何编码任务前，按以下顺序同步上下文：

1. 阅读 `.github/contributing/REQUIREMENTS.md`（理解 P0/P1/P2 优先级）
2. 阅读 `.github/contributing/TASK.md`（确认当前激活阶段和待办项）
3. 阅读 `.github/contributing/IMPLEMENTATION_PLAN.md`（对齐 API、数据流、状态机）
4. 阅读 `.github/contributing/WALKTHROUGH.md`（避免重复建设）
5. 涉及 Genesis 接口时，必须查阅：
`external/Genesis/doc/source/api_reference` 与 `external/Genesis/examples`

---

## 2. 运行环境与命令规范

1. 默认在 `conda` 的 `torch` 环境执行命令
2. 推荐命令前缀（PowerShell）：
`& E:\miniconda\shell\condabin\conda-hook.ps1; conda activate torch; <command>`
3. 若 `conda run -n torch` 出现编码/插件异常，允许切换为 `conda activate torch` 方案

---

## 3. 核心架构原则

1. **Batch-First Everything**
- 所有 Observation/Action/State 均支持 `(B, ...)`
- 单环境单机器人也必须是 `(1, ...)`

2. **Single Source of Truth for Types**
- 跨层数据结构只能定义在 `omninav/core/types.py`
- 禁止在其他模块重复定义 `TaskResult`、`Observation`、`Action`

3. **Registry-Driven Construction**
- 组件必须通过 Registry 构建，不允许硬编码直接实例化
- 使用 `BuildContext` 传递依赖，避免不透明 `**kwargs`

4. **Lifecycle-Managed Components**
- Robot/Sensor/Locomotion/Algorithm/Task 必须遵循统一状态机
- 基本流转：`CREATED -> SPAWNED -> BUILT -> READY`

---

## 4. 编码硬约束

### 4.1 数据契约

1. 每个公共 API 的输入/输出必须标注 shape
2. 关键路径加入 shape 校验（例如 `validate_batch_shape`）
3. 不允许隐式去 batch（如 `(B,3)` 自动退化为 `(3,)`）

### 4.2 组件实现

1. 新增组件必须：
- 继承对应 Base 类
- 在对应 Registry 注册
- 提供最小可运行配置（`configs/...`）

2. 修改 Runtime/Interface 时必须：
- 同步更新测试
- 同步更新文档（TASK/PLAN/WALKTHROUGH）

### 4.3 Hydra 配置

1. 优先使用 `OmniNavEnv.from_config(...)`
2. 覆盖参数必须通过 `overrides: list[str]`
3. 新增配置必须提供默认值并可命令行覆盖

---

## 5. Genesis 接口对齐规则 (强制)

1. 任何涉及 `Scene`/`Entity`/`Sensor` API 的修改前，必须先查阅本地官方文档
2. 任何 Genesis 调用差异，优先以 `external/Genesis/examples` 为行为基准
3. 禁止基于记忆臆测 Genesis API

---

## 6. 测试与验收规则

1. 每次功能变更至少包含：
- 单元测试（对应模块）
- 必要集成测试（跨层行为）

2. Batch 相关功能必须同时验证：
- `n_envs=1`
- `n_envs=4`
  - 若依赖真实 Genesis 重型场景，测试可通过 `OMNINAV_RUN_GENESIS_TESTS=1` 显式开启

3. PR/提交前最低验收：
- 新增测试通过
- 现有核心测试不回归
- 示例脚本可运行（至少 1 个）

---

## 7. 目录职责映射

```text
OmniNav/
├── configs/                        # Hydra 分层配置
├── docs/                           # 用户文档与 API 文档
├── omninav/
│   ├── core/                       # Runtime/Registry/Lifecycle/Types
│   ├── robots/                     # Robot 实现
│   ├── sensors/                    # Sensor 实现
│   ├── locomotion/                 # 运动控制
│   ├── algorithms/                 # 规划与决策
│   ├── evaluation/                 # Task 与 Metric
│   └── interfaces/                 # Python/Gym/ROS2 接口
├── tests/                          # 单元与集成测试
└── examples/                       # 最小可运行示例
```

---

## 8. 常见问题排查

1. Hydra 覆盖不生效：
- 检查 `from_config(..., overrides=...)` 是否透传到 `hydra.compose`

2. Batch 维度异常：
- 检查 `obs/action` 是否被错误降维
- 对关键字段打印 `shape`

3. ROS2 话题行为异常：
- 检查 frame 命名、`/clock` 时间源、消息字段维度

4. Genesis 运行异常：
- 回查 `api_reference` 与官方示例参数签名

---

## 9. 文档同步约定

当出现以下变更时，必须同步更新 `TASK.md` 与 `IMPLEMENTATION_PLAN.md`：

1. API 契约（类型、shape、函数签名）变化
2. 架构执行路径变化（Runtime 编排、生命周期、注册机制）
3. 阶段目标或优先级变化（P0/P1/P2）
