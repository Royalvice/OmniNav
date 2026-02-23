# OmniNav 开发任务清单 (Task List)

本文件只记录**未完成**任务。

说明：
1. 已完成事项不在本文件重复，详见 `WALKTHROUGH.md`。
2. 编号与 `IMPLEMENTATION_PLAN.md`、`WALKTHROUGH.md` 一一对应。
3. 当前版本目标为 `v0.1`。

功能标题标准（与 IMPLEMENTATION_PLAN/WALKTHROUGH 逐字一致）：
- `M1（L0）基础框架与导航原子能力`
- `M2（L1）巡检任务与覆盖评测能力`
- `M3（L2）复杂场景与可通行性能力`
- `P1 插件功能：ROS2 Bridge 能力`
- `P2 插件功能：示例工程化能力`
- `P3 插件功能：文档与发布冻结能力`

---

## 0. 当前未完成焦点 (2026-02-23)

### 主功能主线
- `M1`：L0 导航原子能力（PointNav/ObjectNav/Waypoint）仍需完整评测闭环
- `M2`：L1 巡检能力需从“到点覆盖”升级到“质量覆盖”
- `M3`：L2 复杂场景需形成可复现实验与复杂度评估

### 插件功能主线
- `P3`：文档与发布冻结流程未收口

---

## 1. M1（L0）基础框架与导航原子能力
状态：未完成

### M1.1 PointNavTask 落地
目标：新增标准 PointNav 任务并纳入注册与配置体系。

实施触点：
- `omninav/evaluation/tasks/`
- `omninav/evaluation/__init__.py`
- `configs/task/`

验收标准：
1. `task=point_nav` 可运行
2. 任务结束条件、成功判定、结果汇总完整
3. `TaskResult.metrics` 含基础导航指标

测试要求：
- 单元测试：`tests/evaluation/`
- 集成测试：`tests/integration/`
- Batch 验证：`n_envs=1` 与 `n_envs=4`

勘误说明（基于当前源码）：
1. `configs/task/point_nav.yaml` 已存在，但当前 `omninav/evaluation/tasks/` 下尚无对应已注册的 `PointNavTask` 实现。

### M1.2 ObjectNavTask 落地
目标：新增语义目标导航任务，支持目标类别驱动。

实施触点：
- `omninav/evaluation/tasks/`
- `configs/task/`
- `omninav/core/types.py`（若需新增观察字段）

验收标准：
1. `task=object_nav` 可运行
2. 目标类别输入与终止条件有效
3. 指标输出与 PointNav 可并列比较

测试要求：
- 单元测试：`tests/evaluation/`
- 集成测试：`tests/integration/`
- Batch 验证：`n_envs=1` 与 `n_envs=4`

勘误说明（基于当前源码）：
1. 当前仓库尚无 `ObjectNavTask` 对应已注册任务实现，需要补齐任务类与注册入口。

### M1.3 导航指标标准化
目标：统一 SR/SPL/Collision/Time Efficiency 指标库。

实施触点：
- `omninav/evaluation/metrics/`
- `omninav/evaluation/base.py`
- `configs/task/*.yaml`

验收标准：
1. 指标实现可注册、可配置、可复用
2. 指标输出字段命名统一
3. 可导出对比结果（json/csv）

测试要求：
- 单元测试：指标逐项验证
- 集成测试：任务运行后指标一致性验证

### M1.4 Waypoint 导航稳健性回归
目标：在现有 waypoint 流程上补齐边界行为测试。

实施触点：
- `omninav/algorithms/inspection_planner.py`
- `omninav/algorithms/local_planner.py`
- `tests/algorithms/`

验收标准：
1. waypoint 列表为空、单点、重复点场景可稳定运行
2. waypoint 容差参数变化不破坏任务终止逻辑
3. pipeline 在批量 env 下行为一致

测试要求：
- 单元测试 + 集成测试
- `n_envs=1/4` 对比

---

## 2. M2（L1）巡检任务与覆盖评测能力
状态：未完成

### M2.1 覆盖率从“到点”升级到“质量覆盖”
目标：将巡检成功标准从 waypoint 命中扩展为质量覆盖判定。

实施触点：
- `omninav/evaluation/tasks/inspection_task.py`
- `omninav/evaluation/metrics/inspection_metrics.py`
- `configs/task/inspection.yaml`

验收标准：
1. 覆盖指标包含质量阈值参数（角度/距离/可见性等至少一类）
2. 成功判定可配置地使用“覆盖率 + 质量覆盖率”
3. 与现有 coverage_rate 保持向后兼容配置

测试要求：
- 单元测试：覆盖统计边界条件
- 集成测试：巡检示例结果一致性

### M2.2 巡检结果导出规范
目标：统一巡检任务结果字段，便于后处理和报告。

实施触点：
- `omninav/core/types.py`（`TaskResult.metrics/info` 字段约定）
- `omninav/evaluation/tasks/inspection_task.py`

验收标准：
1. 指标键名稳定（文档化）
2. 结果结构可用于 json/csv 导出
3. 与 M1 导航任务结果格式兼容

测试要求：
- 单元测试：结果结构完整性
- 集成测试：示例脚本结果可解析

### M2.3 巡检任务配置模板扩展
目标：形成可复用的巡检任务模板配置。

实施触点：
- `configs/task/`
- `configs/demo/`

验收标准：
1. 至少一个新增巡检模板可直接运行
2. 所有参数可通过 overrides 覆盖
3. 文档示例命令可复现

测试要求：
- `tests/examples/test_demo_config_contract.py`
- `tests/examples/test_examples_smoke.py`

---

## 3. M3（L2）复杂场景与可通行性能力
状态：未完成

### M3.1 程序化场景生成器
目标：支持规则化生成复杂静态障碍场景。

实施触点：
- `omninav/assets/`（或新增 scene 生成模块）
- `configs/scene/`

验收标准：
1. 同一随机种子可复现实例场景
2. 支持障碍密度、走廊宽度、布局规则参数
3. 输出可直接被现有 runtime 加载

测试要求：
- 单元测试：生成器参数有效性
- 集成测试：生成场景可运行

### M3.2 场景复杂度评估器
目标：提供复杂场景难度量化指标。

实施触点：
- `omninav/evaluation/metrics/`（新增复杂度相关指标）
- `configs/scene/`

验收标准：
1. 至少输出三类复杂度指标（如障碍密度/曲率/遮挡）
2. 指标可用于场景筛选与回归

测试要求：
- 单元测试：指标计算正确性
- 集成测试：不同场景指标可区分

### M3.3 复杂场景基准与复现实验脚本
目标：建立复杂场景最小基准集与实验脚本。

实施触点：
- `configs/scene/`
- `examples/`
- `tests/integration/`

验收标准：
1. 至少 1 个复杂场景 demo 可运行
2. 实验脚本输出关键指标与配置快照

测试要求：
- 集成测试：脚本可跑通
- Batch 验证：`n_envs=1/4`

---

## 4. P3 插件功能：文档与发布冻结能力
状态：未完成

### P3.1 用户文档收口
目标：补齐安装、配置、运行、扩展手册。

实施触点：
- `docs/`
- `README.md`

验收标准：
1. 新用户可按文档完成最小运行
2. 关键配置与 overrides 示例准确

### P3.2 API 文档收口
目标：公开 API docstring 与最小示例对齐当前代码。

实施触点：
- `omninav/interfaces/python_api.py`
- 相关公开模块 docstring

验收标准：
1. 公开 API 输入输出 shape 明确
2. 示例路径与命令无过时引用

### P3.3 v0.1 发布前检查
目标：形成固定发布 checklist。

实施触点：
- `.github/contributing/`
- `tests/`
- `examples/`

验收标准：
1. 新增测试通过、核心测试不回归
2. 至少一个示例可运行
3. 文档与实现与测试一致

---

## 5. 状态迁移规则

1. 任务完成后，从 `TASK.md` 移至 `WALKTHROUGH.md` 对应编号。
2. 若任务导致 API/架构/阶段变化，必须同步更新：
`REQUIREMENTS.md`、`IMPLEMENTATION_PLAN.md`、`TASK.md`、`WALKTHROUGH.md`、`AGENTS.md`。
