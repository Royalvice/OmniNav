# OmniNav 项目进展与技术手册 (Walkthrough)

本文档只记录**已完成**能力，不记录未完成事项。

编号规则：
- 主功能：`M*`
- 插件功能：`P*`
- 与 `IMPLEMENTATION_PLAN.md`、`TASK.md` 一一对应

功能标题标准（与 IMPLEMENTATION_PLAN/TASK 逐字一致）：
- `M1（L0）基础框架与导航原子能力`
- `M2（L1）巡检任务与覆盖评测能力`
- `M3（L2）复杂场景与可通行性能力`
- `P1 插件功能：ROS2 Bridge 能力`
- `P2 插件功能：示例工程化能力`
- `P3 插件功能：文档与发布冻结能力`

---

## 0. 完成态快照 (2026-02-24)

### 已完成主功能
- `M1` 基础框架与导航原子链路已完成核心骨架，并完成 Task/Algorithm 解耦重构
- `M2` 巡检任务主链路已打通（任务语义与规划算法解耦）
- `M3` 复杂场景基础能力已有可运行基线（静态障碍场景）

### 已完成插件功能
- `P1` ROS2 Bridge 插件能力可用并完成关键可靠性修复
- `P2` examples 工程化与 smoke 测试链路完成

---

## 1. M1（L0）基础框架与导航原子能力
状态：已完成

### M1.1 Runtime 编排主循环
已完成内容：
1. 建立 `SimulationRuntime` 统一编排：reset/step/build 生命周期管理
2. 主循环打通 `Observation -> Algorithm -> Locomotion -> Task`
3. 统一输出 `info.step/sim_time/done_mask`

源码证据：
- `omninav/core/runtime.py`
- `omninav/interfaces/python_api.py`

测试证据：
- `tests/interfaces/test_env.py`
- `tests/integration/test_full_pipeline.py`

### M1.2 Registry 驱动构建
已完成内容：
1. Robot/Sensor/Locomotion/Algorithm/Task/Metric 全部注册构建
2. 通过配置驱动实例化，避免硬编码创建

源码证据：
- `omninav/core/registry.py`
- `omninav/interfaces/python_api.py`

测试证据：
- `tests/core/test_registry.py`

### M1.3 Lifecycle 状态机接入
已完成内容：
1. `CREATED -> SPAWNED -> BUILT -> READY` 生命周期统一
2. Algorithm 与 Task 纳入生命周期体系

源码证据：
- `omninav/core/lifecycle.py`
- `omninav/algorithms/base.py`
- `omninav/evaluation/base.py`
- `omninav/robots/base.py`

测试证据：
- `tests/core/test_lifecycle.py`

### M1.4 Batch-First 与类型单一真源
已完成内容：
1. `Action.cmd_vel` 统一 `(B,3)`
2. `TaskResult/Observation/Action` 集中到 `omninav/core/types.py`
3. 关键路径加入 batch shape 校验
4. Runtime 侧支持批量 `cmd_vel` 契约输入与 `done_mask` 输出

源码证据：
- `omninav/core/types.py`
- `omninav/core/runtime.py`

测试证据：
- `tests/core/test_types.py`
- `tests/interfaces/test_env.py`
- `tests/integration/test_batch_pipeline.py`

勘误说明（基于当前源码）：
1. 当前 locomotion 控制器接口仍以单机器人 `step(cmd_vel, obs)` 为入口，Runtime 在机器人维度调度；因此“批量语义”当前主要体现在数据契约与任务终止掩码层。

### M1.5 Task/Algorithm 解耦与 Global+Local 规划管线
已完成内容：
1. 新增 `WaypointTask`，Waypoint 从“算法语义”迁移到 `Task` 层定义
2. `TaskBase` 新增 `build_task_spec` 与 `update_task_feedback`，任务下发与评测边界清晰
3. 新增 `GlobalPlannerBase` + `global_sequential` + `global_route_opt` 原子模块
4. `AlgorithmPipeline` 改为标准 `global + local` 组合，local 使用 `DWAPlanner`
5. 清理旧任务语义算法与配置：删除 `inspection_planner`、`waypoint_follower` 及旧 `configs/algorithm/inspection.yaml`、`waypoint.yaml`
6. 新增 `configs/algorithm/pipeline_default.yaml`、`global_sequential.yaml`、`global_route_opt.yaml`、`local_dwa.yaml`

源码证据：
- `omninav/evaluation/tasks/waypoint_task.py`
- `omninav/evaluation/base.py`
- `omninav/core/runtime.py`
- `omninav/algorithms/global_base.py`
- `omninav/algorithms/global_sequential.py`
- `omninav/algorithms/global_route_opt.py`
- `omninav/algorithms/pipeline.py`
- `configs/algorithm/pipeline_default.yaml`
- `configs/algorithm/global_sequential.yaml`
- `configs/algorithm/global_route_opt.yaml`
- `configs/algorithm/local_dwa.yaml`

测试证据：
- `tests/algorithms/test_pipeline.py`
- `tests/evaluation/test_waypoint_task.py`
- `tests/core/test_runtime_task_spec.py`

---

## 2. M2（L1）巡检任务与覆盖评测能力
状态：已完成（基础版）

### M2.1 巡检全链路（任务-算法-控制）
已完成内容：
1. `InspectionTask` 任务态管理与终止条件
2. `global_sequential/global_route_opt`（全局）+ `DWAPlanner`（局部）组合 pipeline
3. 巡检示例可运行并支持 smoke 降载

源码证据：
- `omninav/evaluation/tasks/inspection_task.py`
- `omninav/algorithms/global_sequential.py`
- `omninav/algorithms/global_route_opt.py`
- `omninav/algorithms/pipeline.py`
- `configs/task/inspection.yaml`
- `configs/algorithm/pipeline_default.yaml`
- `examples/06_inspection_task.py`

测试证据：
- `tests/algorithms/test_pipeline.py`
- `tests/evaluation/test_inspection.py`
- `tests/integration/test_full_pipeline.py`

### M2.2 巡检指标基础版
已完成内容：
1. coverage/detection/time/safety 指标基类与注册
2. 任务结果汇总到 `TaskResult.metrics`

源码证据：
- `omninav/evaluation/metrics/inspection_metrics.py`
- `omninav/evaluation/base.py`

测试证据：
- `tests/evaluation/test_inspection.py`

勘误说明（基于当前源码）：
1. `detection_rate` 当前为事件记录式指标（`record_detection/record_miss`），尚未形成自动异常检测闭环。

---

## 3. M3（L2）复杂场景与可通行性能力
状态：已完成（基础基线）

### M3.1 静态复杂障碍场景可运行
已完成内容：
1. 支持多类静态障碍（box/cylinder/sphere）加载
2. 提供巡检/导航示例场景配置

源码证据：
- `omninav/core/simulation_manager.py`
- `configs/scene/complex_flat_obstacles.yaml`
- `configs/scene/ring_obstacles.yaml`
- `configs/scene/nav_open_space.yaml`

测试证据：
- `tests/examples/test_demo_config_contract.py`
- `tests/examples/test_examples_smoke.py`

---

## 4. P1 插件功能：ROS2 Bridge 能力
状态：已完成

### P1.1 ROS2 契约化与控制源切换
已完成内容：
1. 配置契约升级为 `control_source/profile/topics/frames/publish/qos`
2. 支持 `python|ros2` 控制源
3. 明确 Nav2 对接边界

源码证据：
- `omninav/interfaces/ros2/bridge.py`
- `omninav/interfaces/ros2/components.py`
- `omninav/interfaces/ros2/adapter.py`
- `configs/config.yaml`

测试证据：
- `tests/interfaces/test_ros2_bridge.py`
- `tests/interfaces/test_ros2_adapter.py`
- `tests/examples/test_ros2_example_config_sync.py`

### P1.2 可靠性修复（TF 与 DDS 预热）
已完成内容：
1. static TF 批量发布修复 late-joiner 丢失
2. 增加 warmup/spin timeout/republish 步数配置，改善 WSL2 发现竞态

源码证据：
- `omninav/interfaces/ros2/bridge.py`
- `omninav/interfaces/ros2/components.py`

测试证据：
- `tests/interfaces/test_ros2_bridge.py`

### P1.3 ROS2 示例包
已完成内容：
1. 提供 install-space 示例包 `examples/ros2/omninav_ros2_examples`
2. 支持 `ros2 run/launch` 启动 Nav2/RViz 示例

源码证据：
- `examples/ros2/omninav_ros2_examples/package.xml`
- `examples/ros2/omninav_ros2_examples/setup.py`
- `examples/ros2/omninav_ros2_examples/launch/nav2_full_stack.launch.py`

---

## 5. P2 插件功能：示例工程化能力
状态：已完成

### P2.1 示例脚本自治化
已完成内容：
1. 移除统一 demo runner
2. 每个 `examples/*.py` 保留自身实例化与业务逻辑
3. 通过 `configs/demo/*.yaml` 做配置组合

源码证据：
- `examples/01_teleop_go2.py`
- `examples/02_teleop_go2w.py`
- `examples/03_lidar_visualization.py`
- `examples/04_camera_visualization.py`
- `examples/05_waypoint_navigation.py`
- `examples/06_inspection_task.py`
- `configs/demo/*.yaml`

测试证据：
- `tests/examples/test_examples_smoke.py`
- `tests/examples/test_demo_config_contract.py`

### P2.2 smoke 降载机制
已完成内容：
1. 统一 `--smoke-fast` 参数策略
2. 全示例默认 smoke 可执行

源码证据：
- `examples/06_inspection_task.py`
- 其他 `examples/*.py` 中 test/smoke 参数

测试证据：
- `tests/examples/test_examples_smoke.py`

### P2.3 Getting Started 教学入口（算法模板 + 任务模板）
已完成内容：
1. 新增 `examples/getting_started/run_getting_started.py`，GUI 直接展示 Task + Global + Local 三层协作
2. 新增 `examples/getting_started/algorithm_template.py`，包含 Global/Local 原子算法模板
3. 新增 `examples/getting_started/task_template.py`，提供最小巡检任务定义模板
4. GUI 增强新手信息展示：传感器分辨率/FOV/rays/range、规划器状态、任务指标、overrides 快照

源码证据：
- `examples/getting_started/run_getting_started.py`
- `examples/getting_started/catalog.py`
- `examples/getting_started/algorithm_template.py`
- `examples/getting_started/task_template.py`
- `examples/getting_started/README.md`
- `configs/demo/getting_started.yaml`

测试证据：
- `tests/examples/test_getting_started_catalog.py`
- `tests/examples/test_getting_started_overrides.py`
- `tests/examples/test_getting_started_smoke.py`
- `tests/examples/test_examples_smoke.py`

---

## 6. 说明

1. 未完成事项不在本文件记录，统一见 `TASK.md`。
2. 本文所有条目均与 `IMPLEMENTATION_PLAN.md` 的 `M*/P*` 结构对应。
