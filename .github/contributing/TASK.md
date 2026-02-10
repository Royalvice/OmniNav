# OmniNav 开发任务清单 (Task List)

本文件追踪 OmniNav 项目的开发进度。

## Core v0.1.0 重构与实现 (Active Phases) ✅

### Phase 1: Foundation — 数据契约与基础设施 ✅
- [x] 1.1 创建 `omninav/core/types.py` — 所有 TypedDict 数据契约
- [x] 1.2 创建 `omninav/core/hooks.py` — Event/Hook 系统
- [x] 1.3 创建 `omninav/core/lifecycle.py` — 组件生命周期状态机
- [x] 1.4 重构 `omninav/core/registry.py` — 添加 BuildContext
- [x] 1.5 测试: `tests/core/test_types.py`, `tests/core/test_hooks.py`, `test_lifecycle.py`

### Phase 2: Robot 层重构 ✅
- [x] 2.1 重构 `omninav/robots/base.py` — 删除 apply_command, 添加生命周期
- [x] 2.2 更新 `omninav/robots/go2.py`
- [x] 2.3 更新 `omninav/robots/go2w.py`
- [x] 2.4 测试: `tests/robots/test_robot_base.py`

### Phase 3: Sensor 层解耦 ✅
- [x] 3.1 重构 `omninav/sensors/base.py` — 解耦 scene/robot
- [x] 3.2 更新 `omninav/sensors/lidar.py`
- [x] 3.3 更新 `omninav/sensors/camera.py`
- [x] 3.4 更新 `omninav/sensors/raycaster_depth.py`
- [x] 3.5 测试: `tests/sensors/test_sensors.py`

### Phase 4: Locomotion 层净化 ✅
- [x] 4.1 重构 `omninav/locomotion/base.py` — 添加 bind_sensors, step(cmd_vel, obs=None)
- [x] 4.2 重构 `omninav/locomotion/kinematic_controller.py` — 移除直接 import genesis
- [x] 4.3 更新 `omninav/locomotion/wheel_controller.py`
- [x] 4.4 重构 `omninav/locomotion/rl_controller.py`
- [x] 4.5 测试: `tests/locomotion/test_locomotion.py`

### Phase 5: Algorithm 层增强 ✅
- [x] 5.1 重构 `omninav/algorithms/base.py` — 使用 Observation TypedDict
- [x] 5.2 创建 `omninav/algorithms/pipeline.py` — AlgorithmPipeline
- [x] 5.3 创建 `omninav/algorithms/local_planner.py` — LocalPlannerBase + DWA
- [x] 5.4 创建 `omninav/algorithms/inspection_planner.py` — InspectionPlanner
- [x] 5.5 测试: `tests/algorithms/test_pipeline.py`

### Phase 6: Evaluation 层 — 巡检特化 ✅
- [x] 6.1 更新 `omninav/evaluation/base.py` — 使用 Observation TypedDict
- [x] 6.2 创建 `omninav/evaluation/tasks/inspection_task.py`
- [x] 6.3 创建 `omninav/evaluation/metrics/inspection_metrics.py`
- [x] 6.4 测试: `tests/evaluation/test_inspection.py`

### Phase 7: Interface 层重构 ✅
- [x] 7.1 创建 `omninav/core/runtime.py` — SimulationRuntime 编排器
- [x] 7.2 重构 `omninav/interfaces/python_api.py` — 轻量 OmniNavEnv
- [x] 7.3 创建 `omninav/interfaces/gym_wrapper.py` — OmniNavGymWrapper
- [x] 7.4 重构 `omninav/interfaces/ros2/bridge.py` — 双向通信桥接
- [x] 7.5 测试: `tests/interfaces/test_env.py`

### Phase 8: 配置与示例 ✅
- [x] 8.1 适配 `configs/config.yaml` — 迁移至分层 Hydra 系统
- [x] 8.2 创建 `examples/run_inspection.py` — 全流程巡检演示
- [x] 8.3 验证: 运行示例并确认指标输出

### Phase 9: 验证与文档沉淀 🔄
- [x] 9.1 集成测试: `tests/integration/test_full_pipeline.py`
- [x] 9.2 全面更新 `.github/contributing/` 文档库
- [ ] 9.3 完善 `docs/` 用户手册
- [ ] 9.4 全流程回顾与代码冻结

---

## 历史阶段 (Foundational Work & Demos) ✅

### Phase A: Pure Game-Style Kinematic Controller
- [x] 核心实现：预烘焙动画系统、100Hz 插值
- [x] 性能优化：耗时从 10ms 降至 0.1ms
- [x] 验证：Go2 稳定行走上楼梯

### Phase C: Demo Enhancements
- [x] 修复地面渲染与障碍物环
- [x] Lidar 射线可视化增强
- [x] Go2w 遥控演示

### Phase E: IK Locomotion Jitter Fix
- [x] 解决世界坐标锁定下的抖动问题
- [x] 引入状态机切换 (Stand/Walk)

### Phase G: Enhanced Navigation Demo
- [x] Minimap 实时轨迹绘制
- [x] 点到点导航状态机

---

## 待开始阶段 (Future Roadmap) ⏳

- [ ] **Phase 10: VLA 接入** - 大模型视觉语言策略集成
- [ ] **Phase 11: 复杂地形生成** - 基于噪声的随机地形资产库
- [ ] **Phase 12: 集群仿真** - 1000+ 环境下的多机协同评测
