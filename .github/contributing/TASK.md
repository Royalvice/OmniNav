# OmniNav 开发任务清单 (Task List)

本文件追踪 OmniNav `v0.1` 阶段（架构重构后到需求对齐与工程化收敛）的任务。

## 0. 当前状态快照 (2026-02)

### 已完成能力 (代码已存在)
- [x] Core 分层、Registry、Hook、Runtime 主循环
- [x] Go2 / Go2w 机器人与基础传感器挂载
- [x] 巡检任务链路：`Observation -> Algorithm -> Locomotion -> Task`
- [x] Hydra 配置体系与 `OmniNavEnv.from_config(...)`
- [x] 单元测试 + 基础集成测试骨架

### 与需求文档的关键差距 (需补齐)
- [x] **Batch-First 一致性**：主链路统一为 `(B,3)`，兼容输入 `(3,)` 并在 runtime 归一化
- [x] **强类型单一真源**：`TaskResult` 统一到 `omninav/core/types.py`
- [x] **Lifecycle 全覆盖**：Algorithm / Task 已纳入生命周期管理
- [ ] **ROS2 工程化闭环**：`/tf` 与批量语义已补齐，Nav2 对接仍需补充验证
- [ ] **评测体系完整性**：PointNav/ObjectNav + SR/SPL/Collision 未完整落地
- [ ] **程序化场景与复杂度评估**：尚未形成可复现实验流水线

---

## 1. P0 收敛阶段（正在进行）

### Phase 9A: 架构一致性修复 (P0 - Critical)
- [x] 9A.1 统一 `Action.cmd_vel` 契约为 `(B, 3)`，清理 runtime/algo/loco 单样本分支
- [x] 9A.2 统一 `TaskResult` 定义到 `omninav/core/types.py`，删除重复实现
- [x] 9A.3 `AlgorithmBase`、`TaskBase` 引入 `LifecycleMixin` 并补足状态迁移
- [x] 9A.4 在 Runtime 和关键层加入 batch shape 校验（`validate_batch_shape`）
- [x] 9A.5 补充回归测试：`tests/core/test_types.py`、`tests/interfaces/test_env.py`

### Phase 9B: Runtime 并行语义落地 (P0 - Critical)
- [x] 9B.1 明确 `num_envs` 与 `B` 的映射规则（单机器人/多机器人）
- [x] 9B.2 `SimulationRuntime.step` 支持批量 action 输入与批量 done 输出
- [x] 9B.3 `TaskBase.is_terminated` 升级为批量布尔数组接口
- [x] 9B.4 增加 `n_envs=4` 集成测试与基准脚本（Genesis 实测由开关控制）

### Phase 9C: ROS2 Bridge 可用性达标 (P0 - Critical)
- [x] 9C.1 完成 `/tf` 与 `/clock` 对齐，补齐 frame 约定文档
- [x] 9C.2 修正 `RobotState` 读取方式，统一 TypedDict 访问
- [ ] 9C.3 建立 Nav2 最小闭环样例（发布 + 订阅 + 时钟同步）
- [x] 9C.4 增加 `tests/interfaces/test_ros2_bridge.py` 的端到端断言

### Phase 9D: 文档冻结前清理 (P0 - Critical)
- [ ] 9D.1 完善 `docs/` 用户手册（安装、配置、运行、扩展）
- [ ] 9D.2 更新公开 API docstring 与最小示例
- [ ] 9D.3 发布前 checklist（测试、示例、配置、文档）并冻结 `v0.1`

---

## 2. P1 扩展阶段（需求高优先）

### Phase 10: 评测体系扩展 (P1 - High)
- [ ] 10.1 新增 `PointNavTask` 与 `ObjectNavTask`
- [ ] 10.2 指标库标准化：SR、SPL、Collision、Time Efficiency
- [ ] 10.3 任务/指标注册与 Hydra 配置模板补齐
- [ ] 10.4 对齐报告导出格式（json/csv）

### Phase 11: 资产与场景生成 (P1 - High)
- [ ] 11.1 多格式资产导入最小闭环（USD/GLB/OBJ）
- [ ] 11.2 程序化场景生成器（规则 + 随机化）
- [ ] 11.3 场景复杂度评估器（障碍密度/曲率/遮挡）
- [ ] 11.4 场景基准集与复现实验脚本

### Phase 12: VLA/VLN 接口预留 (P1 - High)
- [ ] 12.1 在 `Observation` 中规范语言字段与多模态输入格式
- [ ] 12.2 增加算法插件模板：视觉编码器 + 文本指令 + `cmd_vel`
- [ ] 12.3 提供最小 fake-policy demo 与 smoke test

---

## 3. P2 高级能力阶段

### Phase 13: Sim2Real 高级链路 (P2 - Medium)
- [ ] 13.1 场景重建导入（Gaussian Splatting / NeRF）
- [ ] 13.2 轨迹回放（ROSbag / 自定义格式）
- [ ] 13.3 参数标定工具（摩擦/阻尼/质量）

### Phase 14: 大规模并行与集群 (P2 - Medium)
- [ ] 14.1 Headless 模式批量运行
- [ ] 14.2 100+/1000+ env 吞吐统计与稳定性报告
- [ ] 14.3 多机任务编排与结果聚合

---

## 4. 验收标准 (Definition of Done)

- [x] 所有新增能力具备 `n_envs=1` 与 `n_envs=4` 测试（Genesis 实测默认可跳过，开关启用）
- [x] 接口文档、实现、测试三者一致（无契约分叉）
- [x] 每个 Phase 至少有一个可运行 Demo
- [x] 关键流程在 `examples/` 与 `docs/` 中有可复现指令
