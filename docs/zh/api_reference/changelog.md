# Changelog（更新记录）

## 0.1.0（当前基线）

当前仓库已落地能力：

1. 运行时与架构基线
- Registry 驱动组件构建
- 统一生命周期管理
- Batch-first 数据契约
- Runtime 主循环编排

2. 导航与巡检管线
- 全局+局部规划 pipeline
- Waypoint 与 Inspection 任务
- 巡检指标基础实现

3. 场景与地图基线
- 静态障碍场景加载
- 占据图服务与 grid path 全局规划接入

4. 接口与示例
- Python API (`OmniNavEnv`)
- ROS2 bridge 示例
- getting_started GUI 与 smoke 示例

源码证据：
- [omninav/core](https://github.com/Royalvice/OmniNav/tree/main/omninav/core)
- [omninav/algorithms](https://github.com/Royalvice/OmniNav/tree/main/omninav/algorithms)
- [omninav/evaluation](https://github.com/Royalvice/OmniNav/tree/main/omninav/evaluation)
- [examples](https://github.com/Royalvice/OmniNav/tree/main/examples)
- [tests](https://github.com/Royalvice/OmniNav/tree/main/tests)
