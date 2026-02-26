# Core Runtime API（核心运行时）

## 覆盖范围

- [omninav/core/runtime.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/core/runtime.py)
- [omninav/core/simulation_manager.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/core/simulation_manager.py)
- [omninav/core/registry.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/core/registry.py)
- [omninav/core/lifecycle.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/core/lifecycle.py)
- [omninav/core/types.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/core/types.py)
- [omninav/core/map/types.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/core/map/types.py)
- [omninav/core/map/service.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/core/map/service.py)
- [omninav/core/map/facade.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/core/map/facade.py)

## 关键类

### `SimulationRuntime`

源码： [runtime.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/core/runtime.py)

职责：
- 构建并持有仿真组件
- reset/step 编排
- 任务反馈与结果汇总
- 可选地图服务注入与楼层信息传播

主要方法：
- `build()`
- `reset()`
- `step(actions=None)`
- `get_result()`

### `GenesisSimulationManager`

源码： [simulation_manager.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/core/simulation_manager.py)

职责：
- Genesis 场景创建
- 场景实体与传感器加载
- 仿真步进与时间管理

### Registry 与生命周期

- Registry： [`Registry`, `BuildContext`](https://github.com/Royalvice/OmniNav/blob/main/omninav/core/registry.py)
- 生命周期： [`LifecycleState`, `LifecycleMixin`](https://github.com/Royalvice/OmniNav/blob/main/omninav/core/lifecycle.py)

## 数据契约

源码： [types.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/core/types.py)

核心类型：
- `Observation`（batch-first）
- `Action`（`cmd_vel` 形状 `(B,3)`）
- `TaskResult`
- `RobotState`, `SensorData`

校验函数：
- `validate_batch_shape(...)`

## 地图服务

源码：
- [types.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/core/map/types.py)
- [service.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/core/map/service.py)
- [facade.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/core/map/facade.py)

能力：
- 多楼层占据图
- world/grid 坐标转换
- connector 图查询
- 按 XY 查询当前楼层
