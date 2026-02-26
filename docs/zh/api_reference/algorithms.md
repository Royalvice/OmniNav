# Algorithms API（算法）

## 覆盖范围

- [omninav/algorithms/base.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/algorithms/base.py)
- [omninav/algorithms/pipeline.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/algorithms/pipeline.py)
- [omninav/algorithms/global_base.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/algorithms/global_base.py)
- [omninav/algorithms/global_sequential.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/algorithms/global_sequential.py)
- [omninav/algorithms/global_route_opt.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/algorithms/global_route_opt.py)
- [omninav/algorithms/global_grid_path.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/algorithms/global_grid_path.py)
- [omninav/algorithms/local_planner.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/algorithms/local_planner.py)

## 架构

算法栈拆分为：
- 全局规划：目标调度 / 路径生成
- 局部规划：避障控制命令生成
- Pipeline：全局 + 局部组合执行

## 关键类

### `AlgorithmBase`

源码： [base.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/algorithms/base.py)

定义：
- `reset(task_info)`
- `step(obs)`
- `is_done`, `info`

### `AlgorithmPipeline`

源码： [pipeline.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/algorithms/pipeline.py)

职责：
- 联合执行 global/local planner
- 将全局目标与路径提示传给局部规划

### 全局规划器

- `SequentialGlobalPlanner`: [global_sequential.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/algorithms/global_sequential.py)
- `RouteOptimizedGlobalPlanner`: [global_route_opt.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/algorithms/global_route_opt.py)
- `GridPathGlobalPlanner`: [global_grid_path.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/algorithms/global_grid_path.py)

### 局部规划器

- `DWAPlanner`: [local_planner.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/algorithms/local_planner.py)

## 配置参考

- [configs/algorithm/pipeline_default.yaml](https://github.com/Royalvice/OmniNav/blob/main/configs/algorithm/pipeline_default.yaml)
- [configs/algorithm/global_sequential.yaml](https://github.com/Royalvice/OmniNav/blob/main/configs/algorithm/global_sequential.yaml)
- [configs/algorithm/global_route_opt.yaml](https://github.com/Royalvice/OmniNav/blob/main/configs/algorithm/global_route_opt.yaml)
- [configs/algorithm/global_grid_path.yaml](https://github.com/Royalvice/OmniNav/blob/main/configs/algorithm/global_grid_path.yaml)
- [configs/algorithm/local_dwa.yaml](https://github.com/Royalvice/OmniNav/blob/main/configs/algorithm/local_dwa.yaml)
