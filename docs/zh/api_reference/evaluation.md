# Evaluation API

## 覆盖范围

- [omninav/evaluation/base.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/evaluation/base.py)
- [omninav/evaluation/tasks/waypoint_task.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/evaluation/tasks/waypoint_task.py)
- [omninav/evaluation/tasks/inspection_task.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/evaluation/tasks/inspection_task.py)
- [omninav/evaluation/metrics/inspection_metrics.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/evaluation/metrics/inspection_metrics.py)

## 基类接口

### `TaskBase`

源码： [base.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/evaluation/base.py)

职责：
- 任务 reset/step
- 向算法下发 task specification
- 生成 task result

### `MetricBase`

源码： [base.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/evaluation/base.py)

职责：
- 指标 update/compute/reset

## 内置任务

- `WaypointTask`: [waypoint_task.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/evaluation/tasks/waypoint_task.py)
- `InspectionTask`: [inspection_task.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/evaluation/tasks/inspection_task.py)

## 内置巡检指标

- `CoverageRate`
- `DetectionRate`
- `InspectionTime`
- `SafetyScore`

源码： [inspection_metrics.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/evaluation/metrics/inspection_metrics.py)

## 任务配置参考

- [configs/task/waypoint.yaml](https://github.com/Royalvice/OmniNav/blob/main/configs/task/waypoint.yaml)
- [configs/task/inspection.yaml](https://github.com/Royalvice/OmniNav/blob/main/configs/task/inspection.yaml)
- [configs/task/point_nav.yaml](https://github.com/Royalvice/OmniNav/blob/main/configs/task/point_nav.yaml)
- [configs/task/object_nav.yaml](https://github.com/Royalvice/OmniNav/blob/main/configs/task/object_nav.yaml)
