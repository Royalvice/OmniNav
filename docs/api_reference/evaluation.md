# Evaluation API

## Scope

- [omninav/evaluation/base.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/evaluation/base.py)
- [omninav/evaluation/tasks/waypoint_task.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/evaluation/tasks/waypoint_task.py)
- [omninav/evaluation/tasks/inspection_task.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/evaluation/tasks/inspection_task.py)
- [omninav/evaluation/metrics/inspection_metrics.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/evaluation/metrics/inspection_metrics.py)

## Base interfaces

### `TaskBase`

Source: [base.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/evaluation/base.py)

Responsibilities:
- task reset/step
- task specification dispatch to algorithms
- task result generation

### `MetricBase`

Source: [base.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/evaluation/base.py)

Responsibilities:
- metric update/compute/reset

## Built-in tasks

- `WaypointTask`: [waypoint_task.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/evaluation/tasks/waypoint_task.py)
- `InspectionTask`: [inspection_task.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/evaluation/tasks/inspection_task.py)

## Built-in inspection metrics

- `CoverageRate`
- `DetectionRate`
- `InspectionTime`
- `SafetyScore`

Source: [inspection_metrics.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/evaluation/metrics/inspection_metrics.py)

## Task configuration references

- [configs/task/waypoint.yaml](https://github.com/Royalvice/OmniNav/blob/main/configs/task/waypoint.yaml)
- [configs/task/inspection.yaml](https://github.com/Royalvice/OmniNav/blob/main/configs/task/inspection.yaml)
- [configs/task/point_nav.yaml](https://github.com/Royalvice/OmniNav/blob/main/configs/task/point_nav.yaml)

```{note}
`object_nav` task config is not shipped in the current baseline. Use the three task configs above as the source of truth.
```
