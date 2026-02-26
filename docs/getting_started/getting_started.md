# Getting Started (Pure Python)

Primary entrypoint:
- [examples/getting_started/run_getting_started.py](https://github.com/Royalvice/OmniNav/blob/main/examples/getting_started/run_getting_started.py)

Reference guide:
- [examples/getting_started/README.md](https://github.com/Royalvice/OmniNav/blob/main/examples/getting_started/README.md)

```{note}
This page is Python-only. For ROS2/Nav2 integration, use [ROS2 / Nav2 Demos](ros2_nav2_demos).
```

## 1. Run

```bash
python -m examples.getting_started.run_getting_started
```

Performance-focused run:

```bash
python -m examples.getting_started.run_getting_started --show-viewer --tick-ms 1 --ui-refresh-ms 200
```

Smoke run:

```bash
python -m examples.getting_started.run_getting_started --test-mode --smoke-fast --max-steps 80 --no-show-viewer
```

## 2. What the GUI shows

- Task switching: waypoint / inspection
- Global planner switching: global_sequential / global_route_opt / global_grid_path
- Local planner: dwa_planner
- Sensor profile switching and metadata
- Minimap route editing and occupancy hints

## 3. Extension templates

- Algorithm template: [examples/getting_started/algorithm_template.py](https://github.com/Royalvice/OmniNav/blob/main/examples/getting_started/algorithm_template.py)
- Task template: [examples/getting_started/task_template.py](https://github.com/Royalvice/OmniNav/blob/main/examples/getting_started/task_template.py)

Target module paths:
- Algorithms: [omninav/algorithms](https://github.com/Royalvice/OmniNav/tree/main/omninav/algorithms)
- Tasks: [omninav/evaluation/tasks](https://github.com/Royalvice/OmniNav/tree/main/omninav/evaluation/tasks)

## 4. Quick troubleshooting

- `ModuleNotFoundError: No module named 'examples'`:
  run with `python -m examples.getting_started.run_getting_started` from repo root.
- Viewer scene recreate error in GUI:
  avoid rapid repeated rebuild/reset clicks; wait for previous environment close cycle.
- Goal convergence oscillation:
  tune `local_dwa` config and verify occupancy map alignment for the selected scene.
