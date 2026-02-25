# OmniNav Getting Started (Pure Python, Task + Global + Local)

This folder is the beginner entrypoint for OmniNav without ROS2.

## 1. Run

```bash
python -m examples.getting_started.run_getting_started
```

Performance-focused launch:

```bash
python -m examples.getting_started.run_getting_started --show-viewer --tick-ms 1 --ui-refresh-ms 200
```

Smoke mode:

```bash
python -m examples.getting_started.run_getting_started --test-mode --smoke-fast --max-steps 80 --no-show-viewer
```

## 2. FPS Notes (Important)

`getting_started` has two FPS layers:

- simulation step FPS (`env.step` throughput)
- GUI refresh FPS (Tk text + minimap drawing)

Low FPS is usually caused by sensor rendering, not waypoint logic. In particular:

- `lidar_camera` uses camera rendering, which is expensive
- enabling Genesis viewer adds another rendering pipeline
- GUI text panel is intentionally refreshed at a lower rate for stability

Current defaults are tuned for responsiveness while keeping the viewer on:

- viewer default: on
- camera default in this demo: `256x144`, `camera_types=[rgb]`, `update_every_n_steps=2~3`
- lidar default in this demo: `update_every_n_steps=2`

If you need higher speed for debugging:

- use `sensor_profile=lidar_only` or `sensor_profile=none`
- optionally use `--no-show-viewer`
- keep `--tick-ms 1`, increase `--ui-refresh-ms` (e.g. `300`)

## 3. What This GUI Demonstrates

- Task definition and switching: `waypoint` / `inspection`
- Global planner switching: `global_sequential` / `global_route_opt`
- Local planner switching: `dwa_planner`
- Sensor profile switching and metadata inspection
- Route editing from minimap (waypoint list)
- Occupancy-aware minimap:
  - static occupancy from scene obstacles
  - live occupancy points from lidar ranges
- Runtime introspection:
  - task/planner states
  - robot state
  - task metrics
  - sensor metadata and runtime summary
  - exact Hydra overrides snapshot

Minimap controls:

- Left click: add waypoint
- Right click: remove last waypoint
- Middle click: clear all waypoints

## 4. Rebuild vs Reset (Important)

- `Start / Rebuild`
  - If config/route changed: rebuild environment with new overrides
  - If unchanged: reset current environment (reuse scene)
- `Reset Env`
  - Always calls `env.reset()` on current scene
  - Does not create a new scene
  - Fails immediately if environment is not initialized

This avoids Genesis interactive-viewer multi-scene conflicts.

## 5. Spawn Position

`getting_started` reads spawn from scene config:

- `configs/scene/<scene>.yaml -> spawn.default`
- optional `spawn.candidates`

For `complex_flat_obstacles`, default spawn is set away from the center pillar.
If spawn config is missing, GUI fails fast with explicit error.

## 6. Best Practice: Add A New Algorithm

Use `algorithm_template.py`.

1. Copy template into `omninav/algorithms/`
2. Register with `@ALGORITHM_REGISTRY.register(...)`
3. Add config in `configs/algorithm/`
4. Select it in GUI and validate with tests

Important contract:

- Always return `cmd_vel` with shape `(B, 3)`
- Keep logic batch-first
- Use shape validation on critical paths

## 7. Best Practice: Define A Minimal Inspection Task

Use `task_template.py`.

1. Copy template into `omninav/evaluation/tasks/`
2. Register with `@TASK_REGISTRY.register(...)`
3. Add task config in `configs/task/`
4. Select `task=<your_task>` and run with existing planners

Task responsibilities:

- define task objective/constraints
- provide TaskSpec (`goal_set`, budgets, thresholds)
- evaluate and terminate task

Task should **not** contain path planning logic.

## 8. Architecture Reminder

- `Task` only defines/dispatches/evaluates mission
- `GlobalPlanner` schedules goals
- `LocalPlanner` computes obstacle-aware control
- `AlgorithmPipeline` composes global + local
