# Getting Started (Pure Python)

<div class="lang-zh">

本页对应：[`examples/getting_started/run_getting_started.py`](https://github.com/Royalvice/OmniNav/blob/main/examples/getting_started/run_getting_started.py)。

## 1. 运行

```bash
python -m examples.getting_started.run_getting_started
```

性能优先：

```bash
python -m examples.getting_started.run_getting_started --show-viewer --tick-ms 1 --ui-refresh-ms 200
```

smoke：

```bash
python -m examples.getting_started.run_getting_started --test-mode --smoke-fast --max-steps 80 --no-show-viewer
```

## 2. 你可以在 GUI 中做什么

1. 切换任务：`waypoint` / `inspection`
2. 切换全局规划：`global_sequential` / `global_route_opt` / `global_grid_path`
3. 切换局部规划：`dwa_planner`
4. 切换传感器 profile，查看传感器元数据
5. 在 minimap 上编辑 waypoint

代码参考：
- [`examples/getting_started/run_getting_started.py`](https://github.com/Royalvice/OmniNav/blob/main/examples/getting_started/run_getting_started.py)
- [`examples/getting_started/catalog.py`](https://github.com/Royalvice/OmniNav/blob/main/examples/getting_started/catalog.py)

## 3. 扩展模板（推荐新手）

1. 新算法模板：[`examples/getting_started/algorithm_template.py`](https://github.com/Royalvice/OmniNav/blob/main/examples/getting_started/algorithm_template.py)
2. 新任务模板：[`examples/getting_started/task_template.py`](https://github.com/Royalvice/OmniNav/blob/main/examples/getting_started/task_template.py)

建议先从模板复制到正式模块：
- 算法：`omninav/algorithms/`
- 任务：`omninav/evaluation/tasks/`

## 4. 常见问题

1. FPS 低：优先降低相机分辨率、传感器刷新率，或使用 `--no-show-viewer`。
2. viewer 报多场景冲突：优先使用“重置当前环境”而不是重复创建新 scene。
3. 交互不响应：确认窗口焦点和键位映射（`W/S/Q/E/Space/Esc`）。

详细说明见：[`examples/getting_started/README.md`](https://github.com/Royalvice/OmniNav/blob/main/examples/getting_started/README.md)。

</div>

<div class="lang-en">

This page maps to: [`examples/getting_started/run_getting_started.py`](https://github.com/Royalvice/OmniNav/blob/main/examples/getting_started/run_getting_started.py).

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

## 2. What you can do in the GUI

1. Switch tasks: `waypoint` / `inspection`
2. Switch global planners: `global_sequential` / `global_route_opt` / `global_grid_path`
3. Switch local planner: `dwa_planner`
4. Switch sensor profiles and inspect metadata
5. Edit waypoint routes directly on minimap

Code references:
- [`examples/getting_started/run_getting_started.py`](https://github.com/Royalvice/OmniNav/blob/main/examples/getting_started/run_getting_started.py)
- [`examples/getting_started/catalog.py`](https://github.com/Royalvice/OmniNav/blob/main/examples/getting_started/catalog.py)

## 3. Extension templates (recommended)

1. Algorithm template: [`examples/getting_started/algorithm_template.py`](https://github.com/Royalvice/OmniNav/blob/main/examples/getting_started/algorithm_template.py)
2. Task template: [`examples/getting_started/task_template.py`](https://github.com/Royalvice/OmniNav/blob/main/examples/getting_started/task_template.py)

Recommended destination modules:
- Algorithms: `omninav/algorithms/`
- Tasks: `omninav/evaluation/tasks/`

## 4. Common issues

1. Low FPS: reduce camera resolution / sensor update rate, or use `--no-show-viewer`.
2. Viewer multi-scene conflict: prefer resetting current environment over rebuilding scenes repeatedly.
3. Input not responding: check window focus and key mapping (`W/S/Q/E/Space/Esc`).

See [`examples/getting_started/README.md`](https://github.com/Royalvice/OmniNav/blob/main/examples/getting_started/README.md) for full details.

</div>
