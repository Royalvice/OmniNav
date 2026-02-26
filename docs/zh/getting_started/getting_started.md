# Getting Started（纯 Python）

主要入口：
- [examples/getting_started/run_getting_started.py](https://github.com/Royalvice/OmniNav/blob/main/examples/getting_started/run_getting_started.py)

参考说明：
- [examples/getting_started/README.md](https://github.com/Royalvice/OmniNav/blob/main/examples/getting_started/README.md)

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

## 2. GUI 能力

- 任务切换：waypoint / inspection
- 全局规划切换：global_sequential / global_route_opt / global_grid_path
- 局部规划：dwa_planner
- 传感器 profile 切换与元信息查看
- minimap 路径编辑

## 3. 扩展模板

- 算法模板： [examples/getting_started/algorithm_template.py](https://github.com/Royalvice/OmniNav/blob/main/examples/getting_started/algorithm_template.py)
- 任务模板： [examples/getting_started/task_template.py](https://github.com/Royalvice/OmniNav/blob/main/examples/getting_started/task_template.py)

目标模块：
- 算法： [omninav/algorithms](https://github.com/Royalvice/OmniNav/tree/main/omninav/algorithms)
- 任务： [omninav/evaluation/tasks](https://github.com/Royalvice/OmniNav/tree/main/omninav/evaluation/tasks)
