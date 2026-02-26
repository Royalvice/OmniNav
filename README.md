<p align="center">
  <img src="docs/_static/logo.png" alt="OmniNav" width="400">
</p>

<h1 align="center">OmniNav</h1>

<p align="center">
  <strong>Navigation Simulation Platform (General Navigation Base + Inspection-First Workflow)</strong>
</p>

<p align="center">
  <a href="https://github.com/Royalvice/OmniNav">
    <img src="https://img.shields.io/github/stars/Royalvice/OmniNav?style=social" alt="GitHub stars">
  </a>
  <a href="https://github.com/Royalvice/OmniNav/issues">
    <img src="https://img.shields.io/github/issues/Royalvice/OmniNav" alt="GitHub Issues">
  </a>
  <a href="https://github.com/Royalvice/OmniNav/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="License">
  </a>
</p>

---

## 中文

OmniNav 是基于 Genesis 的导航仿真平台，当前聚焦：
- 通用导航原子能力（PointNav/ObjectNav/Waypoint）
- 巡检任务链路（Inspection Task + 评测）
- 复杂静态场景与可通行性验证

### 快速入口

1. 安装与环境：`INSTALL.md`
2. 纯 Python 新手入口：`examples/getting_started/run_getting_started.py`
3. ROS2/Nav2 入口：`examples/ros2/omninav_ros2_examples/README.md`
4. 在线文档（GitHub Pages）：<https://royalvice.github.io/OmniNav/>

### 安装（请以 INSTALL.md 为准）

```bash
git clone https://github.com/Royalvice/OmniNav.git
cd OmniNav

# Git LFS + submodules
git lfs install
git submodule update --init external/Genesis
git submodule update --init external/genesis_ros
git lfs pull
```

### 常用示例

```bash
# 纯 Python GUI（推荐新手）
python -m examples.getting_started.run_getting_started

# Waypoint demo
python examples/05_waypoint_navigation.py

# Inspection demo
python examples/06_inspection_task.py
```

### ROS2 / Nav2 示例

```bash
source ~/omninav_ros_env/bin/activate
source /opt/ros/humble/setup.bash

~/omninav_ros_env/bin/python -m colcon build --symlink-install --packages-select omninav_ros2_examples
source install/setup.bash

ros2 launch omninav_ros2_examples nav2_full_stack.launch.py
```

---

## English

OmniNav is a Genesis-based navigation simulation platform focused on:
- Atomic navigation capabilities (PointNav/ObjectNav/Waypoint)
- Inspection workflow (Inspection Task + evaluation)
- Complex static scenes and traversability validation

### Quick Links

1. Installation and environments: `INSTALL.md`
2. Pure Python beginner entry: `examples/getting_started/run_getting_started.py`
3. ROS2/Nav2 entry: `examples/ros2/omninav_ros2_examples/README.md`
4. Full docs (GitHub Pages): <https://royalvice.github.io/OmniNav/>

### Installation (source of truth: INSTALL.md)

```bash
git clone https://github.com/Royalvice/OmniNav.git
cd OmniNav

# Git LFS + submodules
git lfs install
git submodule update --init external/Genesis
git submodule update --init external/genesis_ros
git lfs pull
```

### Common demos

```bash
# Pure Python GUI (recommended for first-time users)
python -m examples.getting_started.run_getting_started

# Waypoint demo
python examples/05_waypoint_navigation.py

# Inspection demo
python examples/06_inspection_task.py
```

### ROS2 / Nav2 demo

```bash
source ~/omninav_ros_env/bin/activate
source /opt/ros/humble/setup.bash

~/omninav_ros_env/bin/python -m colcon build --symlink-install --packages-select omninav_ros2_examples
source install/setup.bash

ros2 launch omninav_ros2_examples nav2_full_stack.launch.py
```

---

## License

OmniNav is licensed under the [Apache-2.0 License](LICENSE).

## Citation

```bibtex
@misc{OmniNav,
  author = {OmniNav Contributors},
  title = {OmniNav: Navigation Simulation Platform},
  year = {2026},
  url = {https://github.com/Royalvice/OmniNav}
}
```
