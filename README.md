<p align="center">
  <img src="docs/_static/logo.png" alt="OmniNav" width="400">
</p>

<h1 align="center">OmniNav</h1>

<p align="center">
  <strong>Navigation Simulation Platform for Embodied AI</strong>
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

English | [中文文档](README_CN.md)

## What Is OmniNav?

OmniNav is a Genesis-based navigation simulation platform.

It is built as:
- 🧱 **A general navigation capability base** for PointNav / ObjectNav / Waypoint tasks.
- 🧭 **An inspection-first workflow platform** where navigation is atomic capability and inspection is business loop.
- 🧪 **A reproducible evaluation framework** with unified Observation / Action / TaskResult contracts.

Current `v0.1` focus:
- ✅ Coverage & Traversability
- ✅ Anomaly & Semantics (non-thermal mandatory baseline)
- ✅ Complex static scenes & traversability

## Key Capabilities

- ⚙️ **Layered architecture**: Core / Assets / Robots / Sensors / Algorithms / Tasks / Interfaces
- 📦 **Registry-driven components**: robot, sensor, locomotion, algorithm, task, metric
- 🔁 **Batch-first runtime**: `(B, ...)` data flow by default
- 🧩 **Config-driven experiments**: Hydra + OmegaConf overrides
- 🤖 **Inspection task pipeline**: `Observation -> Algorithm -> Locomotion -> Task`
- 🌉 **ROS2 bridge support**: RViz sensor demo and Nav2 closed-loop demo

## Quick Links

1. Installation (source of truth): [`INSTALL.md`](INSTALL.md)
2. Getting Started (pure Python): [`examples/getting_started/run_getting_started.py`](examples/getting_started/run_getting_started.py)
3. ROS2/Nav2 demos: [`examples/ros2/omninav_ros2_examples/README.md`](examples/ros2/omninav_ros2_examples/README.md)
4. Full documentation (GitHub Pages): <https://royalvice.github.io/OmniNav/>

## Quick Start

```bash
git clone https://github.com/Royalvice/OmniNav.git
cd OmniNav

git lfs install
git submodule update --init external/Genesis
git submodule update --init external/genesis_ros
git lfs pull

python -m examples.getting_started.run_getting_started
```

## Common Demos

```bash
python examples/01_teleop_go2.py
python examples/02_teleop_go2w.py
python examples/03_lidar_visualization.py
python examples/04_camera_visualization.py
python examples/05_waypoint_navigation.py
python examples/06_inspection_task.py
```

## License

OmniNav is licensed under the [Apache-2.0 License](LICENSE).

## Citation

```bibtex
@misc{OmniNav,
  author = {OmniNav Contributors},
  title = {OmniNav: Navigation Simulation Platform for Embodied AI},
  year = {2026},
  url = {https://github.com/Royalvice/OmniNav}
}
```
