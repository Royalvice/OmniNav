<p align="center">
  <img src="docs/_static/logo.png" alt="OmniNav" width="400">
</p>

<h1 align="center">OmniNav</h1>

<p align="center">
  <strong>面向 Embodied AI 的导航仿真平台</strong>
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

[English](README.md) | 中文

## OmniNav 是什么？

OmniNav 是一个基于 Genesis 的导航仿真平台。

其定位是：
- 🧱 **通用导航原子能力底座**：支持 PointNav / ObjectNav / Waypoint 等任务。
- 🧭 **巡检优先工作流平台**：导航是原子能力，巡检是业务闭环。
- 🧪 **可复现实验评测框架**：统一 Observation / Action / TaskResult 数据契约。

当前 `v0.1` 聚焦：
- ✅ 覆盖率与遍历性
- ✅ 异常与语义（非热力专用硬约束）
- ✅ 复杂静态场景与可通行性

## 关键能力

- ⚙️ **分层架构**：Core / Assets / Robots / Sensors / Algorithms / Tasks / Interfaces
- 📦 **Registry 驱动组件构建**：robot、sensor、locomotion、algorithm、task、metric
- 🔁 **Batch-First 运行时**：默认 `(B, ...)` 数据流
- 🧩 **配置驱动实验**：Hydra + OmegaConf 覆盖机制
- 🤖 **巡检任务链路**：`Observation -> Algorithm -> Locomotion -> Task`
- 🌉 **ROS2 桥接能力**：支持 RViz 传感器示例与 Nav2 闭环示例

## 快速入口

1. 安装指南（唯一事实源）：[`INSTALL.md`](INSTALL.md)
2. 纯 Python 新手入口：[`examples/getting_started/run_getting_started.py`](examples/getting_started/run_getting_started.py)
3. ROS2/Nav2 示例：[`examples/ros2/omninav_ros2_examples/README.md`](examples/ros2/omninav_ros2_examples/README.md)
4. 在线文档（GitHub Pages）：<https://royalvice.github.io/OmniNav/>

## 快速开始

```bash
git clone https://github.com/Royalvice/OmniNav.git
cd OmniNav

git lfs install
git submodule update --init external/Genesis
git submodule update --init external/genesis_ros
git lfs pull

python -m examples.getting_started.run_getting_started
```

## 常用示例

```bash
python examples/01_teleop_go2.py
python examples/02_teleop_go2w.py
python examples/03_lidar_visualization.py
python examples/04_camera_visualization.py
python examples/05_waypoint_navigation.py
python examples/06_inspection_task.py
```

## 许可证

OmniNav 使用 [Apache-2.0 License](LICENSE)。

## 引用

```bibtex
@misc{OmniNav,
  author = {OmniNav Contributors},
  title = {OmniNav: Navigation Simulation Platform for Embodied AI},
  year = {2026},
  url = {https://github.com/Royalvice/OmniNav}
}
```
