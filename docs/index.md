# OmniNav

```{image} _static/logo.png
:align: center
:width: 300px
:class: homepage-logo
```

<p align="center">
  <strong>A General-Purpose Navigation Simulation Platform for Embodied AI</strong>
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

## What is OmniNav?

OmniNav is a general-purpose navigation simulation platform built on top of the [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) physics engine, designed for *Embodied AI / Robotics Navigation / Sim2Real* applications. It is simultaneously multiple things:

1. A **unified navigation benchmark** for evaluating navigation and obstacle avoidance algorithms.
2. A **plug-and-play algorithm framework** supporting both classical planners and neural network-based methods (VLA/VLN).
3. A **robot-agnostic platform** with built-in support for quadruped, wheeled, and humanoid robots.
4. A **high-fidelity simulation environment** with GPU-accelerated physics and photo-realistic rendering.

OmniNav aims to:

- **Simplify navigation research** by providing a ready-to-use benchmark with standardized evaluation metrics.
- **Bridge the Sim2Real gap** with high-fidelity physics simulation and optional ROS2 integration.
- **Accelerate algorithm development** with a modular, extensible architecture.

## Key Features

- 🚀 **High Performance**: Leverages Genesis engine for GPU-accelerated physics simulation (43M+ FPS on RTX 4090).
- 🔌 **Plug-and-Play Algorithms**: Easy integration of classical planners, RL policies, and VLA/VLN models.
- 📊 **Built-in Evaluation**: Pre-defined navigation tasks with standard metrics (SPL, Success Rate, Collision Rate).
- 🤖 **Multi-Robot Support**: Quadruped (Go2), wheeled robots, and extensible to other platforms.
- 🌐 **ROS2 Compatible**: Optional ROS2 bridge for Sim2Real deployment.
- 📦 **Scene Import**: Support for USD, GLB, OBJ, and custom scene assets.
- 🎨 **Photo-Realistic Rendering**: Ray-tracing based rendering for realistic visual observations.
- 🔧 **Configuration-Driven**: Hydra-based configuration for flexible experiment management.

## Quick Start

```python
from omninav import OmniNavEnv

# Create environment with default configuration
env = OmniNavEnv(config_path="configs")
obs = env.reset()

# Run navigation loop
while not env.is_done:
    action = env.algorithm.step(obs)  # Or use your own algorithm
    obs, info = env.step(action)

# Get evaluation results
result = env.get_result()
print(f"Success: {result.success}")
print(f"SPL: {result.metrics.get('spl', 0):.3f}")
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Interface Layer                              │
│              Python API (OmniNavEnv) / ROS2 Bridge               │
├─────────────────────────────────────────────────────────────────┤
│       Evaluation Layer        │      Algorithm Layer             │
│    Tasks & Metrics (SPL...)   │   (Pluggable Algorithms)         │
├─────────────────────────────────────────────────────────────────┤
│                     Locomotion Layer                             │
│           Kinematic Control / RL Policy (Extensible)             │
├─────────────────────────────────────────────────────────────────┤
│       Robot Layer             │       Asset Layer                │
│   Go2 / Go2w / Custom Robots  │   Scene Loaders (USD/GLB/OBJ)    │
├─────────────────────────────────────────────────────────────────┤
│                      Core Layer                                  │
│              Genesis Simulation Manager Wrapper                  │
└─────────────────────────────────────────────────────────────────┘
```

## Getting Started

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} 📥 Installation
:link: getting_started/installation
:link-type: doc

Get OmniNav up and running on your system.
:::

:::{grid-item-card} 🚀 First Simulation
:link: getting_started/first_simulation
:link-type: doc

Run your first navigation simulation step by step.
:::

:::{grid-item-card} 🏗️ Architecture
:link: user_guide/architecture
:link-type: doc

Understand the layered architecture design of OmniNav.
:::

::::

```{toctree}
:maxdepth: 2
:hidden:
:caption: User Guide

getting_started/installation
getting_started/first_simulation
user_guide/architecture
user_guide/robots
user_guide/sensors
user_guide/scenes
user_guide/algorithms
user_guide/evaluation
user_guide/ros2_integration
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: API Reference

api_reference/core
api_reference/robots
api_reference/algorithms
api_reference/evaluation
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: Other

contributing
changelog
```
