# OmniNav

```{toctree}
:maxdepth: 2
:caption: Getting Started

getting_started/installation
getting_started/first_simulation
```

```{toctree}
:maxdepth: 2
:caption: User Guide

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
:caption: API Reference

api_reference/core
api_reference/robots
api_reference/algorithms
api_reference/evaluation
```

```{toctree}
:maxdepth: 1
:caption: Other

contributing
changelog
```

## ✨ What is OmniNav?

OmniNav is an **Embodied AI Simulation Platform** based on the [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) physics engine, designed for rapid verification of robot navigation and obstacle avoidance algorithms.

### Core Features

- 🚀 **High-Performance Simulation** - Based on Genesis engine, supporting GPU acceleration
- 🔌 **Pluggable Algorithms** - Traditional algorithms, VLA/VLN, and other neural network algorithms can be quickly integrated
- 📊 **Built-in Evaluation System** - Predefined navigation tasks and evaluation metrics (SPL, Success Rate, etc.)
- 🤖 **Multi-Robot Support** - Initial version supports Unitree Go2 (Quadruped/Wheeled)
- 🌐 **ROS2 Compatibility** - Optional ROS2 bridge supporting Sim2Real
- 📦 **Scene Asset Import** - Supports USD, GLB, OBJ, and other formats

## 🚀 Quick Start

```python
from omninav import OmniNavEnv

env = OmniNavEnv(config_path="configs")
obs = env.reset()

while not env.is_done:
    action = env.algorithm.step(obs)
    obs, info = env.step(action)

result = env.get_result()
print(f"Success: {result.success}")
```
