# OmniNav

```{image} _static/logo.png
:align: center
:width: 250px
:class: sd-mb-4
```

:::{sd-container}
:className: sd-mt-2

:::{sd-row}
:::{sd-col}
:size: 12
:className: sd-text-center

**Universal Simulation Platform for Embodied AI**

[GitHub](https://github.com/Royalvice/OmniNav) | [PyPI](https://pypi.org/project/omninav/) | [License: Apache 2.0](https://github.com/Royalvice/OmniNav/blob/main/LICENSE)

:::
:::

:::{sd-row}
:gutter: 3
:className: sd-mt-4

:::{sd-col}
:size: 12
:size-md: 6

:::{sd-card}
:header: ✨ What is OmniNav?
:shadow: md

OmniNav is an **Embodied AI Simulation Platform** based on the [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) physics engine. It is designed for researchers and developers to rapidly verify robot navigation and obstacle avoidance algorithms.

- **High-Performance**: GPU-accelerated physics via Genesis.
- **Pythonic**: 100% Python API for seamless integration.
- **Algorithm Agnostic**: From RL to VLA, plug any policy easily.
:::
:::

:::{sd-col}
:size: 12
:size-md: 6

:::{sd-card}
:header: 🚀 Key Features
:shadow: md

- 🎮 **Pluggable Algorithms** - Standard interfaces for VLA, RL, and Traditional controllers.
- 📊 **Evaluation Engine** - Predefined tasks (SPL, Success Rate) for benchmark testing.
- 🤖 **Multi-Robot Support** - Out-of-the-box support for Unitree Go2 (Legged/Wheeled).
- 🔌 **ROS2 Integration** - Seamless bridge for Sim2Real workflows.
:::
:::
:::

:::{sd-row}
:gutter: 3
:className: sd-mt-2

:::{sd-col}
:size: 12

:::{sd-card}
:header: 🏁 Quick Start
:shadow: md

```python
from omninav import OmniNavEnv

env = OmniNavEnv(config_path="configs")
obs = env.reset()

while not env.is_done:
    action = env.algorithm.step(obs)
    obs, info = env.step(action)

print(f"Goal Reached: {env.get_result().success}")
```
:::
:::
:::

:::

```{toctree}
:maxdepth: 2
:hidden:
:caption: Getting Started

getting_started/installation
getting_started/first_simulation
```

```{toctree}
:maxdepth: 2
:hidden:
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
