# OmniNav

```{image} _static/logo.png
:align: center
:width: 200px
:class: sd-mb-4 sd-no-bg
```

:::{sd-container}
:className: sd-mt-2

:::{sd-row}
:::{sd-col}
:size: 12
:className: sd-text-center

**Universal Simulation Platform for Embodied AI**

[![GitHub Repo stars](https://img.shields.io/github/stars/Royalvice/OmniNav?style=plastic&logo=GitHub&logoSize=auto)](https://github.com/Royalvice/OmniNav)
[![PyPI version](https://badge.fury.io/py/omninav.svg?icon=si%3Apython)](https://pypi.org/project/omninav/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/Royalvice/OmniNav/blob/main/LICENSE)

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

OmniNav is a physics-based simulation platform designed for the rapid development and evaluation of **Embodied AI** navigation algorithms. Powered by the [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) physics engine, it provides a high-fidelity environment for testing everything from traditional PID controllers to modern **VLA (Vision-Language-Action)** and **VLN (Vision-Language Navigation)** models.

Our goal is to bridge the gap between abstract algorithm design and real-world robotics by providing a fast, scalable, and user-friendly simulation bridge.
:::
:::

:::{sd-col}
:size: 12
:size-md: 6

:::{sd-card}
:header: 🚀 Key Missions
:shadow: md

- **Scale to Thousands**: Leverage GPU-accelerated physics to run massive parallel simulations for RL training.
- **Sim-to-Real Ready**: Built-in ROS2 integration ensures that policies developed in OmniNav can be deployed to physical robots like the Unitree Go2 with minimal friction.
- **Open standard**: Support for standard formats like USD, GLB, and URDF to allow easy integration of assets from various sources.
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
import omninav as on

# Initialize simulation with your config
env = on.OmniNavEnv(config_path="configs/navigation.yaml")
obs = env.reset()

# Single loop for navigation task
while not env.done:
    action = env.policy.predict(obs)
    obs, reward, done, info = env.step(action)

print(f"Goal Reached! SPL Score: {info['spl']:.2f}")
```
:::
:::
:::

:::

```{toctree}
:maxdepth: 2
:hidden:

getting_started/index
user_guide/index
api_reference/index
changelog
contributing
```
