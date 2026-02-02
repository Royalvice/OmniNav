<p align="center">
  <img src="docs/_static/logo.png" alt="OmniNav" width="400">
</p>

<h1 align="center">OmniNav</h1>

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

## Table of Contents

1. [What is OmniNav?](#what-is-omninav)
2. [Key Features](#key-features)
3. [Quick Installation](#quick-installation)
4. [Quick Start](#quick-start)
5. [Architecture](#architecture)
6. [Documentation](#documentation)
7. [Contributing](#contributing)
8. [License and Acknowledgments](#license-and-acknowledgments)
9. [Citation](#citation)

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

## Quick Installation

### Prerequisites

- Python >= 3.10
- CUDA-compatible GPU (recommended)
- PyTorch (see [official instructions](https://pytorch.org/get-started/locally/))

### Install from Source

```bash
# Clone repository with submodules
git clone --recurse-submodules https://github.com/Royalvice/OmniNav.git
cd OmniNav

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/macOS

# Install Genesis
cd external/Genesis && pip install -e . && cd ../..

# Install OmniNav
pip install -e .
```

## Quick Start

### Basic Example

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

### Using Custom Algorithm

```python
from omninav import OmniNavEnv
import numpy as np

env = OmniNavEnv(config_path="configs")
obs = env.reset()

while not env.is_done:
    # Your custom navigation logic
    robot_pos = obs["robot_state"].position
    goal_pos = obs.get("goal_position", [5.0, 0.0, 0.0])
    
    # Simple proportional controller
    direction = np.array(goal_pos[:2]) - robot_pos[:2]
    cmd_vel = np.array([direction[0] * 0.5, direction[1] * 0.5, 0.0])
    
    obs, info = env.step(cmd_vel)

env.close()
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

## Documentation

Comprehensive documentation is available at: [https://royalvice.github.io/OmniNav/](https://royalvice.github.io/OmniNav/)

- **Getting Started**: Installation and basic usage
- **Tutorials**: Step-by-step guides for common tasks
- **API Reference**: Complete API documentation
- **Configuration Guide**: Hydra configuration system

## Contributing

The OmniNav project welcomes contributions from the community:

- **Pull requests** for new features or bug fixes
- **Bug reports** through GitHub Issues
- **Suggestions** to improve usability and documentation

## License and Acknowledgments

OmniNav is licensed under the [Apache-2.0 License](LICENSE).

OmniNav is built on top of these excellent open-source projects:

- [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) - High-performance physics engine for Embodied AI
- [Hydra](https://github.com/facebookresearch/hydra) - Framework for elegantly configuring complex applications
- [OmegaConf](https://github.com/omry/omegaconf) - Hierarchical configuration system

## Citation

If you use OmniNav in your research, please consider citing:

```bibtex
@misc{OmniNav,
  author = {OmniNav Contributors},
  title = {OmniNav: A General-Purpose Navigation Simulation Platform for Embodied AI},
  year = {2025},
  url = {https://github.com/Royalvice/OmniNav}
}
```
