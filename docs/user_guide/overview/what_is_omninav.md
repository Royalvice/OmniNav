# What is OmniNav?

OmniNav is a general-purpose navigation simulation platform built on top of the [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) physics engine, designed for *Embodied AI / Robotics Navigation / Sim2Real* applications.

It is simultaneously multiple things:

1. A **unified navigation benchmark** for evaluating navigation and obstacle avoidance algorithms.
2. A **plug-and-play algorithm framework** supporting both classical planners and neural network-based methods (VLA/VLN).
3. A **robot-agnostic platform** with built-in support for quadruped, wheeled, and humanoid robots.
4. A **high-fidelity simulation environment** with GPU-accelerated physics and photo-realistic rendering.

## Key Features

- 🚀 **High Performance**: Leverages Genesis engine for GPU-accelerated physics simulation.
- 🔌 **Plug-and-Play Algorithms**: Easy integration of classical planners, RL policies, and VLA/VLN models.
- 📊 **Built-in Evaluation**: Pre-defined navigation tasks with standard metrics (SPL, Success Rate).
- 🤖 **Multi-Robot Support**: Quadruped (Go2), wheeled robots, and extensible to other platforms.
- 🌐 **ROS2 Compatible**: Optional ROS2 bridge for Sim2Real deployment.
- 📦 **Scene Import**: Support for USD, GLB, OBJ, and custom scene assets.
- 🔧 **Configuration-Driven**: Hydra-based configuration for flexible experiment management.

## Long-term Missions

1. **Simplifying navigation research** by providing a ready-to-use benchmark with standardized evaluation metrics.
2. **Bridging the Sim2Real gap** with high-fidelity physics simulation and optional ROS2 integration.
3. **Accelerating algorithm development** with a modular, extensible architecture.
