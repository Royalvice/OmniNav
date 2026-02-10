# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-02-10

### Added
- **Registry-Based Architecture**: Unified component discovery and instantiation.
- **Lifecycle Management**: Deterministic state machine for simulation entities.
- **SimulationRuntime**: Decoupled orchestrator for simulation loops.
- **Optimized Kinematic Controller**: Game-style pre-baked animations for quadruped robots (0.1ms/frame).
- **Inspection Task Suite**: Specialized evaluation for coverage-based navigation.
- **Unitree Go2/Go2w Support**: High-fidelity robot integration.
- **Gymnasium Wrapper**: Seamless integration with standard RL libraries.
- **ROS2 Bridge**: Publisher/Subscriber support for sensor data and commands.
- **Integration Test Suite**: Automated verification of the full simulation pipeline.

### Changed
- Refactored `OmniNavEnv` to be a lightweight interface delegating to `SimulationRuntime`.
- Standardized all sensors to return batch-first data layouts.
- Re-organized configuration hierarchy using Hydra composition.

### Foundational
- Initial project structure and Base classes for all layers.
- Hydra/OmegaConf integration.
- Documentation infrastructure.
