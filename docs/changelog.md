# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project structure
- Core layer: SimulationManager base class
- Robot layer: RobotBase, SensorBase base classes
- Algorithm layer: AlgorithmBase base class and Waypoint Following algorithm
- Evaluation layer: TaskBase, MetricBase base classes and Point Navigation task
- Interface layer: OmniNavEnv (Gym-like API)
- Config management: Hydra/OmegaConf integration
- Documentation: MkDocs + Material theme

### Planned
- Kinematic controller implementation
- Go2/Go2w robot support
- ROS2 bridge module
- More navigation algorithm examples
