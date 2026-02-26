# Core Runtime API

## Scope

This page documents core runtime modules:
- [omninav/core/runtime.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/core/runtime.py)
- [omninav/core/simulation_manager.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/core/simulation_manager.py)
- [omninav/core/registry.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/core/registry.py)
- [omninav/core/lifecycle.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/core/lifecycle.py)
- [omninav/core/types.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/core/types.py)
- [omninav/core/map/types.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/core/map/types.py)
- [omninav/core/map/service.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/core/map/service.py)
- [omninav/core/map/facade.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/core/map/facade.py)

## Key classes

### `SimulationRuntime`

Source: [runtime.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/core/runtime.py)

Responsibilities:
- Build and hold simulation components
- Reset/step orchestration
- Task feedback and result collection
- Optional map service injection and floor metadata propagation

Main methods:
- `build()`
- `reset()`
- `step(actions=None)`
- `get_result()`

### `GenesisSimulationManager`

Source: [simulation_manager.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/core/simulation_manager.py)

Responsibilities:
- Genesis scene creation
- Scene entities and sensors loading
- Simulation stepping and time

### Registry and lifecycle

- Registry: [`Registry`, `BuildContext`](https://github.com/Royalvice/OmniNav/blob/main/omninav/core/registry.py)
- Lifecycle: [`LifecycleState`, `LifecycleMixin`](https://github.com/Royalvice/OmniNav/blob/main/omninav/core/lifecycle.py)

## Data contracts

Source: [types.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/core/types.py)

Primary contracts:
- `Observation` (batch-first)
- `Action` (`cmd_vel` shape `(B,3)`)
- `TaskResult`
- `RobotState`, `SensorData`

Validation helper:
- `validate_batch_shape(...)`

## Map service

Sources:
- [types.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/core/map/types.py)
- [service.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/core/map/service.py)
- [facade.py](https://github.com/Royalvice/OmniNav/blob/main/omninav/core/map/facade.py)

Capabilities:
- Occupancy maps by floor
- World/grid coordinate conversion
- Connector graph access
- Runtime floor query by XY
