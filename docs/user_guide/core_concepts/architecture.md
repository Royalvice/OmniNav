# Architecture Overview

OmniNav adopts a layered, registry-driven architecture to ensure effective decoupling and scalability. The core of the simulation is orchestrated by the `SimulationRuntime`.

## System Architecture

The following diagram illustrates the relationship between the major layers:

```mermaid
graph TB
    subgraph "Interface Layer"
        I1[OmniNavEnv]
        I2[ROS2Bridge]
        I3[GymWrapper]
    end

    subgraph "Orchestration Layer"
        RT[SimulationRuntime]
    end

    subgraph "Application Layer"
        E[Evaluation Layer<br/>Tasks & Metrics]
        A[Algorithm Layer<br/>Navigation & Perception]
    end

    subgraph "Control Layer"
        L[Locomotion Layer<br/>Locomotion Controller]
    end

    subgraph "Entity Layer"
        Reg[Registry System]
        R[Robot Layer<br/>Robot & Sensors]
        S[Asset Layer<br/>Scene Generators]
    end

    subgraph "Foundation Layer"
        C[Core Layer<br/>SimulationManager]
        G[Genesis Engine]
    end

    I1 --> RT
    I2 --> RT
    I3 --> I1
    RT --> Reg
    Reg --> R
    RT --> E
    RT --> A
    A --> L
    L --> R
    R --> C
    S --> C
    C --> G
```

## Runtime Lifecycle

Component initialization and simulation steps follow a strict state machine managed by `LifecycleMixin`:

```mermaid
stateDiagram-v2
    [*] --> CREATED: Initialize objects
    CREATED --> SPAWNED: robot.spawn() / sensor.create()
    SPAWNED --> BUILT: simulation.build()
    BUILT --> READY: post_build() logic
    READY --> STEPPING: runtime.step() loop
    STEPPING --> STEPPING: Simulation step
    STEPPING --> DONE: task.is_terminated()
    DONE --> [*]
```

## Core Design Philosophy

OmniNav is built on three pillars that ensure flexibility and performance:

```mermaid
graph LR
    P1[<b>Registry-Based</b><br/>Config-driven instantiation]
    P2[<b>Lifecycle-Managed</b><br/>Deterministic initialization]
    P3[<b>Batch-First</b><br/>GPU-parallel native]
    
    P1 --- P2
    P2 --- P3
    P3 --- P1
```

| Principle             | Description                                                                                                                                                                                          |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Registry-Based**    | All components (robots, sensors, controllers) are registered in a central registry. This allows the system to instantiate components dynamically from YAML configurations without hardcoded imports. |
| **Lifecycle-Managed** | Explicit states (CREATED, SPAWNED, BUILT, READY) prevent timing issues between sensor mounting, physics building, and robot post-processing.                                                         |
| **Batch-First**       | All data interfaces support `(num_envs, ...)` tensors. This allows seamless switching between single-environment debugging and massive multi-environment RL training.                                |

## Layer Responsibilities

| Layer                | Responsibility                                         | Key Class/Interface                      |
| -------------------- | ------------------------------------------------------ | ---------------------------------------- |
| **Core Layer**       | Genesis wrapper, SimulationRuntime orchestrator, Hooks | `SimulationManager`, `SimulationRuntime` |
| **Asset Layer**      | Procedural scene generation and asset loading          | `SceneGeneratorBase`, `AssetLoader`      |
| **Robot Layer**      | Robot kinematic/dynamic state, sensor mounting         | `RobotBase`, `SensorBase`                |
| **Locomotion Layer** | High-level cmd_vel to low-level joint target mapping   | `LocomotionControllerBase`               |
| **Algorithm Layer**  | Planning, collision avoidance, and perception logic    | `AlgorithmBase`                          |
| **Evaluation Layer** | Goal-directed tasks and metric calculation             | `TaskBase`, `MetricBase`                 |
| **Interface Layer**  | External APIs (Gym-like, ROS2, Gymnasium)              | `OmniNavEnv`, `ROS2Bridge`               |

## Data Flow (Runtime Step)

```mermaid
sequenceDiagram
    participant Env as Interface
    participant Runtime
    participant Algo
    participant Loco
    participant Robot
    participant Sim as Genesis
    
    Env->>Runtime: step(action)
    Runtime->>Robot: get_observations()
    
    opt Use Internal Algorithm
        Runtime->>Algo: plan(obs)
        Algo-->>Runtime: cmd_vel
    end
    
    Runtime->>Loco: control(cmd_vel, obs)
    Loco->>Robot: set_joint_targets()
    Runtime->>Sim: physics_step()
    Runtime->>Task: update_metrics(obs)
    Runtime-->>Env: List[obs], info
```

## Next Steps

- [Robot Configuration](robots.md) - How to configure and extend robots
- [Algorithm Integration](algorithms.md) - How to add custom algorithms
- [Evaluation Tasks](evaluation.md) - How to define evaluation tasks and metrics
