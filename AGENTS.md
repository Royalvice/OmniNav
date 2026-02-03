# AGENTS.md - OmniNav AI Agent Guide

Guide for AI coding assistants working with the OmniNav simulation platform.

## 1. Context Synchronization Protocol (CRITICAL)

**Before strictly executing any coding task**, you MUST perform the following **Context Sync Routine** to ensure you are aligned with the latest project state. **Execute this sequence in order**:

1.  **Understand Vision (`.github/contributing/REQUIREMENTS.md`)**:
    *   **Priority 1**: Read this FIRST.
    *   Understand the P0-P2 priorities and the strategic "Why" behind features.
    *   Align any new request with the defined System Architecture.

2.  **Check Status (`.github/contributing/TASK.md`)**:
    *   Determine the current active Phase.
    *   Identify the immediate next tasks marked as `[ ]`.
    *   *Self-Correction*: If the user's request conflicts with `TASK.md`, ask for clarification.

3.  **Verify Blueprint (`.github/contributing/IMPLEMENTATION_PLAN.md`)**:
    *   **Architecture Validity**: Review the Mermaid diagrams and data flow.
    *   **API Compliance**: STRICTLY check API signatures (esp. **Batch-First** requirements).
    *   *Constraint*: Do not invent new patterns; follow the blueprints in this file.

4.  **Confirm Capability (`.github/contributing/WALKTHROUGH.md`)**:
    *   Understand what is already built and working (Demos/Sensors).
    *   Avoid re-implementing existing features.

## 2. Coding Standards & Principles

### 2.1 Architecture Principles
*   **Batch-First Everything**: All APIs must support `(num_envs, ...)` tensor shapes.
    *   Single env? -> `(1, ...)`
    *   Multi env? -> `(N, ...)`
*   **Typed Interfaces**: Use `TypedDict` for all data exchange (`Observation`, `Action`, `RobotState`).
*   **Separation of Concerns**:
    *   **Core**: Engine wrappers only.
    *   **Robot**: Hardware abstraction (Joints/Motors).
    *   **Locomotion**: `cmd_vel` -> Joint Control.
    *   **Algorithm**: High-level planning (`obs` -> `cmd_vel`).

### 2.2 Genesis Integration Rules
*   **Source of Truth**: The local `external/Genesis` codebase is the **ONLY** authority.
    *   ❌ DO NOT rely on training data about Genesis (it changes fast).
    *   ✅ DO use `view_file` on `external/Genesis/examples/**` to find correct API patterns.
    *   ✅ DO use `view_file` on `external/Genesis/doc/**` for parameter details.
*   **Performance**: Use Genesis's bulk APIs (e.g., `control_dofs_velocity` with indices) instead of loops.

### 2.3 Documentation Hygiene
*   **Update-As-You-Go**: If you implement a feature, you MUST check off the task in `.github/contributing/TASK.md` and update `.github/contributing/WALKTHROUGH.md`.
*   **English Comments**: All code comments and docstrings MUST be in English.
*   **Bilingual Docs**: Documentation content should be in **Chinese (中文)** for clarity, as requested by the user.

## 3. Directory Structure Map

```text
OmniNav/
├── configs/                        # Hydra Configs
│   ├── algorithm/                  # Nav/Planning algos
│   ├── locomotion/                 # Locomotion (wheel, ik, rl)
│   ├── robot/                      # Robot defs (go2, go2w)
│   ├── sensor/                     # Sensor configs
│   ├── scene/                      # Scene configurations
│   ├── task/                       # Evaluation tasks
│   └── config.yaml                 # Entry config
├── .github/                        # GitHub config & Docs
│   └── contributing/               # [Source of Truth] Dev Docs
│       ├── REQUIREMENTS.md         
│       ├── IMPLEMENTATION_PLAN.md  
│       ├── TASK.md                 
│       └── WALKTHROUGH.md          
├── docs/                           # User Docs (Sphinx)
├── external/                       # [Reference] Git Submodules
│   ├── Genesis/                    # >> READ ME for Physics APIs
│   └── genesis_ros/                # >> READ ME for ROS2 Bridge patterns
├── omninav/                        # [Source Code] Core Package
│   ├── algorithms/                 # [Layer] Algos (A*, RL, VLA)
│   ├── assets/                     # [Layer] Asset Mgmt & Scene Gen
│   ├── core/                       # [Layer] Core Engine & Registry
│   ├── evaluation/                 # [Layer] Evaluation System
│   ├── interfaces/                 # [Layer] External Interfaces (Gym, ROS2)
│   ├── locomotion/                 # [Layer] Locomotion Control
│   ├── robots/                     # [Layer] Robot Definitions
│   └── sensors/                    # [Layer] Sensore Impl
├── tests/                          # Tests
└── examples/                       # [Validation] Interactive Examples
```

## 4. Reference Documentation

| Document | Description |
|----------|-------------|
| [REQUIREMENTS.md](.github/contributing/REQUIREMENTS.md) | Project vision, goals, and P0-P2 priorities |
| [IMPLEMENTATION_PLAN.md](.github/contributing/IMPLEMENTATION_PLAN.md) | Technical architecture, API definitions, and data flow |
| [TASK.md](.github/contributing/TASK.md) | Current task checklist and progress tracker |
| [WALKTHROUGH.md](.github/contributing/WALKTHROUGH.md) | Current project status, completed features, and demos |

## 5. Current Development Focus (Dynamic Slot)

> *This section is dynamically interpreted based on `TASK.md`.*

**Sample Focus (Phase 3)**:
*   Refactoring core types to `TypedDict`.
*   Updating `OmniNavEnv` to be Batch-First compatible.
*   Implementing standardized Algorithm interfaces.

**Instruction to Agent**: When the user asks "Status?" or "Next?", synthesize your answer from the documents listed in Section 1. Do not hallucinate progress.
