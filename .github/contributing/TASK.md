# Locomotion Dual Mode Implementation

## Phase A: SimpleGaitController
- [x] Add `gravity_compensation` support in `SimpleGaitController`
- [x] Implement `_apply_kinematic_motion` for direct base control
- [x] Implement sinusoidal leg animation (decoration)
- [x] Create/Update `configs/locomotion/simple_gait.yaml`
- [x] Verify with `examples/06_teleop_simple_gait.py`

## Phase B: IKController
- [x] Restore physics-based PD control
- [x] Implement `_raycast_terrain` for ground detection
- [x] Adjust IK foot targets based on terrain height
- [x] Tune PD gains for stable physical walking
- [x] Create/Update `configs/locomotion/ik_gait.yaml`
- [x] Verify terrain adaptation on stairs/slopes

## Phase C: Demo Enhancements (Lidar & Camera)
- [x] Enhance Demo 03 and Demo 04 following Genesis pattern
    - [x] Fix ground plane rendering in both demos
    - [x] Integrate Go2 locomotion control (WASD/QE) from Demo 01
    - [x] Update Demo 03 to use Raycaster Depth Camera (Depth pattern)
    - [x] Add Obstacle Ring for sensor verification

## Phase D: Documentation & Standardization
- [x] Update `WALKTHROUGH.md` with new modes
- [x] Synchronization with docs

## Phase E: Redesign IK Locomotion (Jitter Fix)
- [x] Analyze and Redesign Controller Strategy
    - [x] Identify root cause (Body-relative target feedback loop)
    - [x] Propose "World-Frame Target Locking" state machine
    - [x] Implement `LocomotionStateMachine` (Walk/Stand)
    - [x] Implement smooth transitions (interpolation)
    - [x] Verify stability in Demo 01 and Demo 07

## Phase F: Migrate Demos to Go2w
- [x] Analyze `02_teleop_go2w.py` for control logic
- [x] Migrate `03_lidar_visualization.py` to Go2w
- [x] Migrate `04_camera_visualization.py` to Go2w
- [x] Migrate `05_waypoint_navigation.py` to Go2w
- [x] Verify all migrated demos

## Phase G: Enhanced Navigation Demo
- [ ] Create `implementation_plan.md` (Done)
- [x] Implement `MinimapVisualizer` class with trajectory drawing
- [x] Implement `NavigationStateMachine` (Stop-Turn-Go logic)
- [x] Update `05_waypoint_navigation.py` to integrate new features
- [x] Verify strict control and click-to-nav functionality

## Phase H: Lidar Visualization Refinement
- [x] Modify `03_lidar_visualization.py`
    - [x] Update `RaycasterDepthSensor` config: bigger size (256x256), no debug.
    - [x] Add `gs.sensors.Lidar` for sparse red line visualization.
- [x] Verify visualization and performance
