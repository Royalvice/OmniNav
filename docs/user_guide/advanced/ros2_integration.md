# ROS2 Integration

OmniNav provides an optional ROS2 bridge. You can run pure Python simulation, or switch to Nav2-driven control.

## Integration Modes

- `control_source=python`: default mode. OmniNav internal algorithm controls the robot.
- `control_source=ros2`: Nav2 (or any ROS2 node) publishes `/cmd_vel` and OmniNav executes it.

## Core Responsibility Split

OmniNav publishes:
- `/clock`
- `/odom`
- `/tf` (`odom -> base_link`)
- `/tf_static` (`base_link -> laser`)
- `/scan`

OmniNav subscribes:
- `/cmd_vel` (only when `control_source=ros2`)

Nav2 stack publishes/provides:
- `/map` from `map_server`
- `map -> odom` from localization (`amcl`)

## Why `/map` Is Needed

OmniNav has a 3D simulation scene, but Nav2 global planning/localization expects a 2D occupancy grid (`nav_msgs/OccupancyGrid`).

Without `/map`:
- global planner and global costmap usually fail to start,
- AMCL cannot localize,
- you only get local reactive behavior.

## Recommended Config

```yaml
ros2:
  enabled: true
  control_source: ros2
  profile: nav2_full
  topics:
    cmd_vel_in: /cmd_vel
    odom: /odom
    scan: /scan
  frames:
    map: map
    odom: odom
    base_link: base_link
    laser: laser_frame
```

`profile` behavior:
- `all_off`: publish nothing by default.
- `nav2_minimal` / `nav2_full`: enables `clock/tf/tf_static/odom/scan`.

## End-to-End Nav2 Chain

1. OmniNav bridge publishes `/clock`, `/odom`, `/tf`, `/tf_static`, `/scan`.
2. `map_server` publishes `/map`.
3. `amcl` consumes `/map + /scan + /odom + tf` and publishes `map -> odom`.
4. Nav2 controller publishes `/cmd_vel`.
5. OmniNav consumes `/cmd_vel` and executes motion.

## Bringup Order (Humble)

1. Start OmniNav bridge:

```bash
python examples/06_ros2_nav2_bridge.py
```

2. Start map server and AMCL (with `use_sim_time:=true`).
3. Start Nav2 bringup and RViz (also `use_sim_time:=true`).
4. Publish initial pose (`/initialpose`) and send goal (`/goal_pose` or action API).

Minimum checks:
- `ros2 topic echo /clock` has data
- `ros2 topic echo /map` has data
- `ros2 topic echo /scan` has data
- `ros2 run tf2_tools view_frames` includes `map -> odom -> base_link -> laser_frame`

## Safety Behavior

When `control_source=ros2`, OmniNav applies a command timeout (`cmd_vel_timeout_sec`).
If `/cmd_vel` becomes stale, command decays to zero velocity.

## Multi-Environment Note

Current stable Nav2 workflow targets `n_envs=1`.
For `n_envs>1`, use Python mode now, then extend to per-environment namespace (`/env_0/*`, `/env_1/*`) in the next phase.
