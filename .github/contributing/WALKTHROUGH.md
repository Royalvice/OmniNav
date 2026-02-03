# OmniNav Demo Updates (Go2w)

This walkthrough documents the enhancements made to the OmniNav simulation demos using the Unitree Go2w (wheeled) robot.

## Demo 05: Enhanced Waypoint Navigation
We have upgraded the navigation demo to support interactive control and improved stability.

### Key Features
-   **Minimap Interaction**: A new "Minimap" window provides a top-down view. **Left-click** on the map to set a navigation target.
-   **Live Trajectory**: The robot's path is drawn in red on the minimap.
-   **Strict Control Logic**: Implemented a "Stop -> Turn -> Stop -> Forward" state machine to ensure precise and stable movement.
-   **Safety Speed Limits**: Capped at **0.5 m/s** (linear) and **0.5 rad/s** (angular).

## Demo 03: Lidar Visualization (Refined)
Updates to improve visualization clarity:
-   **Red Lidar Lines**: Added a `gs.sensors.Lidar` (debug mode) to visualize raycasts as red lines, providing clear spatial feedback similar to teleop tools.
-   **High-Res Depth**: Increased the Raycaster depth window resolution to **256x256** for a sharper view.

## Demo 04: Camera Visualization (Fixed)
Corrections to the camera setup:
-   **Orientation**: Rotated camera to `[90, 0, -90]` to correctly face forward relative to the robot base.
-   **Position**: Mounted at `[0.45, 0.0, 0.2]` to clear the robot's chassis and wheels.

## Usage
Run the demos to see the changes:
```bash
# Interactive Navigation
python examples/05_waypoint_navigation.py

# Lidar Visualization
python examples/03_lidar_visualization.py

# Camera Visualization
python examples/04_camera_visualization.py
```
