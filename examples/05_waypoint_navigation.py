#!/usr/bin/env python3
"""
Demo 05: Enhanced Waypoint Navigation (Go2w)

Features:
1.  **Minimap**: Open a separate top-down 2D map window.
2.  **Trajectory**: Draw robot path (Red line) on minimap.
3.  **Click-to-Nav**: Click on minimap to set target.
4.  **Strict Control**: "Stop -> Turn -> Stop -> Forward" logic.
5.  **Speed Limits**: Aligned with teleop demo (0.5 m/s, 0.5 rad/s).
"""

import sys
import math
import numpy as np
import cv2
from omegaconf import OmegaConf
from collections import deque

from omninav.core import GenesisSimulationManager
from omninav.robots import Go2wRobot
from omninav.locomotion import WheelController

# =============================================================================
# Constants
# =============================================================================
# Speed Limits (from Demo 02)
MAX_LINEAR_VEL = 0.5   # m/s
MAX_ANGULAR_VEL = 0.5  # rad/s

# Navigation Parameters
GOAL_TOLERANCE = 0.15        # meters
ANGLE_TOLERANCE = 0.05       # radians (~3 degrees)
BRAKE_DURATION = 10          # steps (0.1s at dt=0.01)

# Minimap Config
MAP_SIZE = 500               # pixels
MAP_SCALE = 50               # pixels per meter
MAP_CENTER_X = MAP_SIZE // 2
MAP_CENTER_Y = MAP_SIZE // 2

# =============================================================================
# Utilities
# =============================================================================
def normalize_angle(angle):
    """Normalize angle to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi

# =============================================================================
# Minimap Visualizer
# =============================================================================
class MinimapVisualizer:
    def __init__(self, name="Minimap: Click to Navigate"):
        self.name = name
        self.image = np.zeros((MAP_SIZE, MAP_SIZE, 3), dtype=np.uint8)
        self.trajectory = deque(maxlen=1000)
        self.target = None # (x, y) world coords
        
        # Interaction
        cv2.namedWindow(self.name)
        cv2.setMouseCallback(self.name, self._mouse_callback)

    def _world_to_map(self, x, y):
        # World X -> Map U (Right)
        # World Y -> Map V (Up? Usually Map Y is Down)
        # Let's align: World X+ is Right, World Y+ is Up.
        # Map U = Center + X * Scale
        # Map V = Center - Y * Scale (Flip Y for image coords)
        u = int(MAP_CENTER_X + x * MAP_SCALE)
        v = int(MAP_CENTER_Y - y * MAP_SCALE)
        return u, v

    def _map_to_world(self, u, v):
        x = (u - MAP_CENTER_X) / MAP_SCALE
        y = (MAP_CENTER_Y - v) / MAP_SCALE
        return x, y

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            wx, wy = self._map_to_world(x, y)
            self.set_target(wx, wy)
            print(f"[Minimap] Target set to: ({wx:.2f}, {wy:.2f})")

    def set_target(self, x, y):
        self.target = np.array([x, y])

    def update(self, robot_pos, robot_yaw):
        # Clear background
        self.image.fill(240) # Light gray
        
        # 1. Draw Grid (every 1 meter)
        for i in range(-5, 6):
            # Vertical lines
            u, _ = self._world_to_map(i, 0)
            cv2.line(self.image, (u, 0), (u, MAP_SIZE), (200, 200, 200), 1)
            # Horizontal lines
            _, v = self._world_to_map(0, i)
            cv2.line(self.image, (0, v), (MAP_SIZE, v), (200, 200, 200), 1)
            
        # 2. Draw Trajectory
        self.trajectory.append(robot_pos[:2])
        if len(self.trajectory) > 1:
            pts = [self._world_to_map(p[0], p[1]) for p in self.trajectory]
            cv2.polylines(self.image, [np.array(pts)], False, (0, 0, 255), 2) # Red

        # 3. Draw Target
        if self.target is not None:
            tx, ty = self._world_to_map(self.target[0], self.target[1])
            cv2.drawMarker(self.image, (tx, ty), (0, 100, 0), cv2.MARKER_CROSS, 20, 2)
            cv2.circle(self.image, (tx, ty), 5, (0, 100, 0), -1)

        # 4. Draw Robot
        rx, ry = self._world_to_map(robot_pos[0], robot_pos[1])
        # Body
        cv2.circle(self.image, (rx, ry), 8, (0, 0, 0), -1)
        # Heading indicator
        hx = int(rx + 15 * math.cos(robot_yaw))
        hy = int(ry - 15 * math.sin(robot_yaw)) # Y-flip
        cv2.line(self.image, (rx, ry), (hx, hy), (0, 0, 0), 2)

        # Show
        cv2.imshow(self.name, self.image)
        cv2.waitKey(1)

    def close(self):
        cv2.destroyWindow(self.name)

# =============================================================================
# Navigation Controller
# =============================================================================
class NavigationStateMachine:
    # States
    IDLE = "IDLE"
    ALIGNING = "ALIGNING"
    MOVING = "MOVING"
    BRAKING = "BRAKING"

    def __init__(self, visualizer):
        self.state = self.IDLE
        self.vis = visualizer
        self.current_target = None
        self.brake_counter = 0
        self.next_state_after_brake = None
        
    def step(self, robot_pos, robot_yaw):
        # Check for new target from UI
        if self.vis.target is not None:
            # Check if target changed significantly
            if self.current_target is None or np.linalg.norm(self.vis.target - self.current_target) > 0.01:
                print("[Nav] New target received. BRAKING.")
                self.current_target = self.vis.target.copy()
                self._enter_braking(self.ALIGNING)

        if self.current_target is None:
            return np.zeros(3)

        # Compute errors
        error_pos = self.current_target - robot_pos[:2]
        dist = np.linalg.norm(error_pos)
        target_angle = math.atan2(error_pos[1], error_pos[0])
        angle_error = normalize_angle(target_angle - robot_yaw)

        # State Machine
        cmd = np.zeros(3)

        if self.state == self.BRAKING:
            self.brake_counter -= 1
            if self.brake_counter <= 0:
                print(f"[Nav] Braking done. Transition to {self.next_state_after_brake}")
                self.state = self.next_state_after_brake
            return np.zeros(3) # Stop

        elif self.state == self.IDLE:
             # If we have a target but in IDLE, start moving (via brake->align)
             if self.current_target is not None and dist > GOAL_TOLERANCE:
                 self._enter_braking(self.ALIGNING)

        elif self.state == self.ALIGNING:
            if abs(angle_error) < ANGLE_TOLERANCE:
                print("[Nav] Aligned. BRAKING before Moving.")
                self._enter_braking(self.MOVING)
            else:
                # Turn in place
                cmd[2] = np.clip(angle_error * 2.0, -MAX_ANGULAR_VEL, MAX_ANGULAR_VEL)

        elif self.state == self.MOVING:
            if dist < GOAL_TOLERANCE:
                print("[Nav] Target Reached. BRAKING -> IDLE.")
                self.current_target = None
                self.vis.target = None # Clear target visual
                self._enter_braking(self.IDLE)
            elif abs(angle_error) > 0.5: # If we drifted too much / overshot
                print("[Nav] Large angle error. BRAKING -> ALIGNING.")
                self._enter_braking(self.ALIGNING)
            else:
                # Move forward with heading correction
                cmd[0] = np.clip(dist * 1.0, -MAX_LINEAR_VEL, MAX_LINEAR_VEL)
                cmd[2] = np.clip(angle_error * 1.5, -MAX_ANGULAR_VEL, MAX_ANGULAR_VEL)

        return cmd

    def _enter_braking(self, next_state):
        self.state = self.BRAKING
        self.brake_counter = BRAKE_DURATION
        self.next_state_after_brake = next_state

# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 60)
    print("  OmniNav Demo 05: Enhanced Navigation")
    print("  Controls: LEFT CLICK on Minimap to navigate.")
    print("=" * 60)

    # 1. Config
    cfg = OmegaConf.create({
        "simulation": {
            "dt": 0.01,
            "show_viewer": True,
            "camera_pos": [0.0, 0.0, 5.0],
            "camera_lookat": [0.0, 0.0, 0.0],
            "disable_keyboard_shortcuts": True,
        },
        "scene": {
            "ground_plane": {"enabled": True},
            "obstacles": []
        },
        "robot": OmegaConf.load("configs/robot/go2w.yaml"),
        "control": OmegaConf.load("configs/locomotion/wheel.yaml"),
    })

    # 2. Init
    sim = GenesisSimulationManager()
    sim.initialize(cfg)
    
    # 3. Setup Robot
    robot = Go2wRobot(cfg.robot, sim.scene)
    sim.add_robot(robot)
    
    # 4. Build
    sim.load_scene(cfg.scene)
    sim.build()
    
    robot.reset()
    
    # 5. Controller
    controller = WheelController(cfg.control, robot)
    controller.reset()
    
    # 6. Navigation System
    minimap = MinimapVisualizer()
    nav_sm = NavigationStateMachine(minimap)
    
    # Set initial waypoint
    initial_wp = [2.0, 0.0]
    minimap.set_target(initial_wp[0], initial_wp[1])
    print(f"[Main] Initial target set to: {initial_wp}")

    # 7. Loop
    try:
        while True:
            # A. Get State
            state = robot.get_state()
            pos = state.position  # [x, y, z]
            
            # Quat to Yaw
            q = state.orientation
            siny_cosp = 2 * (q[0] * q[3] + q[1] * q[2])
            cosy_cosp = 1 - 2 * (q[2] * q[2] + q[3] * q[3])
            yaw = math.atan2(siny_cosp, cosy_cosp)
            
            # B. Update UI
            minimap.update(pos, yaw)
            
            # C. Compute Control
            cmd_vel = nav_sm.step(pos, yaw)
            
            # D. Act
            controller.step(cmd_vel)
            sim.step()
            
    except KeyboardInterrupt:
        pass
    finally:
        minimap.close()

if __name__ == "__main__":
    main()
