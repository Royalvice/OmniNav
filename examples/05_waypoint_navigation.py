#!/usr/bin/env python3
"""
Demo 05: Waypoint Navigation (Refactored)

This demo showcases:
1. Closed-loop control: Get State -> Plan -> Act
2. Simple proportional controller to navigate to a target point
"""

import sys
import math
import numpy as np
from omegaconf import OmegaConf

from omninav.core import GenesisSimulationManager
from omninav.robots import Go2wRobot
from omninav.locomotion import WheelController

def normalize_angle(angle):
    """Normalize angle to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi

def main():
    print("=" * 60)
    print("  OmniNav Demo 05: Waypoint Navigation")
    print("=" * 60)

    # 1. Config (Using Go2w for simpler holonomic control demonstration)
    cfg = OmegaConf.create({
        "simulation": {
            "dt": 0.01,
            "show_viewer": True,
            "camera_pos": [0.0, 0.0, 5.0],  # Top-down view
            "camera_lookat": [0.0, 0.0, 0.0],
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
    
    # 5. Controller
    controller = WheelController(cfg.control, robot)
    controller.reset()

    # Targets (x, y)
    waypoints = [
        [2.0, 0.0],
        [2.0, 2.0],
        [0.0, 2.0],
        [0.0, 0.0]
    ]
    current_wp_idx = 0
    
    print(f"Starting navigation. Targets: {waypoints}")

    # 6. Loop
    try:
        while True:
            # A. Get State
            state = robot.get_state()
            pos = state.position  # [x, y, z]
            
            # Convert quaternion to yaw
            q = state.orientation # [w, x, y, z]
            # yaw = atan2(2(wz + xy), 1 - 2(y^2 + z^2))
            siny_cosp = 2 * (q[0] * q[3] + q[1] * q[2])
            cosy_cosp = 1 - 2 * (q[2] * q[2] + q[3] * q[3])
            yaw = math.atan2(siny_cosp, cosy_cosp)
            
            # B. Check Waypoint Reached
            target = np.array(waypoints[current_wp_idx])
            error = target - pos[:2]
            dist = np.linalg.norm(error)
            
            if dist < 0.2:
                print(f"Reached waypoint {current_wp_idx}: {target}")
                current_wp_idx = (current_wp_idx + 1) % len(waypoints)
                continue
            
            # C. Compute Control (Proportional)
            # Strategy: Head towards target while facing it
            target_angle = math.atan2(error[1], error[0])
            angle_error = normalize_angle(target_angle - yaw)
            
            # Simple P-controller
            v_linear = 1.0 * min(dist, 1.0) # max speed 1.0
            v_angular = 2.0 * angle_error
            
            # For holonomic robot (Go2w), we can also move laterally
            # But here we act like a differential drive for simplicity: 
            # rotate to face, then move forward.
            
            # Omni-directional approach:
            # Global velocity vector required:
            # v_global = error_dir * speed
            # Robot frame velocity = R^T * v_global
            
            c, s = math.cos(yaw), math.sin(yaw)
            R = np.array([[c, -s], [s, c]])
            v_global = error / dist * min(dist, 1.0)
            v_robot = R.T @ v_global
            
            cmd_vel = np.array([v_robot[0], v_robot[1], v_angular])
            
            # D. Act
            controller.step(cmd_vel)
            sim.step()
            
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
