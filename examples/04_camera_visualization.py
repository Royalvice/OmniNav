#!/usr/bin/env python3
"""
Demo 04: Camera Visualization (Refactored)

This demo showcases:
1. Mounting a CameraSensor (RGB-D)
2. Displaying camera stream using OpenCV
"""

import sys
import numpy as np
import cv2
from omegaconf import OmegaConf

from omninav.core import GenesisSimulationManager
from omninav.robots import Go2Robot
from omninav.sensors import CameraSensor

def main():
    print("=" * 60)
    print("  OmniNav Demo 04: Camera Visualization")
    print("=" * 60)

    # 1. Config
    cfg = OmegaConf.create({
        "simulation": {
            "dt": 0.01,
            # For camera rendering, we need graphics backend
            "backend": "gpu", 
            "show_viewer": True,
        },
        "scene": {
            "ground_plane": {"enabled": True},
            "obstacles": [
                {"type": "box", "size": [0.5, 0.5, 1.0], "position": [2.0, 0.0, 0.5]},
                {"type": "box", "size": [0.2, 2.0, 1.0], "position": [0.0, 2.0, 0.5]},
            ]
        },
        "robot": OmegaConf.load("configs/robot/go2.yaml"),
        "sensor": OmegaConf.load("configs/sensor/camera_rgbd.yaml"),
    })

    # 2. Init
    sim = GenesisSimulationManager()
    sim.initialize(cfg)
    
    # 3. Robot + Sensor
    robot = Go2Robot(cfg.robot, sim.scene)
    sim.add_robot(robot)
    
    # Mount camera: Forward looking, slightly elevated
    camera = CameraSensor(cfg.sensor, sim.scene, robot)
    # orientation: [roll, pitch, yaw]. Camera looks along X by default.
    camera.attach(link_name="base", position=[0.2, 0.0, 0.1], orientation=[0, 0, 0])
    camera.create()
    
    # 4. Build
    sim.load_scene(cfg.scene)
    sim.build()
    
    print("Simulation started. Press 'q' in OpenCV window to exit.")

    # 5. Loop
    try:
        while True:
            # Rotate robot to see surroundings
            robot.apply_command([0.0, 0.0, 0.3])
            
            sim.step()
            
            # Get camera data
            data = camera.get_data()
            rgb = data.get("rgb")     # [H, W, 3]
            depth = data.get("depth") # [H, W]
            
            if rgb is not None:
                # Convert RGB to BGR for OpenCV
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                cv2.imshow("Robot RGB Eye", bgr)
            
            if depth is not None:
                # Normalize depth for visualization (0-5m range)
                depth_vis = np.clip(depth, 0, 5.0) / 5.0
                cv2.imshow("Robot Depth Eye", depth_vis)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
