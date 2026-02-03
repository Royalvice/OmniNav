#!/usr/bin/env python3
"""
Demo 03: Lidar Visualization (Refactored)

This demo showcases:
1. Mounting a Lidar2DSensor to Go2
2. Visualizing laserscan points using Genesis debug tools
"""

import math
import time
import numpy as np
from omegaconf import OmegaConf

from omninav.core import GenesisSimulationManager
from omninav.robots import Go2Robot
from omninav.sensors import Lidar2DSensor

def main():
    print("=" * 60)
    print("  OmniNav Demo 03: Lidar Visualization")
    print("=" * 60)

    # 1. Config
    cfg = OmegaConf.create({
        "simulation": {
            "dt": 0.01,
            "show_viewer": True,
        },
        "scene": {
            "ground_plane": {"enabled": True},
            # Add some obstacles to scan
            "obstacles": [
                {"type": "box", "size": [0.5, 0.5, 1.0], "position": [1.5, 0.0, 0.5]},
                {"type": "cylinder", "radius": 0.3, "height": 1.0, "position": [0.0, 1.5, 0.5]},
                {"type": "sphere", "radius": 0.4, "position": [-1.0, -1.0, 0.4]},
            ]
        },
        "robot": OmegaConf.load("configs/robot/go2.yaml"),
        "sensor": OmegaConf.load("configs/sensor/lidar_2d.yaml"),
    })
    
    # Enable debug drawing for sensor
    cfg.sensor.draw_debug = True

    # 2. Init
    sim = GenesisSimulationManager()
    sim.initialize(cfg)
    
    # 3. Robot + Sensor
    robot = Go2Robot(cfg.robot, sim.scene)
    sim.add_robot(robot)
    
    # Mount sensor manually for this demo (normally handled via mount_sensors config)
    # We mount it on the 'base' link, raised 0.2m up
    lidar = Lidar2DSensor(cfg.sensor, sim.scene, robot)
    lidar.attach(link_name="base", position=[0.0, 0.0, 0.2])
    lidar.create()
    
    # 4. Build
    sim.load_scene(cfg.scene)
    sim.build()
    
    # 5. Loop
    print("Visualizing Lidar rays...")
    
    t_start = time.time()
    while True:
        sim.step()
        
        # In this demo, Genesis handles visualization automatically 
        # because we set draw_debug=True in config.
        # But we can also access data:
        data = lidar.get_data()
        # ranges = data['ranges']
        # points = data['points']
        
        # Simple circular motion for robot
        t = time.time() - t_start
        robot.apply_command(np.array([0.0, 0.0, 0.5])) # Rotate in place
        
        # Check close window (Genesis viewer check handled by sim.step internally)

if __name__ == "__main__":
    main()
