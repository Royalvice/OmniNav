"""
Observation Inspection Example

Runs the full OmniNav stack with:
- Robot: Go2
- Locomotion: Kinematic Gait
- Sensor: Lidar, Camera, Raycasters (terrain)
- Algorithm: Inspection Pipeline (Global TSP + Local DWA)
- Task: Inspection Task (Waypoints + Coverage)

Demonstrates the new Interface Layer and Evaluation System.
"""

import sys
import logging
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from omninav.interfaces import OmniNavEnv
from omninav.core.types import Action


def main():
    # Configure logging
    logging.basicConfig(level=logging.WARN)
    # Suppress Genesis logs
    logging.getLogger("genesis").setLevel(logging.ERROR)
    
    # Create environment using default configuration (which we just updated)
    # This loads configs/config.yaml -> defaults: inspection, kinematic_gait, etc.
    with OmniNavEnv(config_path="configs") as env:
        
        print("Initializing Inspection Mission...")
        obs_list = env.reset()
        
        print(f"Mission started at time: {env.sim_time:.2f}")
        print(f"Initial robot position: {obs_list[0]['robot_state']['position'][0]}")
        
        # Run simulation loop
        while not env.is_done:
            # Step without actions uses the built-in InspectionAlgorithm pipeline
            obs_list, info = env.step()
            
            step = info["step"]
            sim_time = info.get("sim_time", 0.0)
            
            if step % 100 == 0:
                # Access internals for status update
                # In real usage, you'd look at info or obs
                task_result = env.get_result()
                coverage = task_result.metrics.get("coverage_rate", 0.0) if task_result else 0.0
                print(f"Step {step}: Time={sim_time:.1f}s, Coverage={coverage*100:.1f}%")
        
        # Mission complete
        result = env.get_result()
        print("\nMission Finished!")
        print(f"Success: {result.success}")
        print("Metrics:")
        for k, v in result.metrics.items():
            print(f"  {k}: {v:.4f}")
            
        print(f"Trajectory length: {len(result.info.get('trajectory', []))}")


if __name__ == "__main__":
    main()
