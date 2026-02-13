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
import argparse
import logging
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from omninav.interfaces import OmniNavEnv
from omninav.core.types import Action


def main():
    parser = argparse.ArgumentParser(description="Inspection pipeline demo")
    parser.add_argument("--test-mode", action="store_true", help="Run for a bounded number of steps and exit")
    parser.add_argument("--max-steps", type=int, default=400, help="Max steps in test mode")
    parser.add_argument("--show-viewer", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=logging.WARN)
    # Suppress Genesis logs
    logging.getLogger("genesis").setLevel(logging.ERROR)
    
    # Create environment using default configuration (which we just updated)
    # This loads configs/config.yaml -> defaults: inspection, kinematic_gait, etc.
    overrides = [f"simulation.show_viewer={str(args.show_viewer)}"]
    if args.test_mode:
        overrides.append("task.time_budget=5.0")
    with OmniNavEnv.from_config(config_path="configs", overrides=overrides) as env:
        
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
            if args.test_mode and step >= args.max_steps:
                print("Test mode reached max steps, stopping early.")
                break
        
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
