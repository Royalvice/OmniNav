"""
OmniNav Simple Example - Go2 Robot Simulation

Demonstrates how to use OmniNav to load a Go2 robot and run simulation.

Usage:
    python examples/simple_go2.py
    python examples/simple_go2.py --cpu  # Use CPU backend
"""

import argparse
from omegaconf import OmegaConf

from omninav.core import GenesisSimulationManager
from omninav.robots import Go2Robot


def main():
    parser = argparse.ArgumentParser(description="OmniNav Go2 Robot Simulation Example")
    parser.add_argument("--cpu", action="store_true", help="Use CPU backend")
    parser.add_argument("--steps", type=int, default=1000, help="Simulation steps")
    args = parser.parse_args()
    
    # Create configuration
    cfg = OmegaConf.create({
        "simulation": {
            "dt": 0.01,
            "substeps": 2,
            "backend": "cpu" if args.cpu else "gpu",
            "n_envs": 1,
            "show_viewer": True,
            "enable_self_collision": False,
            "camera_pos": [2.0, 0.0, 2.5],
            "camera_lookat": [0.0, 0.0, 0.0],
            "camera_fov": 40,
        },
        "robot": {
            "urdf_source": "genesis_builtin",
            "urdf_path": "urdf/go2/urdf/go2.urdf",
            "initial_pos": [0.0, 0.0, 0.4],
            "initial_quat": [1.0, 0.0, 0.0, 0.0],
            "control": {
                "kp": 20.0,
                "kd": 0.5,
            }
        },
        "scene": {
            "ground_plane": {"enabled": True},
            "obstacles": [],
        }
    })
    
    # 1. Create simulation manager
    sim = GenesisSimulationManager()
    sim.initialize(cfg)
    
    # 2. Load scene
    sim.load_scene(cfg.scene)
    
    # 3. Create and add robot
    robot = Go2Robot(cfg.robot, sim.scene)
    sim.add_robot(robot)
    
    # 4. Build scene
    sim.build()
    
    # 5. Run simulation loop
    print(f"Starting simulation, total {args.steps} steps...")
    for i in range(args.steps):
        sim.step()
        
        # Print status every 100 steps
        if i % 100 == 0:
            state = robot.get_state()
            print(f"Step {i}: pos={state.position}, time={sim.get_sim_time():.2f}s")
    
    print("Simulation completed!")


if __name__ == "__main__":
    main()
