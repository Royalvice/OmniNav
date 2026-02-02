# First Simulation

This tutorial will guide you through running a simple navigation simulation example.

## Prerequisites

Ensure you have completed the [Installation](installation.md) steps.

## Basic Example

Create a new file `my_first_sim.py`:

```python
import omninav
from omninav import OmniNavEnv

# Initialize environment
env = OmniNavEnv(config_path="configs/config.yaml")

# Reset environment, get initial observation
obs = env.reset()

print(f"Robot initial position: {obs['robot_state'].position}")
print(f"Goal position: {obs.get('goal_position', 'N/A')}")

# Run simulation loop
step_count = 0
while not env.is_done:
    # Calculate action using algorithm specified in config
    action = env.algorithm.step(obs)
    
    # Execute one simulation step
    obs, info = env.step(action)
    step_count += 1
    
    if step_count % 100 == 0:
        print(f"Step {step_count}: Position = {obs['robot_state'].position[:2]}")

# Get evaluation result
result = env.get_result()
print(f"\n=== Simulation Ended ===")
print(f"Success: {result.success}")
print(f"Total steps: {step_count}")
print(f"Metrics: {result.metrics}")

# Cleanup resources
env.close()
```

Run:

```bash
python my_first_sim.py
```

## Using Visualization

Enable Genesis Viewer to see the simulation process:

```python
# Method 1: Via config file
# Set in configs/config.yaml:
# simulation:
#   show_viewer: true

# Method 2: Via code override
from omegaconf import OmegaConf

cfg = OmegaConf.load("configs/config.yaml")
cfg.simulation.show_viewer = True
env = OmniNavEnv(cfg=cfg)
```

## Custom Configuration

You can custom the simulation by modifying the config file:

```yaml
# configs/config.yaml
defaults:
  - robot: go2          # Use Go2 robot
  - algorithm: apf      # Use Artificial Potential Field algorithm
  - task: point_nav     # Point-to-Point Navigation task

simulation:
  backend: "gpu"        # Use GPU acceleration
  dt: 0.01              # Simulation step size
  show_viewer: true     # Show visualization window
```

## Next Steps

- Check out [Robot Configuration](../user_guide/robots.md)
- Learn how to [Integrate Custom Algorithms](../user_guide/algorithms.md)
- Explore [Evaluation System](../user_guide/evaluation.md)
