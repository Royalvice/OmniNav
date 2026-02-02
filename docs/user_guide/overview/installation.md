# Installation

This guide will help you install OmniNav and its dependencies.

## System Requirements

| Component | Minimum | Recommended |
|------|---------|---------|
| Python | > 3.10 | 3.13 |
| CUDA | 11.8+ (GPU mode) | 12.x |
| Memory | 8 GB | 16 GB+ |
| GPU | - | NVIDIA RTX Series |

## Installation Steps

### 1. Clone Repository

```bash
git clone https://github.com/Royalvice/OmniNav.git
cd OmniNav
```

### 2. Initialize Submodules

Only initialize the necessary submodules (Genesis is required).

```bash
git submodule update --init external/Genesis
```

If you need ROS2 support (Optional):

```bash
git submodule update --init external/genesis_ros
```

### 3. Create Virtual Environment

=== "Linux / macOS"
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

=== "Windows"
    ```powershell
    python -m venv .venv
    .venv\Scripts\activate
    ```

### 4. Install Dependencies & OmniNav

```bash
pip install -r requirements.txt
pip install -e .
```

### 5. (Optional) Install ROS2 Support

If you need ROS2 bridge functionality:

1. Ensure ROS2 Humble is installed
2. Source ROS2 environment
3. Install genesis_ros:

```bash
cd external/genesis_ros
colcon build
source install/setup.bash
```

## Verify Installation

```python
import omninav
print(omninav.__version__)
```

If no error occurs, installation is successful!

## Next Steps

- Continue to [First Simulation](first_simulation.md) tutorial
- Learn about [Architecture Overview](../user_guide/architecture.md)
