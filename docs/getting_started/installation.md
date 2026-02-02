# Installation

This guide will help you install OmniNav and its dependencies.

## System Requirements

| Component | Minimum | Recommended |
|------|---------|---------|
| Python | 3.9+ | 3.10 / 3.11 |
| CUDA | 11.8+ (GPU mode) | 12.x |
| Memory | 8 GB | 16 GB+ |
| GPU | - | NVIDIA RTX Series |

## Installation Steps

### 1. Clone Repository

```bash
git clone --recurse-submodules https://github.com/Royalvice/OmniNav.git
cd OmniNav
```

!!! note "Submodules"
    The `--recurse-submodules` flag automatically pulls Genesis and genesis_ros submodules.
    If you forgot this flag, you can run later:
    ```bash
    git submodule update --init --recursive
    ```

### 2. Create Virtual Environment

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

### 3. Install Genesis

```bash
cd external/Genesis
pip install -e .
cd ../..
```

!!! tip "GPU Support"
    Genesis supports CPU backend by default. For GPU acceleration, ensure CUDA is installed and PyTorch is configured correctly.

### 4. Install OmniNav

```bash
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
