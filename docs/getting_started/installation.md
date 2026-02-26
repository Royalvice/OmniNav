# Installation

This page summarizes user installation. The source of truth is:
- [INSTALL.md](https://github.com/Royalvice/OmniNav/blob/main/INSTALL.md)

## 1. Clone and prerequisites

```bash
git clone https://github.com/Royalvice/OmniNav.git
cd OmniNav

git lfs install
git submodule update --init external/Genesis
git submodule update --init external/genesis_ros

cd external/Genesis
git submodule update --init doc
cd ../..

git lfs pull
```

## 2. Pure Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cuxxx
pip install -r requirements.txt
pip install -e .
```

## 3. ROS2/Nav2 environment (optional)

```bash
sudo apt update
sudo apt install -y \
  ros-humble-desktop \
  ros-humble-navigation2 \
  ros-humble-nav2-bringup

python3 -m venv --system-site-packages ~/omninav_ros_env
source ~/omninav_ros_env/bin/activate
source /opt/ros/humble/setup.bash
```

## 4. Verify

```bash
python -c "import omninav; print('omninav import ok')"
python examples/05_waypoint_navigation.py --test-mode --smoke-fast --max-steps 40 --no-show-viewer
```
