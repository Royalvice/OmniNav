# Installation

<div class="lang-zh">

本页是用户安装入口。安装细节的唯一事实源是仓库根目录 [`INSTALL.md`](https://github.com/Royalvice/OmniNav/blob/main/INSTALL.md)。

## 1. 硬件建议

请先阅读 `INSTALL.md` 中的硬件配置提示（最小可运行配置 + 推荐开发配置）。

## 2. 获取源码

```bash
git clone https://github.com/Royalvice/OmniNav.git
cd OmniNav
```

## 3. Git LFS + 子模块

```bash
git lfs install
git submodule update --init external/Genesis
git submodule update --init external/genesis_ros

cd external/Genesis
git submodule update --init doc
cd ../..

git lfs pull
```

## 4. 纯 Python 环境

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cuxxx
pip install -r requirements.txt
pip install -e .
```

## 5. ROS2 / Nav2 环境（可选）

```bash
sudo apt update
sudo apt install -y \
  ros-humble-desktop \
  ros-humble-navigation2 \
  ros-humble-nav2-bringup

python3 -m venv --system-site-packages ~/omninav_ros_env
source ~/omninav_ros_env/bin/activate
source /opt/ros/humble/setup.bash

pip install --upgrade pip
pip install setuptools==77.0.1
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cuxxx
pip install -r requirements.txt
pip install -e .
```

## 6. 推荐验证

```bash
python -c "import omninav; print('omninav import ok')"
python examples/05_waypoint_navigation.py --test-mode --smoke-fast --max-steps 40 --no-show-viewer
```

更多安装场景（WSL2、NumPy/OpenCV 固定版本）请直接看：[`INSTALL.md`](https://github.com/Royalvice/OmniNav/blob/main/INSTALL.md)。

</div>

<div class="lang-en">

This is the user-facing installation entry. The source of truth is repository root [`INSTALL.md`](https://github.com/Royalvice/OmniNav/blob/main/INSTALL.md).

## 1. Hardware guidance

Read hardware notes in [`INSTALL.md`](https://github.com/Royalvice/OmniNav/blob/main/INSTALL.md) first (minimum runnable + recommended dev setup).

## 2. Clone source

```bash
git clone https://github.com/Royalvice/OmniNav.git
cd OmniNav
```

## 3. Git LFS + submodules

```bash
git lfs install
git submodule update --init external/Genesis
git submodule update --init external/genesis_ros

cd external/Genesis
git submodule update --init doc
cd ../..

git lfs pull
```

## 4. Pure Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cuxxx
pip install -r requirements.txt
pip install -e .
```

## 5. ROS2 / Nav2 environment (optional)

```bash
sudo apt update
sudo apt install -y \
  ros-humble-desktop \
  ros-humble-navigation2 \
  ros-humble-nav2-bringup

python3 -m venv --system-site-packages ~/omninav_ros_env
source ~/omninav_ros_env/bin/activate
source /opt/ros/humble/setup.bash

pip install --upgrade pip
pip install setuptools==77.0.1
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cuxxx
pip install -r requirements.txt
pip install -e .
```

## 6. Quick verification

```bash
python -c "import omninav; print('omninav import ok')"
python examples/05_waypoint_navigation.py --test-mode --smoke-fast --max-steps 40 --no-show-viewer
```

For WSL2 and NumPy/OpenCV compatibility pins, follow [`INSTALL.md`](https://github.com/Royalvice/OmniNav/blob/main/INSTALL.md) directly.

</div>
