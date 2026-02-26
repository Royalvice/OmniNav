# OmniNav Python Environment Installation

本文档说明 OmniNav 在两种场景下的 Python 环境安装流程：
- 非 ROS2 环境（普通 Python venv）
- ROS2 Humble 环境（venv + system-site-packages）

## -1. 硬件配置提示（参考 Isaac Sim 经验）

以下为 OmniNav 的运行门槛建议，参考 Isaac Sim 常见运行需求并结合 OmniNav 当前示例负载给出。

### 最小可运行配置（建议从这里起步）
1. CPU：4 核 8 线程（x86_64）
2. 内存：16 GB RAM
3. GPU：NVIDIA 独立显卡，显存 8 GB（支持 CUDA）
4. 存储：50 GB 可用 SSD 空间
5. 系统：Ubuntu 22.04（推荐）或等效 Linux 发行版

### 推荐开发配置（更稳定）
1. CPU：8 核 16 线程及以上
2. 内存：32 GB RAM 及以上
3. GPU：NVIDIA RTX 30/40 系列，显存 12 GB 及以上
4. 存储：100 GB+ NVMe SSD

说明：
- 上述“最小可运行配置”针对基础示例和单环境调试；复杂场景、高分辨率相机、多传感器与并行环境会显著提高硬件需求。
- 若显存不足，优先降低：`camera` 分辨率、传感器刷新频率、并行环境数（`n_envs`）与渲染负载。

## 0. 获取源码与子模块（两种场景通用）

### 0.1 安装 Git LFS（首次安装必做）

> 如果系统尚未安装 Git LFS，请先安装并初始化，再拉取仓库中的 LFS 文件。

Ubuntu/Debian:
```bash
sudo apt update
sudo apt install -y git-lfs
git lfs install
```

macOS (Homebrew):
```bash
brew install git-lfs
git lfs install
```

### 0.2 拉取源码、子模块与 LFS 内容

```bash
git clone https://github.com/Royalvice/OmniNav.git
cd OmniNav

git submodule update --init external/Genesis
git submodule update --init external/genesis_ros

cd external/Genesis
git submodule update --init doc
cd ../..

git lfs pull
```

## 1. 非 ROS2 环境安装

### 1.1 创建并激活虚拟环境

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 1.2 安装 PyTorch 与项目依赖

```bash
pip install --upgrade pip
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cuxxx
pip install -r requirements.txt
pip install -e .
```

说明：
- `cuxxx` 需要替换成你的 CUDA 版本通道（例如 `cu118`、`cu121`）。
- 如果你使用 CPU 版本，请改用 PyTorch 官方 CPU 安装索引。

## 2. ROS2 Humble 环境安装

### 2.0 安装 ROS2 / Nav2（仅 ROS2 场景需要）

如果你只使用纯 Python（不使用 ROS2/Nav2），可跳过本小节。

Ubuntu 22.04 + ROS2 Humble 推荐安装：
```bash
sudo apt update
sudo apt install -y \
  ros-humble-desktop \
  ros-humble-navigation2 \
  ros-humble-nav2-bringup
```

### 2.1 创建并激活虚拟环境（必须使用 system-site-packages）

```bash
python3 -m venv --system-site-packages ~/omninav_ros_env
source ~/omninav_ros_env/bin/activate
```

### 2.2 安装 Python 依赖

```bash
pip install --upgrade pip
pip install setuptools==77.0.1
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cuxxx
pip install -r requirements.txt
pip install -e .
```

### 2.3 固定 NumPy / OpenCV 兼容版本（推荐）

Genesis/Numba 与部分 ROS2 Python 依赖在 NumPy 2.x 下可能出现兼容问题，建议固定：

```bash
pip install numpy==1.26.4
pip install opencv-python==4.10.0.84
```

如果你使用 `opencv-contrib-python` 或 `opencv-python-headless`，建议也固定到 `4.10.0.84`。

可用以下命令确认版本：

```bash
python -c "import numpy,cv2; print(numpy.__version__, cv2.__version__)"
```

### 2.4 加载 ROS2 Humble 环境

```bash
source /opt/ros/humble/setup.bash
```

### 2.5 在 WSL2 下配置动态库路径（可选，但推荐）

在 WSL2 环境下，为避免与图形/仿真相关的动态库加载问题，建议在 **已激活 ROS2 虚拟环境** 的终端中执行：

```bash
echo 'export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

说明：
- 除了虚拟环境创建方式和最后 `source /opt/ros/humble/setup.bash` 外，其余步骤与非 ROS2 环境一致。
