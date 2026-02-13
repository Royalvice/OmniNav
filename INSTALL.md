# OmniNav Python Environment Installation

本文档说明 OmniNav 在两种场景下的 Python 环境安装流程：
- 非 ROS2 环境（普通 Python venv）
- ROS2 Humble 环境（venv + system-site-packages）

## 0. 获取源码与子模块（两种场景通用）

```bash
git clone https://github.com/Royalvice/OmniNav.git
cd OmniNav

git submodule update --init external/Genesis
git submodule update --init external/genesis_ros

cd external/Genesis
git submodule update --init doc
cd ../..
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

### 2.3 加载 ROS2 Humble 环境

```bash
source /opt/ros/humble/setup.bash
```

### 2.4 在 WSL2 下配置动态库路径（可选，但推荐）

在 WSL2 环境下，为避免与图形/仿真相关的动态库加载问题，建议在 **已激活 ROS2 虚拟环境** 的终端中执行：

```bash
echo 'export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

说明：
- 除了虚拟环境创建方式和最后 `source /opt/ros/humble/setup.bash` 外，其余步骤与非 ROS2 环境一致。
