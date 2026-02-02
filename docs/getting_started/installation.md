# 安装

本指南将帮助你安装 OmniNav 及其依赖项。

## 系统要求

| 组件 | 最低要求 | 推荐配置 |
|------|---------|---------|
| Python | 3.9+ | 3.10 / 3.11 |
| CUDA | 11.8+ (GPU 模式) | 12.x |
| 内存 | 8 GB | 16 GB+ |
| 显卡 | - | NVIDIA RTX 系列 |

## 安装步骤

### 1. 克隆仓库

```bash
git clone --recurse-submodules https://github.com/Royalvice/OmniNav.git
cd OmniNav
```

!!! note "子模块"
    `--recurse-submodules` 参数会自动拉取 Genesis 和 genesis_ros 子模块。
    如果忘记添加此参数，可以稍后运行：
    ```bash
    git submodule update --init --recursive
    ```

### 2. 创建虚拟环境

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

### 3. 安装 Genesis

```bash
cd external/Genesis
pip install -e .
cd ../..
```

!!! tip "GPU 支持"
    Genesis 默认支持 CPU 后端。如需 GPU 加速，请确保已安装 CUDA 并正确配置 PyTorch。

### 4. 安装 OmniNav

```bash
pip install -e .
```

### 5. (可选) 安装 ROS2 支持

如果你需要使用 ROS2 桥接功能：

1. 确保已安装 ROS2 Humble
2. Source ROS2 环境
3. 安装 genesis_ros：

```bash
cd external/genesis_ros
colcon build
source install/setup.bash
```

## 验证安装

```python
import omninav
print(omninav.__version__)
```

如果没有报错，说明安装成功！

## 下一步

- 继续阅读 [第一个仿真](first_simulation.md) 教程
- 了解 [架构概览](../user_guide/architecture.md)
