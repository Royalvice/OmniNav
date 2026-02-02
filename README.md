# OmniNav

<p align="center">
  <strong>面向具身智能的通用仿真平台</strong>
</p>

<p align="center">
  <a href="https://github.com/Royalvice/OmniNav">
    <img src="https://img.shields.io/github/stars/Royalvice/OmniNav?style=social" alt="GitHub stars">
  </a>
  <a href="https://github.com/Royalvice/OmniNav/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="License">
  </a>
</p>

---

## ✨ 什么是 OmniNav？

OmniNav 是一个基于 [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) 物理引擎的**具身智能仿真平台**，专为机器人导航与避障算法的快速验证而设计。

### 核心特性

- 🚀 **高性能仿真** - 基于 Genesis 引擎，支持 GPU 加速
- 🔌 **算法可插拔** - 传统算法、VLA/VLN 等神经网络算法均可快速集成
- 📊 **内置评测系统** - 预定义导航任务与评价指标 (SPL, Success Rate 等)
- 🤖 **多机器人支持** - 初版支持宇树 Go2 (四足/轮式)
- 🌐 **ROS2 兼容** - 可选的 ROS2 桥接，支持 Sim2Real
- 📦 **场景资产导入** - 支持 USD、GLB、OBJ 等格式

---

## 🚀 快速开始

### 安装

```bash
# 克隆仓库
git clone --recurse-submodules https://github.com/Royalvice/OmniNav.git
cd OmniNav

# 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/macOS

# 安装 Genesis
cd external/Genesis && pip install -e . && cd ../..

# 安装 OmniNav
pip install -e .
```

### 运行示例

```python
from omninav import OmniNavEnv

env = OmniNavEnv(config_path="configs")
obs = env.reset()

while not env.is_done:
    action = env.algorithm.step(obs)
    obs, info = env.step(action)

result = env.get_result()
print(f"Success: {result.success}")
```

---

## 📖 文档

完整文档: [https://royalvice.github.io/OmniNav/](https://royalvice.github.io/OmniNav/)

---

## 🏗️ 架构

```
┌─────────────────────────────────────────────────────────┐
│                    接口层 (Interface)                    │
├─────────────────────────────────────────────────────────┤
│  评测层 (Evaluation)  │  算法层 (Algorithm - 可插拔)     │
├─────────────────────────────────────────────────────────┤
│              运动层 (Locomotion Controller)              │
├─────────────────────────────────────────────────────────┤
│  机器人层 (Robot)     │   资产层 (Asset Loader)          │
├─────────────────────────────────────────────────────────┤
│                    核心层 (Genesis Wrapper)              │
└─────────────────────────────────────────────────────────┘
```

---

## 📝 许可证

[Apache-2.0](LICENSE)

## 🙏 致谢

- [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) - 高性能物理仿真引擎
