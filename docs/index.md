# OmniNav

```{toctree}
:maxdepth: 2
:caption: 快速开始

getting_started/installation
getting_started/first_simulation
```

```{toctree}
:maxdepth: 2
:caption: 用户指南

user_guide/architecture
user_guide/robots
user_guide/sensors
user_guide/scenes
user_guide/algorithms
user_guide/evaluation
user_guide/ros2_integration
```

```{toctree}
:maxdepth: 2
:caption: API 参考

api_reference/core
api_reference/robots
api_reference/algorithms
api_reference/evaluation
```

```{toctree}
:maxdepth: 1
:caption: 其他

contributing
changelog
```

## ✨ 什么是 OmniNav？

OmniNav 是一个基于 [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) 物理引擎的**具身智能仿真平台**，专为机器人导航与避障算法的快速验证而设计。

### 核心特性

- 🚀 **高性能仿真** - 基于 Genesis 引擎，支持 GPU 加速
- 🔌 **算法可插拔** - 传统算法、VLA/VLN 等神经网络算法均可快速集成
- 📊 **内置评测系统** - 预定义导航任务与评价指标 (SPL, Success Rate 等)
- 🤖 **多机器人支持** - 初版支持宇树 Go2 (四足/轮式)
- 🌐 **ROS2 兼容** - 可选的 ROS2 桥接，支持 Sim2Real
- 📦 **场景资产导入** - 支持 USD、GLB、OBJ 等格式

## 🚀 快速开始

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
