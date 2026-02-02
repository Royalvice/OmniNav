# 第一个仿真

本教程将引导你运行一个简单的导航仿真示例。

## 准备工作

确保你已完成 [安装](installation.md) 步骤。

## 基础示例

创建一个新文件 `my_first_sim.py`：

```python
import omninav
from omninav import OmniNavEnv

# 初始化环境
env = OmniNavEnv(config_path="configs/config.yaml")

# 重置环境，获取初始观测
obs = env.reset()

print(f"机器人初始位置: {obs['robot_state'].position}")
print(f"目标位置: {obs.get('goal_position', 'N/A')}")

# 运行仿真循环
step_count = 0
while not env.is_done:
    # 使用配置文件中指定的算法计算动作
    action = env.algorithm.step(obs)
    
    # 执行一步仿真
    obs, info = env.step(action)
    step_count += 1
    
    if step_count % 100 == 0:
        print(f"Step {step_count}: 位置 = {obs['robot_state'].position[:2]}")

# 获取评测结果
result = env.get_result()
print(f"\n=== 仿真结束 ===")
print(f"成功: {result.success}")
print(f"总步数: {step_count}")
print(f"指标: {result.metrics}")

# 清理资源
env.close()
```

运行：

```bash
python my_first_sim.py
```

## 使用可视化

启用 Genesis Viewer 查看仿真过程：

```python
# 方法 1: 通过配置文件
# 在 configs/config.yaml 中设置:
# simulation:
#   show_viewer: true

# 方法 2: 通过代码覆盖
from omegaconf import OmegaConf

cfg = OmegaConf.load("configs/config.yaml")
cfg.simulation.show_viewer = True
env = OmniNavEnv(cfg=cfg)
```

## 自定义配置

你可以通过修改配置文件来自定义仿真：

```yaml
# configs/config.yaml
defaults:
  - robot: go2          # 使用 Go2 机器人
  - algorithm: apf      # 使用人工势场算法
  - task: point_nav     # 点到点导航任务

simulation:
  backend: "gpu"        # 使用 GPU 加速
  dt: 0.01              # 仿真步长
  show_viewer: true     # 显示可视化窗口
```

## 下一步

- 了解 [机器人配置](../user_guide/robots.md)
- 学习如何 [集成自定义算法](../user_guide/algorithms.md)
- 探索 [评测系统](../user_guide/evaluation.md)
