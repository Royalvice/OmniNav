# 更新日志

所有重要更改将记录在此文件中。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [未发布]

### 新增
- 初始项目结构
- 核心层：SimulationManager 基类
- 机器人层：RobotBase, SensorBase 基类
- 算法层：AlgorithmBase 基类与航点跟随算法
- 评测层：TaskBase, MetricBase 基类与点导航任务
- 接口层：OmniNavEnv (Gym-like API)
- 配置管理：Hydra/OmegaConf 集成
- 文档：MkDocs + Material 主题

### 计划中
- 运动学控制器实现
- Go2/Go2w 机器人支持
- ROS2 桥接模块
- 更多导航算法示例
