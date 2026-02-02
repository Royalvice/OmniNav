# OmniNav Phase 2 Implementation Walkthrough

> Phase 2 核心框架已完成实现并通过全部 32 个测试用例。

---

## 1. Registry 机制 (Core Layer)

### 文件
- [registry.py](file:///e:/code/python/OmniNav/omninav/core/registry.py) - 通用组件注册器

### 功能
```python
from omninav.core.registry import ROBOT_REGISTRY, SENSOR_REGISTRY

# 注册组件
@SENSOR_REGISTRY.register("lidar_2d")
class Lidar2DSensor(SensorBase):
    pass

# 从配置构建
sensor = SENSOR_REGISTRY.build(cfg, scene=scene, robot=robot)
```

### 全局注册器
| Registry | 用途 |
|----------|------|
| `ROBOT_REGISTRY` | Go2, Go2w 等机器人 |
| `SENSOR_REGISTRY` | Lidar, Camera 等传感器 |
| `LOCOMOTION_REGISTRY` | WheelController, IKController 等 |
| `ALGORITHM_REGISTRY` | 导航/感知算法 |
| `TASK_REGISTRY` | 评测任务 |
| `METRIC_REGISTRY` | 评测指标 |

---

## 2. Sensor Layer

### 文件
| File | Description |
|------|-------------|
| [base.py](file:///e:/code/python/OmniNav/omninav/sensors/base.py) | `SensorBase` 抽象基类 |
| [lidar.py](file:///e:/code/python/OmniNav/omninav/sensors/lidar.py) | `Lidar2DSensor` - 2D 激光雷达 |
| [camera.py](file:///e:/code/python/OmniNav/omninav/sensors/camera.py) | `CameraSensor` - RGB-D 相机 |

### SensorBase 生命周期
```
__init__(cfg, scene, robot)  # 存储配置
    ↓
attach(link_name, pos, euler)  # 设置挂载点
    ↓
create()  # 创建 Genesis 传感器 (scene.build 前)
    ↓
get_data()  # 读取传感器数据 (scene.step 后)
```

### Lidar2DSensor 实现
- 使用 `gs.sensors.Lidar` + `SphericalPattern(fov=(360, 0.5), n_points=(720, 1))`
- 输出: `{'ranges': np.ndarray, 'points': np.ndarray}`

### CameraSensor 实现
- 使用 `gs.vis.camera.Camera`
- 输出: `{'rgb': np.ndarray, 'depth': np.ndarray}`

### 配置文件
- [lidar_2d.yaml](file:///e:/code/python/OmniNav/configs/sensor/lidar_2d.yaml)
- [camera_rgbd.yaml](file:///e:/code/python/OmniNav/configs/sensor/camera_rgbd.yaml)

---

## 3. Locomotion Layer

### 文件
| File | Description |
|------|-------------|
| [wheel_controller.py](file:///e:/code/python/OmniNav/omninav/locomotion/wheel_controller.py) | Go2w 轮式控制 |
| [ik_controller.py](file:///e:/code/python/OmniNav/omninav/locomotion/ik_controller.py) | Go2 IK 步态控制 |
| [rl_controller.py](file:///e:/code/python/OmniNav/omninav/locomotion/rl_controller.py) | RL 控制 (占位) |

### WheelController (Go2w)
- Mecanum 轮逆运动学: `[vx, vy, wz] → [FL, FR, RL, RR] wheel velocities`
- 使用 `robot.entity.control_dofs_velocity()`

### IKController (Go2)
- Trot 步态: 对角腿同步 (FL+RR, FR+RL)
- Bezier 曲线生成足端摆动轨迹
- 使用 Genesis `entity.inverse_kinematics()` API
- 使用 `robot.entity.control_dofs_position()`

### RLController (占位)
- 接口已定义，调用时抛出 `NotImplementedError`
- 预留用于集成 RL 策略 (如 Legged Gym)

### 配置文件
- [wheel.yaml](file:///e:/code/python/OmniNav/configs/locomotion/wheel.yaml)
- [ik_gait.yaml](file:///e:/code/python/OmniNav/configs/locomotion/ik_gait.yaml)

---

## 4. ROS2 Interface Layer

### 文件
- [bridge.py](file:///e:/code/python/OmniNav/omninav/interfaces/ros2/bridge.py)

### Ros2Bridge 功能
| 类型 | Topic | Message Type |
|------|-------|--------------|
| Publisher | `/clock` | `rosgraph_msgs/Clock` |
| Publisher | `/scan` | `sensor_msgs/LaserScan` |
| Publisher | `/camera/image_raw` | `sensor_msgs/Image` |
| Publisher | `/camera/depth` | `sensor_msgs/Image` |
| Publisher | `/odom` | `nav_msgs/Odometry` |
| Subscriber | `/cmd_vel` | `geometry_msgs/Twist` |

### 使用方式
```python
bridge = Ros2Bridge(cfg.ros2, sim_manager)
bridge.setup(robot)

while running:
    sim_manager.step()
    bridge.spin_once()
    cmd_vel = bridge.get_cmd_vel()
```

---

## 5. 测试结构

```
tests/
├── conftest.py              # Mock Genesis 对象 & Pytest fixtures
├── core/
│   └── test_registry.py     # 9 tests - Registry 机制
├── sensors/
│   └── test_sensors.py      # 8 tests - SensorBase, Lidar, Camera
├── locomotion/
│   └── test_locomotion.py   # 9 tests - Wheel, IK, RL controllers
├── interfaces/
│   └── test_ros2_bridge.py  # 4 tests - ROS2 Bridge (mock-based)
└── robots/
    └── (待添加集成测试)
```

### 测试结果
```
32 passed in 0.28s
```

---

## 6. Genesis API 使用模式

```python
import genesis as gs

# 1. 初始化
gs.init(backend=gs.gpu)

# 2. 创建场景
scene = gs.Scene(sim_options=gs.options.SimOptions(dt=0.01))

# 3. 添加实体
robot = scene.add_entity(gs.morphs.URDF(file="path/to/robot.urdf"))

# 4. 添加传感器 (build 前)
lidar = scene.add_sensor(gs.sensors.Lidar(...))

# 5. 构建
scene.build(n_envs=1)

# 6. 控制循环
for _ in range(1000):
    robot.control_dofs_position(targets)
    scene.step()
    data = lidar.read()
```

---

## 7. 后续任务

### Phase 2.5 Integration
- [ ] 更新 `Go2Robot` / `Go2wRobot` 支持 Sensor 挂载
- [ ] 创建 `tests/robots/test_robot_sensor_integration.py`

### Phase 3 算法与验证
- [ ] 实现 Waypoint Follower 算法
- [ ] 实现 Evaluation Layer 框架
- [ ] 全流程验证 (Sim2Real 预备)

### Phase 4 文档与示例
- [ ] 编写用户文档 (docs/)
- [ ] 创建示例脚本 (examples/)
