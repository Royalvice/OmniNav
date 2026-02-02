"""
OmniNav 简单示例 - Go2 机器人仿真

演示如何使用 OmniNav 加载 Go2 机器人并运行仿真。

用法:
    python examples/simple_go2.py
    python examples/simple_go2.py --cpu  # 使用 CPU 后端
"""

import argparse
from omegaconf import OmegaConf

from omninav.core import GenesisSimulationManager
from omninav.robots import Go2Robot


def main():
    parser = argparse.ArgumentParser(description="OmniNav Go2 机器人仿真示例")
    parser.add_argument("--cpu", action="store_true", help="使用 CPU 后端")
    parser.add_argument("--steps", type=int, default=1000, help="仿真步数")
    args = parser.parse_args()
    
    # 创建配置
    cfg = OmegaConf.create({
        "simulation": {
            "dt": 0.01,
            "substeps": 2,
            "backend": "cpu" if args.cpu else "gpu",
            "n_envs": 1,
            "show_viewer": True,
            "enable_self_collision": False,
            "camera_pos": [2.0, 0.0, 2.5],
            "camera_lookat": [0.0, 0.0, 0.0],
            "camera_fov": 40,
        },
        "robot": {
            "urdf_source": "genesis_builtin",
            "urdf_path": "urdf/go2/urdf/go2.urdf",
            "initial_pos": [0.0, 0.0, 0.4],
            "initial_quat": [1.0, 0.0, 0.0, 0.0],
            "control": {
                "kp": 20.0,
                "kd": 0.5,
            }
        },
        "scene": {
            "ground_plane": {"enabled": True},
            "obstacles": [],
        }
    })
    
    # 1. 创建仿真管理器
    sim = GenesisSimulationManager()
    sim.initialize(cfg)
    
    # 2. 加载场景
    sim.load_scene(cfg.scene)
    
    # 3. 创建并添加机器人
    robot = Go2Robot(cfg.robot, sim.scene)
    sim.add_robot(robot)
    
    # 4. 构建场景
    sim.build()
    
    # 5. 运行仿真循环
    print(f"开始仿真，共 {args.steps} 步...")
    for i in range(args.steps):
        sim.step()
        
        # 每 100 步打印状态
        if i % 100 == 0:
            state = robot.get_state()
            print(f"Step {i}: pos={state.position}, time={sim.get_sim_time():.2f}s")
    
    print("仿真完成！")


if __name__ == "__main__":
    main()
