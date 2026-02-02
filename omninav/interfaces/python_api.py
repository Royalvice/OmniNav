"""
OmniNav Python API - 主接口类

提供类 Gym 风格的仿真环境接口。
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from omninav.robots.base import RobotBase, RobotState
from omninav.locomotion.base import LocomotionControllerBase
from omninav.algorithms.base import AlgorithmBase
from omninav.evaluation.base import TaskBase, TaskResult


class OmniNavEnv:
    """
    OmniNav 主接口类 (类 Gym 风格)。
    
    提供简洁的 API 用于:
    - 创建仿真环境
    - 控制机器人
    - 运行评测任务
    
    使用示例:
        >>> from omninav import OmniNavEnv
        >>> 
        >>> env = OmniNavEnv(config_path="configs")
        >>> obs = env.reset()
        >>> 
        >>> while not env.is_done:
        ...     action = env.algorithm.step(obs)  # 或自定义算法
        ...     obs, info = env.step(action)
        >>> 
        >>> result = env.get_result()
        >>> print(f"Success: {result.success}")
    """
    
    def __init__(
        self, 
        cfg: Optional[DictConfig] = None,
        config_path: Optional[str] = None,
        config_name: str = "config",
    ):
        """
        初始化 OmniNav 环境。
        
        Args:
            cfg: 直接传入的配置对象 (优先级最高)
            config_path: Hydra 配置目录路径
            config_name: 配置文件名 (不含扩展名)
        """
        self.cfg = self._load_config(cfg, config_path, config_name)
        
        # 组件引用 (延迟初始化)
        self.sim = None  # SimulationManager
        self.robot: Optional[RobotBase] = None
        self.locomotion: Optional[LocomotionControllerBase] = None
        self.algorithm: Optional[AlgorithmBase] = None
        self.task: Optional[TaskBase] = None
        
        self._initialized = False
        self._step_count = 0
    
    def _load_config(
        self, 
        cfg: Optional[DictConfig],
        config_path: Optional[str],
        config_name: str,
    ) -> DictConfig:
        """加载配置。"""
        if cfg is not None:
            return cfg
        
        if config_path is not None:
            # 使用 Hydra 组合配置
            try:
                from hydra import compose, initialize_config_dir
                from hydra.core.global_hydra import GlobalHydra
                
                # 清理可能存在的 Hydra 实例
                if GlobalHydra.instance().is_initialized():
                    GlobalHydra.instance().clear()
                
                config_dir = str(Path(config_path).absolute())
                with initialize_config_dir(config_dir=config_dir, version_base=None):
                    cfg = compose(config_name=config_name)
                return cfg
            except ImportError:
                # 回退到直接加载 YAML
                import yaml
                config_file = Path(config_path) / f"{config_name}.yaml"
                with open(config_file) as f:
                    return OmegaConf.create(yaml.safe_load(f))
        
        # 默认空配置
        return OmegaConf.create({})
    
    def _initialize(self) -> None:
        """
        根据配置初始化所有组件。
        
        延迟初始化，在第一次 reset() 时调用。
        """
        if self._initialized:
            return
        
        # TODO: 实现各组件的初始化
        # 1. 初始化 SimulationManager
        # 2. 加载场景
        # 3. 创建机器人
        # 4. 挂载传感器
        # 5. 创建运动控制器
        # 6. 创建算法 (可选)
        # 7. 创建评测任务 (可选)
        # 8. 构建场景
        
        self._initialized = True
    
    def reset(self) -> Dict[str, Any]:
        """
        重置环境。
        
        Returns:
            初始观测
        """
        if not self._initialized:
            self._initialize()
        
        self._step_count = 0
        
        # 重置仿真
        if self.sim is not None:
            self.sim.reset()
        
        # 重置运动控制器
        if self.locomotion is not None:
            self.locomotion.reset()
        
        # 重置任务并获取任务信息
        task_info = {}
        if self.task is not None:
            task_info = self.task.reset()
        
        # 重置算法
        if self.algorithm is not None:
            self.algorithm.reset(task_info)
        
        return self._get_observation()
    
    def step(self, action: Optional[np.ndarray] = None) -> Tuple[Dict[str, Any], Dict]:
        """
        执行一步仿真。
        
        Args:
            action: cmd_vel [vx, vy, wz]，如果为 None 则使用内置算法
        
        Returns:
            obs: 新的观测
            info: 额外信息
        """
        # 如果没有传入 action，使用内置算法
        if action is None and self.algorithm is not None:
            obs = self._get_observation()
            action = self.algorithm.step(obs)
        
        if action is None:
            action = np.zeros(3)
        
        # 运动控制
        if self.locomotion is not None:
            self.locomotion.step(action)
        
        # 物理仿真
        if self.sim is not None:
            self.sim.step()
        
        self._step_count += 1
        
        # 记录任务数据
        if self.task is not None and self.robot is not None:
            robot_state = self.robot.get_state()
            self.task.step(robot_state, action)
        
        obs = self._get_observation()
        info = {"action": action, "step": self._step_count}
        
        return obs, info
    
    def _get_observation(self) -> Dict[str, Any]:
        """获取当前观测。"""
        obs = {}
        
        if self.robot is not None:
            obs.update(self.robot.get_observations())
            obs["robot_state"] = self.robot.get_state()
        
        if self.sim is not None:
            obs["sim_time"] = self.sim.get_sim_time()
        
        return obs
    
    @property
    def is_done(self) -> bool:
        """任务是否结束。"""
        if self.task is not None and self.robot is not None:
            return self.task.is_terminated(self.robot.get_state())
        if self.algorithm is not None:
            return self.algorithm.is_done
        return False
    
    def get_result(self) -> Optional[TaskResult]:
        """获取任务结果。"""
        if self.task is not None:
            return self.task.compute_result()
        return None
    
    def close(self) -> None:
        """关闭环境，释放资源。"""
        if self.sim is not None:
            # self.sim.scene.destroy()
            pass
        self._initialized = False
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
