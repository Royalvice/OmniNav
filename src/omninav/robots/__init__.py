"""
OmniNav Robots Layer - 机器人管理

提供机器人和传感器的抽象基类及具体实现。
"""

from omninav.robots.base import RobotBase, SensorBase, RobotState, SensorMount

__all__ = [
    "RobotBase",
    "SensorBase",
    "RobotState",
    "SensorMount",
]
