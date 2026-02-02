"""
OmniNav Robots Layer

Provides abstract interfaces and concrete implementations for robots.
Sensors are moved to omninav.sensors module.
"""

from omninav.robots.base import RobotBase, RobotState
from omninav.robots.go2 import Go2Robot
from omninav.robots.go2w import Go2wRobot

__all__ = [
    "RobotBase",
    "RobotState",
    "Go2Robot",
    "Go2wRobot",
]
