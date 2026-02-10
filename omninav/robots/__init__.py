"""
OmniNav Robots Layer

Provides abstract interfaces and concrete implementations for robots.
RobotState is now defined in omninav.core.types module.
"""

from omninav.robots.base import RobotBase
from omninav.robots.go2 import Go2Robot
from omninav.robots.go2w import Go2wRobot

__all__ = [
    "RobotBase",
    "Go2Robot",
    "Go2wRobot",
]
