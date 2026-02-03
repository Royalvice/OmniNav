"""
OmniNav Locomotion Layer - Motion Control

Provides abstract interface for converting high-level velocity commands to joint control.
"""

from omninav.locomotion.base import LocomotionControllerBase
from omninav.locomotion.wheel_controller import WheelController
from omninav.locomotion.ik_controller import IKController
from omninav.locomotion.rl_controller import RLController
from omninav.locomotion.simple_gait import SimpleGaitController

__all__ = [
    "LocomotionControllerBase",
    "WheelController",
    "IKController",
    "RLController",
    "SimpleGaitController",
]
