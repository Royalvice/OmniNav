"""
Tests for Locomotion controllers.
"""

import pytest
import numpy as np
from omegaconf import OmegaConf
from unittest.mock import MagicMock


class TestWheelController:
    """Test suite for WheelController."""
    
    def test_compute_action_forward(self, mock_robot, sample_wheel_cfg):
        """Test forward motion produces positive wheel velocities."""
        from omninav.locomotion.wheel_controller import WheelController
        
        # Add wheel joints to mock entity
        mock_robot.entity.joint_names = [
            "FL_wheel_joint", "FR_wheel_joint",
            "RL_wheel_joint", "RR_wheel_joint",
        ]
        
        controller = WheelController(sample_wheel_cfg, mock_robot)
        
        cmd_vel = np.array([1.0, 0.0, 0.0])  # Forward
        wheel_velocities = controller.compute_action(cmd_vel)
        
        # All wheels should rotate forward (positive)
        assert wheel_velocities.shape == (4,)
        assert np.all(wheel_velocities > 0), "Forward motion should produce positive wheel velocities"
    
    def test_compute_action_backward(self, mock_robot, sample_wheel_cfg):
        """Test backward motion produces negative wheel velocities."""
        from omninav.locomotion.wheel_controller import WheelController
        
        mock_robot.entity.joint_names = [
            "FL_wheel_joint", "FR_wheel_joint",
            "RL_wheel_joint", "RR_wheel_joint",
        ]
        
        controller = WheelController(sample_wheel_cfg, mock_robot)
        
        cmd_vel = np.array([-1.0, 0.0, 0.0])  # Backward
        wheel_velocities = controller.compute_action(cmd_vel)
        
        assert np.all(wheel_velocities < 0), "Backward motion should produce negative wheel velocities"
    
    def test_compute_action_turn_in_place(self, mock_robot, sample_wheel_cfg):
        """Test turning in place produces differential wheel velocities."""
        from omninav.locomotion.wheel_controller import WheelController
        
        mock_robot.entity.joint_names = [
            "FL_wheel_joint", "FR_wheel_joint",
            "RL_wheel_joint", "RR_wheel_joint",
        ]
        
        controller = WheelController(sample_wheel_cfg, mock_robot)
        
        cmd_vel = np.array([0.0, 0.0, 1.0])  # Turn left (CCW)
        wheel_velocities = controller.compute_action(cmd_vel)
        
        # For CCW rotation: left wheels backward, right wheels forward
        # FL and RL should be negative, FR and RR should be positive
        assert wheel_velocities[0] < 0  # FL
        assert wheel_velocities[1] > 0  # FR
        assert wheel_velocities[2] < 0  # RL
        assert wheel_velocities[3] > 0  # RR
    
    def test_compute_action_zero(self, mock_robot, sample_wheel_cfg):
        """Test zero velocity produces zero wheel velocities."""
        from omninav.locomotion.wheel_controller import WheelController
        
        mock_robot.entity.joint_names = [
            "FL_wheel_joint", "FR_wheel_joint",
            "RL_wheel_joint", "RR_wheel_joint",
        ]
        
        controller = WheelController(sample_wheel_cfg, mock_robot)
        
        cmd_vel = np.array([0.0, 0.0, 0.0])
        wheel_velocities = controller.compute_action(cmd_vel)
        
        assert np.allclose(wheel_velocities, 0.0)
    
    def test_wheel_speed_clipping(self, mock_robot, sample_wheel_cfg):
        """Test wheel velocities are clipped to max speed."""
        from omninav.locomotion.wheel_controller import WheelController
        
        mock_robot.entity.joint_names = [
            "FL_wheel_joint", "FR_wheel_joint",
            "RL_wheel_joint", "RR_wheel_joint",
        ]
        
        controller = WheelController(sample_wheel_cfg, mock_robot)
        max_speed = sample_wheel_cfg.max_wheel_speed
        
        # Very high velocity command
        cmd_vel = np.array([100.0, 0.0, 0.0])
        wheel_velocities = controller.compute_action(cmd_vel)
        
        assert np.all(np.abs(wheel_velocities) <= max_speed)


class TestIKController:
    """Test suite for IKController."""
    
    def test_phase_update(self, mock_robot):
        """Test that phase updates correctly."""
        from omninav.locomotion.ik_controller import IKController
        
        cfg = OmegaConf.create({
            "type": "ik_controller",
            "gait_frequency": 2.0,
            "dt": 0.01,
            "foot_links": ["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
        })
        
        controller = IKController(cfg, mock_robot)
        
        initial_phase = controller._phase
        controller.step(np.zeros(3))
        
        # Phase should have increased
        expected_phase = (initial_phase + 0.01 * 2.0) % 1.0
        assert abs(controller._phase - expected_phase) < 1e-6
    
    def test_reset_clears_phase(self, mock_robot):
        """Test that reset clears the phase."""
        from omninav.locomotion.ik_controller import IKController
        
        cfg = OmegaConf.create({
            "type": "ik_controller",
            "gait_frequency": 2.0,
            "dt": 0.01,
            "foot_links": ["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
        })
        
        controller = IKController(cfg, mock_robot)
        controller._phase = 0.5
        controller.reset()
        
        assert controller._phase == 0.0
    
    def test_compute_foot_targets_shape(self, mock_robot):
        """Test foot targets have correct shape."""
        from omninav.locomotion.ik_controller import IKController
        
        cfg = OmegaConf.create({
            "type": "ik_controller",
            "step_height": 0.05,
            "step_length": 0.15,
            "foot_links": ["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
        })
        
        controller = IKController(cfg, mock_robot)
        
        cmd_vel = np.array([0.5, 0.0, 0.0])
        foot_targets = controller._compute_foot_targets(cmd_vel)
        
        assert foot_targets.shape == (4, 3), "Should have 4 feet with 3D positions"


class TestRLController:
    """Test suite for RLController placeholder."""
    
    def test_compute_action_raises(self, mock_robot):
        """Test that compute_action raises NotImplementedError."""
        from omninav.locomotion.rl_controller import RLController
        
        cfg = OmegaConf.create({"type": "rl_controller"})
        controller = RLController(cfg, mock_robot)
        
        with pytest.raises(NotImplementedError):
            controller.compute_action(np.zeros(3))
    
    def test_step_raises(self, mock_robot):
        """Test that step raises NotImplementedError."""
        from omninav.locomotion.rl_controller import RLController
        
        cfg = OmegaConf.create({"type": "rl_controller"})
        controller = RLController(cfg, mock_robot)
        
        with pytest.raises(NotImplementedError):
            controller.step(np.zeros(3))
