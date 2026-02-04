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


class TestKinematicController:
    """Test suite for KinematicController."""
    
    def test_phase_update_walking(self, mock_robot):
        """Test that phase updates when walking."""
        from omninav.locomotion.kinematic_controller import KinematicController
        
        cfg = OmegaConf.create({
            "type": "kinematic_gait",
            "gait_frequency": 2.0,
            "dt": 0.01,
            "foot_links": ["FL_calf", "FR_calf", "RL_calf", "RR_calf"],
            "use_terrain_sensing": False,
        })
        
        controller = KinematicController(cfg, mock_robot)
        
        initial_phase = controller._phase
        # Walking command
        controller._update_phase(np.array([0.5, 0.0, 0.0]))
        
        # Phase should have increased
        assert controller._phase > initial_phase
    
    def test_phase_stops_at_rest(self, mock_robot):
        """Test that phase settles to 0 or 0.5 when stopped."""
        from omninav.locomotion.kinematic_controller import KinematicController
        
        cfg = OmegaConf.create({
            "type": "kinematic_gait",
            "gait_frequency": 2.0,
            "dt": 0.01,
            "foot_links": ["FL_calf", "FR_calf", "RL_calf", "RR_calf"],
            "use_terrain_sensing": False,
        })
        
        controller = KinematicController(cfg, mock_robot)
        controller._phase = 0.1  # Near 0
        
        # Stop command, run multiple updates
        for _ in range(100):
            controller._update_phase(np.array([0.0, 0.0, 0.0]))
        
        # Phase should settle to 0 or 0.5
        assert controller._phase == 0.0 or controller._phase == 0.5
    
    def test_reset_clears_state(self, mock_robot):
        """Test that reset clears the phase and state."""
        from omninav.locomotion.kinematic_controller import KinematicController
        
        cfg = OmegaConf.create({
            "type": "kinematic_gait",
            "gait_frequency": 2.0,
            "dt": 0.01,
            "foot_links": ["FL_calf", "FR_calf", "RL_calf", "RR_calf"],
            "use_terrain_sensing": False,
        })
        
        controller = KinematicController(cfg, mock_robot)
        controller._phase = 0.7
        controller._state = 1
        controller.reset()
        
        assert controller._phase == 0.0
        assert controller._state == controller.STATE_STAND
    
    def test_leg_phase_trot(self, mock_robot):
        """Test trot gait: FL+RR same phase, FR+RL opposite."""
        from omninav.locomotion.kinematic_controller import KinematicController
        
        cfg = OmegaConf.create({
            "type": "kinematic_gait",
            "foot_links": ["FL_calf", "FR_calf", "RL_calf", "RR_calf"],
            "use_terrain_sensing": False,
        })
        
        controller = KinematicController(cfg, mock_robot)
        controller._phase = 0.3
        
        # FL and RR should have same phase
        assert controller._get_leg_phase(0) == controller._get_leg_phase(3)
        # FR and RL should have same phase
        assert controller._get_leg_phase(1) == controller._get_leg_phase(2)
        # The two groups should be 0.5 apart
        assert abs(controller._get_leg_phase(0) - controller._get_leg_phase(1)) == pytest.approx(0.5, abs=0.01)


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
