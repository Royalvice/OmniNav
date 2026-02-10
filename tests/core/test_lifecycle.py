"""
Tests for Component Lifecycle State Machine.
"""

import pytest

from omninav.core.lifecycle import LifecycleState, LifecycleMixin


class DummyComponent(LifecycleMixin):
    """Test component using the lifecycle mixin."""

    def __init__(self):
        self._state = LifecycleState.CREATED

    def spawn(self):
        self._require_state(LifecycleState.CREATED, "spawn")
        self._transition_to(LifecycleState.SPAWNED)

    def mount_sensors(self):
        self._require_state(LifecycleState.SPAWNED, "mount_sensors")
        self._transition_to(LifecycleState.SENSORS_MOUNTED)

    def post_build(self):
        self._require_state(LifecycleState.SENSORS_MOUNTED, "post_build")
        self._transition_to(LifecycleState.BUILT)

    def reset(self):
        self._require_state(LifecycleState.BUILT, "reset")
        self._transition_to(LifecycleState.READY)


class TestLifecycleState:
    """Tests for LifecycleState ordering."""

    def test_state_ordering(self):
        """Test that states are in correct order."""
        assert LifecycleState.CREATED < LifecycleState.SPAWNED
        assert LifecycleState.SPAWNED < LifecycleState.SENSORS_MOUNTED
        assert LifecycleState.SENSORS_MOUNTED < LifecycleState.BUILT
        assert LifecycleState.BUILT < LifecycleState.READY

    def test_state_comparison(self):
        """Test state comparison operators."""
        assert LifecycleState.CREATED < LifecycleState.READY
        assert LifecycleState.READY > LifecycleState.CREATED
        assert LifecycleState.SPAWNED >= LifecycleState.SPAWNED


class TestLifecycleMixin:
    """Tests for LifecycleMixin."""

    def test_initial_state(self):
        """Test component starts in CREATED state."""
        comp = DummyComponent()
        assert comp.lifecycle_state == LifecycleState.CREATED

    def test_happy_path(self):
        """Test normal lifecycle progression."""
        comp = DummyComponent()
        comp.spawn()
        assert comp.lifecycle_state == LifecycleState.SPAWNED

        comp.mount_sensors()
        assert comp.lifecycle_state == LifecycleState.SENSORS_MOUNTED

        comp.post_build()
        assert comp.lifecycle_state == LifecycleState.BUILT

        comp.reset()
        assert comp.lifecycle_state == LifecycleState.READY

    def test_require_state_too_early(self):
        """Test that calling methods too early raises."""
        comp = DummyComponent()
        with pytest.raises(RuntimeError, match="SPAWNED is required"):
            comp.mount_sensors()

    def test_require_state_skipping(self):
        """Test that skipping states raises."""
        comp = DummyComponent()
        with pytest.raises(RuntimeError, match="BUILT is required"):
            comp.reset()

    def test_re_reset_allowed(self):
        """Test that re-reset from READY is allowed."""
        comp = DummyComponent()
        comp.spawn()
        comp.mount_sensors()
        comp.post_build()
        comp.reset()
        assert comp.lifecycle_state == LifecycleState.READY

        # Re-reset should be fine
        comp.reset()
        assert comp.lifecycle_state == LifecycleState.READY

    def test_backward_transition_blocked(self):
        """Test that going backward raises."""
        comp = DummyComponent()
        comp.spawn()

        with pytest.raises(RuntimeError, match="must go forward"):
            comp._transition_to(LifecycleState.CREATED)

    def test_error_message_includes_class_name(self):
        """Test that error messages include the component name."""
        comp = DummyComponent()
        with pytest.raises(RuntimeError, match="DummyComponent"):
            comp.mount_sensors()

    def test_error_message_includes_lifecycle_hint(self):
        """Test that error messages include lifecycle order hint."""
        comp = DummyComponent()
        with pytest.raises(RuntimeError, match="Lifecycle:"):
            comp.reset()
