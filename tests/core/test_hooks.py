"""
Tests for Event/Hook system.
"""

import pytest

from omninav.core.hooks import EventType, Hook, HookManager


class TestEventType:
    """Tests for EventType enum."""

    def test_lifecycle_events_exist(self):
        """Test that all lifecycle events are defined."""
        assert EventType.PRE_BUILD
        assert EventType.POST_BUILD
        assert EventType.PRE_STEP
        assert EventType.POST_STEP
        assert EventType.ON_RESET

    def test_runtime_events_exist(self):
        """Test that inspection-related events are defined."""
        assert EventType.ON_COLLISION
        assert EventType.ON_WAYPOINT_REACHED
        assert EventType.ON_ANOMALY_DETECTED
        assert EventType.ON_TASK_COMPLETE


class TestHook:
    """Tests for Hook dataclass."""

    def test_hook_auto_name(self):
        """Test that hook auto-generates name from callback."""
        def my_callback(**kwargs):
            pass
        hook = Hook(callback=my_callback)
        assert "my_callback" in hook.name

    def test_hook_explicit_name(self):
        """Test hook with explicit name."""
        hook = Hook(callback=lambda **kwargs: None, name="logger")
        assert hook.name == "logger"


class TestHookManager:
    """Tests for HookManager."""

    def test_register_and_emit(self):
        """Test basic register + emit flow."""
        manager = HookManager()
        results = []

        def on_step(**kwargs):
            results.append(kwargs.get("step", -1))

        manager.register(EventType.POST_STEP, on_step)
        manager.emit(EventType.POST_STEP, step=42)

        assert results == [42]

    def test_multiple_hooks_same_event(self):
        """Test multiple hooks fired for same event."""
        manager = HookManager()
        results = []

        manager.register(EventType.POST_STEP, lambda **kw: results.append("a"))
        manager.register(EventType.POST_STEP, lambda **kw: results.append("b"))

        manager.emit(EventType.POST_STEP)
        assert len(results) == 2

    def test_priority_ordering(self):
        """Test that lower priority executes first."""
        manager = HookManager()
        results = []

        manager.register(EventType.POST_STEP, lambda **kw: results.append("high"), priority=100)
        manager.register(EventType.POST_STEP, lambda **kw: results.append("low"), priority=1)
        manager.register(EventType.POST_STEP, lambda **kw: results.append("mid"), priority=50)

        manager.emit(EventType.POST_STEP)
        assert results == ["low", "mid", "high"]

    def test_emit_no_hooks(self):
        """Test emit with no registered hooks doesn't error."""
        manager = HookManager()
        manager.emit(EventType.PRE_BUILD)  # Should not raise

    def test_unregister(self):
        """Test removing a hook."""
        manager = HookManager()
        results = []

        def my_hook(**kwargs):
            results.append(1)

        manager.register(EventType.POST_STEP, my_hook)
        assert manager.unregister(EventType.POST_STEP, my_hook) is True

        manager.emit(EventType.POST_STEP)
        assert results == []

    def test_unregister_nonexistent(self):
        """Test unregistering a non-registered callback returns False."""
        manager = HookManager()
        result = manager.unregister(EventType.POST_STEP, lambda **kw: None)
        assert result is False

    def test_error_isolation(self):
        """Test that one hook error doesn't block others."""
        manager = HookManager()
        results = []

        manager.register(EventType.POST_STEP, lambda **kw: results.append("first"), priority=1)
        manager.register(
            EventType.POST_STEP,
            lambda **kw: (_ for _ in ()).throw(ValueError("boom")),
            priority=2,
        )
        manager.register(EventType.POST_STEP, lambda **kw: results.append("third"), priority=3)

        manager.emit(EventType.POST_STEP)
        # First should run, second errors, third should still run
        assert "first" in results
        assert "third" in results

    def test_context_passing(self):
        """Test that context kwargs are passed to hooks."""
        manager = HookManager()
        received = {}

        def capture(**kwargs):
            received.update(kwargs)

        manager.register(EventType.ON_WAYPOINT_REACHED, capture)
        manager.emit(EventType.ON_WAYPOINT_REACHED, waypoint_id=5, position=[1.0, 2.0])

        assert received["waypoint_id"] == 5
        assert received["position"] == [1.0, 2.0]

    def test_clear_specific_event(self):
        """Test clearing hooks for one event only."""
        manager = HookManager()
        manager.register(EventType.POST_STEP, lambda **kw: None)
        manager.register(EventType.PRE_STEP, lambda **kw: None)

        manager.clear(EventType.POST_STEP)
        assert manager.count(EventType.POST_STEP) == 0
        assert manager.count(EventType.PRE_STEP) == 1

    def test_clear_all(self):
        """Test clearing all hooks."""
        manager = HookManager()
        manager.register(EventType.POST_STEP, lambda **kw: None)
        manager.register(EventType.PRE_STEP, lambda **kw: None)

        manager.clear()
        assert manager.count() == 0

    def test_count(self):
        """Test hook counting."""
        manager = HookManager()
        assert manager.count() == 0

        manager.register(EventType.POST_STEP, lambda **kw: None)
        manager.register(EventType.POST_STEP, lambda **kw: None)
        manager.register(EventType.PRE_STEP, lambda **kw: None)

        assert manager.count(EventType.POST_STEP) == 2
        assert manager.count(EventType.PRE_STEP) == 1
        assert manager.count() == 3

    def test_repr(self):
        """Test string representation."""
        manager = HookManager()
        manager.register(EventType.POST_STEP, lambda **kw: None)
        repr_str = repr(manager)
        assert "POST_STEP" in repr_str

    def test_different_events_independent(self):
        """Test that different events don't interfere."""
        manager = HookManager()
        step_results = []
        reset_results = []

        manager.register(EventType.POST_STEP, lambda **kw: step_results.append(1))
        manager.register(EventType.ON_RESET, lambda **kw: reset_results.append(1))

        manager.emit(EventType.POST_STEP)
        assert len(step_results) == 1
        assert len(reset_results) == 0
