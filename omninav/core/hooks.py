"""
Event/Hook System for OmniNav

Provides a publish-subscribe event system for lifecycle and runtime events.
Replaces implicit ordering dependencies with explicit, priority-ordered hooks.

Usage:
    hooks = HookManager()
    hooks.register(EventType.POST_STEP, my_callback, priority=10)
    hooks.emit(EventType.POST_STEP, obs=current_obs, step=42)
"""

from __future__ import annotations

import logging
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Callable, Any

logger = logging.getLogger(__name__)


# =============================================================================
# Event Types
# =============================================================================

class EventType(Enum):
    """
    Lifecycle and runtime events.

    Lifecycle events (emitted by SimulationRuntime):
        PRE_BUILD   - Before scene.build(), sensors added
        POST_BUILD  - After scene.build(), physics ready
        PRE_STEP    - Before each simulation step
        POST_STEP   - After each simulation step
        ON_RESET    - When environment is reset

    Runtime events (emitted by Task/Algorithm):
        ON_COLLISION          - Collision detected (safety critical)
        ON_WAYPOINT_REACHED   - Inspection waypoint reached
        ON_ANOMALY_DETECTED   - Anomaly detected during inspection
        ON_TASK_COMPLETE      - Task episode finished
    """

    # Lifecycle
    PRE_BUILD = auto()
    POST_BUILD = auto()
    PRE_STEP = auto()
    POST_STEP = auto()
    ON_RESET = auto()

    # Runtime
    ON_COLLISION = auto()
    ON_WAYPOINT_REACHED = auto()
    ON_ANOMALY_DETECTED = auto()
    ON_TASK_COMPLETE = auto()


# =============================================================================
# Hook
# =============================================================================

@dataclass
class Hook:
    """A registered callback with priority ordering."""

    callback: Callable[..., None]
    """The callback function. Receives **context from emit()."""

    priority: int = 0
    """Execution priority. Lower values execute first."""

    name: str = ""
    """Optional human-readable name for debugging."""

    def __post_init__(self):
        if not self.name:
            self.name = getattr(self.callback, "__qualname__", str(self.callback))


# =============================================================================
# Hook Manager
# =============================================================================

class HookManager:
    """
    Manages registration and emission of lifecycle/runtime events.

    Hooks are executed in priority order (lower = earlier).
    Multiple hooks on the same event are supported.

    Example:
        >>> hooks = HookManager()
        >>> hooks.register(EventType.POST_STEP, log_data, priority=100)
        >>> hooks.register(EventType.POST_STEP, save_frame, priority=50)
        >>> # save_frame runs before log_data (lower priority = earlier)
        >>> hooks.emit(EventType.POST_STEP, obs=obs, step_count=42)
    """

    def __init__(self):
        self._hooks: dict[EventType, list[Hook]] = {event: [] for event in EventType}
        self._sorted: dict[EventType, bool] = {event: True for event in EventType}

    def register(
        self,
        event: EventType,
        callback: Callable[..., None],
        priority: int = 0,
        name: str = "",
    ) -> None:
        """
        Register a callback for an event.

        Args:
            event: Event type to listen for
            callback: Function to call when event is emitted.
                      Receives keyword arguments from emit().
            priority: Execution order (lower = earlier). Default: 0
            name: Optional human-readable name for debugging
        """
        hook = Hook(callback=callback, priority=priority, name=name)
        self._hooks[event].append(hook)
        self._sorted[event] = False
        logger.debug(f"Registered hook '{hook.name}' for {event.name} (priority={priority})")

    def unregister(self, event: EventType, callback: Callable[..., None]) -> bool:
        """
        Remove a callback from an event.

        Args:
            event: Event type
            callback: The callback to remove

        Returns:
            True if callback was found and removed, False otherwise
        """
        hooks = self._hooks[event]
        for i, hook in enumerate(hooks):
            if hook.callback is callback:
                hooks.pop(i)
                logger.debug(f"Unregistered hook '{hook.name}' from {event.name}")
                return True
        return False

    def emit(self, event: EventType, **context: Any) -> None:
        """
        Emit an event, triggering all registered callbacks in priority order.

        Args:
            event: Event type to emit
            **context: Keyword arguments passed to each callback
        """
        hooks = self._hooks[event]
        if not hooks:
            return

        # Sort by priority if needed (lazy sorting)
        if not self._sorted[event]:
            hooks.sort(key=lambda h: h.priority)
            self._sorted[event] = True

        for hook in hooks:
            try:
                hook.callback(**context)
            except Exception as e:
                logger.error(
                    f"Error in hook '{hook.name}' for {event.name}: {e}",
                    exc_info=True,
                )

    def clear(self, event: EventType = None) -> None:
        """
        Clear registered hooks.

        Args:
            event: If specified, clear only this event's hooks.
                   If None, clear all hooks.
        """
        if event is not None:
            self._hooks[event].clear()
            self._sorted[event] = True
        else:
            for ev in EventType:
                self._hooks[ev].clear()
                self._sorted[ev] = True

    def count(self, event: EventType = None) -> int:
        """
        Count registered hooks.

        Args:
            event: If specified, count only this event. Otherwise count all.

        Returns:
            Number of registered hooks
        """
        if event is not None:
            return len(self._hooks[event])
        return sum(len(hooks) for hooks in self._hooks.values())

    def __repr__(self) -> str:
        counts = {e.name: len(h) for e, h in self._hooks.items() if h}
        return f"HookManager({counts})"
