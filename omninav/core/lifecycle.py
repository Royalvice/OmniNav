"""
Component Lifecycle State Machine for OmniNav

Provides a mixin for components that follow a strict initialization lifecycle.
Prevents out-of-order method calls with clear error messages.

Lifecycle order:
    CREATED → spawn() → SPAWNED → mount_sensors() → SENSORS_MOUNTED
    → [scene.build()] → post_build() → BUILT → reset() → READY

Usage:
    class MyRobot(LifecycleMixin, RobotBase):
        def spawn(self, scene):
            self._require_state(LifecycleState.CREATED, "spawn")
            # ... do work ...
            self._transition_to(LifecycleState.SPAWNED)
"""

from __future__ import annotations

import logging
from enum import IntEnum, auto

logger = logging.getLogger(__name__)


# =============================================================================
# Lifecycle States
# =============================================================================

class LifecycleState(IntEnum):
    """
    Ordered lifecycle states for components.

    Uses IntEnum so states can be compared with < > operators:
        LifecycleState.CREATED < LifecycleState.READY  →  True
    """

    CREATED = auto()
    """Component initialized via __init__(), not yet in scene."""

    SPAWNED = auto()
    """Component added to scene (spawn() called)."""

    SENSORS_MOUNTED = auto()
    """Sensors attached to component (mount_sensors() called)."""

    BUILT = auto()
    """Scene built, physics ready (post_build() called)."""

    READY = auto()
    """Fully initialized, ready for stepping (reset() called)."""


# =============================================================================
# Lifecycle Mixin
# =============================================================================

class LifecycleMixin:
    """
    Mixin that adds lifecycle state tracking to components.

    Provides:
    - _state: current lifecycle state
    - _require_state(): guard that raises if state is too low
    - _transition_to(): advance to a new state with validation

    Designed to be used with multiple inheritance:
        class MyRobot(LifecycleMixin, RobotBase):
            ...
    """

    _state: LifecycleState = LifecycleState.CREATED

    @property
    def lifecycle_state(self) -> LifecycleState:
        """Get current lifecycle state."""
        return self._state

    def _require_state(self, min_state: LifecycleState, action: str = "") -> None:
        """
        Assert that the component is at least in the given state.

        Args:
            min_state: Minimum required state
            action: Name of the action being attempted (for error messages)

        Raises:
            RuntimeError: If current state is below min_state
        """
        if self._state < min_state:
            action_str = f" '{action}'" if action else ""
            raise RuntimeError(
                f"Cannot perform{action_str}: component '{type(self).__name__}' "
                f"is in state {self._state.name}, but {min_state.name} is required.\n"
                f"Lifecycle: CREATED → spawn() → SPAWNED → mount_sensors() → "
                f"SENSORS_MOUNTED → [build] → post_build() → BUILT → reset() → READY"
            )

    def _transition_to(self, new_state: LifecycleState) -> None:
        """
        Transition to a new lifecycle state.

        Validates that the transition is forward (no going back to earlier states
        except from READY back to READY on re-reset).

        Args:
            new_state: Target state

        Raises:
            RuntimeError: If transition is invalid
        """
        # Allow READY → READY (re-reset)
        if new_state == LifecycleState.READY and self._state == LifecycleState.READY:
            logger.debug(f"{type(self).__name__}: re-entered READY state")
            return

        # Allow BUILT → READY and READY → READY (re-reset after build)
        if new_state <= self._state and not (
            new_state == LifecycleState.READY and self._state >= LifecycleState.BUILT
        ):
            raise RuntimeError(
                f"Invalid lifecycle transition for '{type(self).__name__}': "
                f"{self._state.name} → {new_state.name}. "
                f"Transitions must go forward."
            )

        old_state = self._state
        self._state = new_state
        logger.debug(
            f"{type(self).__name__}: {old_state.name} → {new_state.name}"
        )
