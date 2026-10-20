"""What a controller gives the environment: the action, and the throttles it means."""

import numpy as np


def action_to_throttles(action: np.ndarray) -> np.ndarray:
    """An action, one value in [-1, 1] per nozzle, as throttles in [0, 1]."""
    return (np.clip(action, -1.0, 1.0) + 1.0) / 2.0


def throttles_to_action(throttles: np.ndarray) -> np.ndarray:
    """Throttles in [0, 1], one per nozzle, as an action in [-1, 1]."""
    return (2.0 * throttles - 1.0).astype(np.float32)
