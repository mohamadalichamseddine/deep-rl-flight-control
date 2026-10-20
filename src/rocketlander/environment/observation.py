"""What the environment shows a controller: the observation, and the state it describes."""

import math

import numpy as np

from rocketlander.rocket.rocket_state import RocketState


def state_to_observation(state: RocketState, mass: float) -> np.ndarray:
    """The observation ``[x, y, vx, vy, sin(tilt), cos(tilt), spin, mass]``.

    sin/cos avoids a jump when the tilt wraps around.
    """
    return np.array(
        [
            state.x,
            state.y,
            state.vx,
            state.vy,
            math.sin(state.tilt),
            math.cos(state.tilt),
            state.spin,
            mass,
        ],
        dtype=np.float32,
    )


def observation_to_state(observation: np.ndarray) -> RocketState:
    """The rocket state an observation describes (the mass is left out)."""
    x, y, vx, vy, sin_tilt, cos_tilt, spin, _ = (float(value) for value in observation)
    return RocketState(x=x, y=y, vx=vx, vy=vy, tilt=math.atan2(sin_tilt, cos_tilt), spin=spin)
