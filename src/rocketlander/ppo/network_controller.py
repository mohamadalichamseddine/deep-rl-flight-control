"""A network trained with PPO as a controller, and the input scaling it is trained with."""

from pathlib import Path

import gymnasium as gym
import numpy as np

# The environment observes in physical units: [x, y, vx, vy, sin(tilt), cos(tilt), spin,
# mass]. Neural networks learn best from inputs of about -1 to 1, so each is divided by a
# typical size first. The scale is fixed (not learned), so nothing extra needs saving.
OBSERVATION_SCALE = np.array([10.0, 50.0, 5.0, 10.0, 1.0, 1.0, 1.0, 100.0], dtype=np.float32)


def scale_observation(observation: np.ndarray) -> np.ndarray:
    return (np.asarray(observation, dtype=np.float32) / OBSERVATION_SCALE).astype(np.float32)


class ScaledObservation(gym.ObservationWrapper):
    """Gives the network scaled observations during training."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=env.observation_space.shape, dtype=np.float32
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return scale_observation(observation)


class NetworkController:
    """A trained network flying the rocket: physical observation in, action out.

    Uses the network's best action (``deterministic=True``), without the random exploration
    noise of training, so the same start always gives the same flight.
    """

    def __init__(self, network_path: str | Path) -> None:
        from stable_baselines3 import PPO  # imported here: loading torch takes a moment

        # The file holds the whole PPO trainer; we only use its network, through predict().
        self.ppo = PPO.load(network_path, device="cpu")

    def act(self, observation: np.ndarray) -> np.ndarray:
        action, _ = self.ppo.predict(scale_observation(observation), deterministic=True)
        return action
