import numpy as np
import pytest
from stable_baselines3 import PPO

from rocketlander.environment.rocket_landing_env import RocketLandingEnv
from rocketlander.evaluation.controllers import make_controller
from rocketlander.ppo.network_controller import (
    OBSERVATION_SCALE,
    NetworkController,
    ScaledObservation,
    scale_observation,
)
from rocketlander.ppo.rocket_network import SB3PolicyAdapter


def test_observations_are_scaled_to_about_one() -> None:
    observation = np.array([10, 50, 5, -10, 0.1, 0.99, 0.5, 100], dtype=np.float32)
    scaled = scale_observation(observation)
    assert scaled.dtype == np.float32
    assert scaled == pytest.approx([1, 1, 1, -1, 0.1, 0.99, 0.5, 1])


def test_wrapper_gives_the_network_scaled_observations() -> None:
    env = RocketLandingEnv()
    wrapped = ScaledObservation(RocketLandingEnv())
    raw, _ = env.reset(seed=3)
    scaled, _ = wrapped.reset(seed=3)
    assert scaled == pytest.approx(raw / OBSERVATION_SCALE)
    assert wrapped.observation_space.shape == env.observation_space.shape


def test_saved_network_flies_as_a_deterministic_controller(tmp_path) -> None:
    ppo = PPO(SB3PolicyAdapter, ScaledObservation(RocketLandingEnv()), n_steps=64, batch_size=64)
    ppo.save(tmp_path / "network")
    controller = NetworkController(tmp_path / "network.zip")

    observation = RocketLandingEnv().reset(seed=0)[0]
    first, second = controller.act(observation), controller.act(observation)

    assert first.shape == (3,)
    assert np.all((first >= -1) & (first <= 1))
    assert np.array_equal(first, second)  # no exploration noise


def test_network_controller_needs_a_network_file() -> None:
    with pytest.raises(ValueError, match="needs a trained network file"):
        make_controller("network", RocketLandingEnv())


def test_unknown_controller_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown controller 'lqr'"):
        make_controller("lqr", RocketLandingEnv())
