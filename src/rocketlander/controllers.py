"""The controllers that can fly the rocket, by name, for the evaluation and watch scripts."""

from collections.abc import Callable

from rocketlander.evaluation import Controller
from rocketlander.pd_controller import PDController
from rocketlander.rocket_landing_env import RocketLandingEnv

CONTROLLERS: dict[str, Callable[[RocketLandingEnv], Controller]] = {
    "pd": lambda env: PDController.for_env(env).act,
}


def make_controller(name: str, env: RocketLandingEnv) -> Controller:
    try:
        return CONTROLLERS[name](env)
    except KeyError:
        raise ValueError(f"unknown controller {name!r}; available: {sorted(CONTROLLERS)}") from None
