"""The controllers that can fly the rocket, by name, for the evaluation and watch scripts."""

from pathlib import Path

from rocketlander.evaluation import Controller
from rocketlander.pd_controller import PDController
from rocketlander.rocket_landing_env import RocketLandingEnv

CONTROLLERS = ("pd", "ppo")


def make_controller(
    name: str, env: RocketLandingEnv, model_path: str | Path | None = None
) -> Controller:
    """``"pd"``: the hand-written autopilot. ``"ppo"``: a trained network from ``model_path``."""
    if name == "pd":
        return PDController.for_env(env).act
    if name == "ppo":
        if model_path is None:
            raise ValueError("the ppo controller needs a trained network file (--model)")
        from rocketlander.ppo_controller import PPOController  # loads torch only when needed

        return PPOController(model_path).act
    raise ValueError(f"unknown controller {name!r}; available: {list(CONTROLLERS)}")
