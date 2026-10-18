"""The controllers that can fly the rocket, by name, for the evaluation and watch scripts."""

from pathlib import Path

from rocketlander.environment.rocket_landing_env import RocketLandingEnv
from rocketlander.evaluation.evaluation import Controller
from rocketlander.pd.pd_controller import PDController

CONTROLLERS = ("pd", "network")


def make_controller(
    name: str, env: RocketLandingEnv, network_path: str | Path | None = None
) -> Controller:
    """``"pd"``: the hand-written autopilot. ``"network"``: a trained network from a file."""
    if name == "pd":
        return PDController.for_env(env).act
    if name == "network":
        if network_path is None:
            raise ValueError("the network controller needs a trained network file (--network)")
        # Imported here, so torch only loads when a network is used.
        from rocketlander.ppo.network_controller import NetworkController

        return NetworkController(network_path).act
    raise ValueError(f"unknown controller {name!r}; available: {list(CONTROLLERS)}")
