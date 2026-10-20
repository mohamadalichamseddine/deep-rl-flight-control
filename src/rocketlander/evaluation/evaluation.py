"""Run a controller on a fixed test set of starting conditions and measure how it lands."""

import math
import statistics
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rocketlander.environment.landing_judge import LandingCheck, parts_other_than_legs
from rocketlander.environment.observation import observation_to_state
from rocketlander.environment.outcome import Outcome
from rocketlander.environment.rocket_landing_env import RocketLandingEnv
from rocketlander.rocket.rocket_simulator import RocketSimulator, Touchdown
from rocketlander.rocket.rocket_state import RocketState

# Test seeds start far from the small seeds used in training, so the two never overlap.
FIRST_TEST_SEED = 1_000_000

Controller = Callable[[np.ndarray], np.ndarray]  # observation -> action in [-1, 1]


def seed_for_test_episode(episode: int) -> int:
    """The seed of test episode number ``episode`` (0, 1, 2, ...)."""
    return FIRST_TEST_SEED + episode


@dataclass
class EpisodeResult:
    seed: int
    outcome: Outcome
    failed_checks: list[LandingCheck]
    touchdown: Touchdown | None  # the state just before contact, and the parts that touched
    duration_s: float
    total_reward: float  # the rewards of every step, summed
    fuel_used: float  # seconds of full main-engine thrust used (side nozzles count by thrust)
    trajectory: list[tuple[float, float]] = field(repr=False)  # center of mass (x, y)


def run_episode(env: RocketLandingEnv, controller: Controller, seed: int) -> EpisodeResult:
    """Fly one episode from the start given by ``seed``, with the controller deciding."""
    observation, _ = env.reset(seed=seed)
    state = observation_to_state(observation)
    trajectory = [(state.x, state.y)]
    total_reward, fuel_used, steps, done = 0.0, 0.0, 0, False
    while not done:
        observation, reward, terminated, truncated, info = env.step(controller(observation))
        state = observation_to_state(observation)
        trajectory.append((state.x, state.y))
        total_reward += reward
        fuel_used += info["fuel_used"]
        steps += 1
        done = terminated or truncated
    return EpisodeResult(
        seed=seed,
        outcome=info["outcome"],
        failed_checks=info["failed_checks"],
        touchdown=info["touchdown"],
        duration_s=steps * env.env_config.dt,
        total_reward=total_reward,
        fuel_used=fuel_used,
        trajectory=trajectory,
    )


def run_test_set(
    controller: Controller, env: RocketLandingEnv, episodes: int
) -> list[EpisodeResult]:
    """Fly the level's test set: test episodes 0 to ``episodes - 1``."""
    return [run_episode(env, controller, seed_for_test_episode(i)) for i in range(episodes)]


def summarize(results: list[EpisodeResult]) -> dict[str, Any]:
    """The evaluation metrics: landing rate, why the others failed, and touchdown quality."""
    n = len(results)
    touchdowns = [r.touchdown for r in results if r.touchdown is not None]
    failed = Counter(check.name for r in results for check in r.failed_checks)
    failed.update(
        f"{part} touched the ground"
        for touchdown in touchdowns
        for part in parts_other_than_legs(touchdown)
    )
    contacts = [touchdown.state for touchdown in touchdowns]

    def spread(values: list[float]) -> dict[str, float]:
        if not values:
            return {}
        return {
            "mean": statistics.fmean(values),
            "p95": float(np.percentile(values, 95)),
            "max": max(values),
        }

    return {
        "episodes": n,
        "landing_rate": sum(r.outcome == Outcome.LANDED for r in results) / n,
        "outcomes": dict(Counter(r.outcome for r in results)),
        "failed_checks": dict(failed),
        "touchdown": {
            "off center [m]": spread([abs(s.x) for s in contacts]),
            "horizontal speed [m/s]": spread([abs(s.vx) for s in contacts]),
            "vertical speed [m/s]": spread([abs(s.vy) for s in contacts]),
            "tilt [deg]": spread([math.degrees(abs(s.tilt)) for s in contacts]),
            "spin [deg/s]": spread([math.degrees(abs(s.spin)) for s in contacts]),
        },
        "duration [s]": spread([r.duration_s for r in results]),
        "fuel [s of full thrust]": spread([r.fuel_used for r in results]),
    }


def can_stop_before_the_ground(rocket_simulator: RocketSimulator, state: RocketState) -> bool:
    """Whether full main thrust could stop the fall before the legs reach the ground.

    A start that fails this is impossible for any controller, so it doesn't belong in a
    test set.
    """
    if state.vy >= 0:
        return True
    main_thrust = rocket_simulator.rocket_spec.get_nozzle("main").max_thrust
    braking = main_thrust / rocket_simulator.mass - rocket_simulator.gravity  # [m/s²]
    if braking <= 0:
        return False
    stopping_distance = state.vy**2 / (2 * braking)
    return stopping_distance < state.y - rocket_simulator.y_when_standing


def format_summary(summary: dict[str, Any]) -> str:
    """The summary as a readable text table."""
    lines = [
        f"landed {summary['landing_rate']:.1%} of {summary['episodes']} episodes",
        "outcomes: " + ", ".join(f"{k} {v}" for k, v in sorted(summary["outcomes"].items())),
    ]
    if summary["failed_checks"]:
        checks = ", ".join(f"{k} {v}" for k, v in sorted(summary["failed_checks"].items()))
        lines.append(f"failed checks: {checks}")
    lines.append(f"\n{'':<34}{'mean':>8}{'p95':>8}{'max':>8}")
    rows = {f"touchdown {k}": v for k, v in summary["touchdown"].items()}
    rows["duration [s]"] = summary["duration [s]"]
    rows["fuel [s of full thrust]"] = summary["fuel [s of full thrust]"]
    for name, values in rows.items():
        if values:
            cells = "".join(f"{values[key]:8.2f}" for key in ("mean", "p95", "max"))
            lines.append(f"{name:<34}{cells}")
    return "\n".join(lines)
