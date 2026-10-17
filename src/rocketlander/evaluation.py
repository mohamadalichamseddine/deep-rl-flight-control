"""Run a controller on a fixed test set of starting conditions and measure how it lands."""

import statistics
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rocketlander.rocket_landing_env import RocketLandingEnv

# Test seeds start far from the small seeds used in training, so the two never overlap.
TEST_SEED_OFFSET = 1_000_000

Controller = Callable[[np.ndarray], np.ndarray]  # observation -> action in [-1, 1]


def seed_for_test_episode(episode: int) -> int:
    """The seed of test episode number ``episode`` (0, 1, 2, ...)."""
    return TEST_SEED_OFFSET + episode


@dataclass
class EpisodeResult:
    seed: int
    outcome: str
    failed_checks: list[str]
    touchdown: dict[str, float] | None  # x, vx, vy, tilt_deg, spin_deg_s just before contact
    duration_s: float
    fuel_s: float  # seconds of full main-engine thrust used (side nozzles count by thrust)
    trajectory: list[tuple[float, float]] = field(repr=False)  # center of mass (x, y)


def run_episode(env: RocketLandingEnv, controller: Controller, seed: int) -> EpisodeResult:
    """Fly one episode from the start given by ``seed``, with the controller deciding."""
    observation, _ = env.reset(seed=seed)
    trajectory = [(float(observation[0]), float(observation[1]))]
    thrusts = np.array([n.max_thrust for n in env.rocket.nozzles])
    fuel_s, steps, done = 0.0, 0, False
    while not done:
        action = controller(observation)
        throttles = (np.clip(action, -1.0, 1.0) + 1.0) / 2.0
        fuel_s += float(throttles @ thrusts) / thrusts.max() * env.config.dt
        observation, _, terminated, truncated, info = env.step(action)
        trajectory.append((float(observation[0]), float(observation[1])))
        steps += 1
        done = terminated or truncated
    return EpisodeResult(
        seed=seed,
        outcome=info["outcome"],
        failed_checks=info.get("failed_checks", []),
        touchdown=info.get("touchdown"),
        duration_s=steps * env.config.dt,
        fuel_s=fuel_s,
        trajectory=trajectory,
    )


def evaluate(
    controller: Controller, env: RocketLandingEnv, episodes: int = 500
) -> list[EpisodeResult]:
    """Fly the level's test set: test episodes 0 to ``episodes - 1``."""
    return [run_episode(env, controller, seed_for_test_episode(i)) for i in range(episodes)]


def summarize(results: list[EpisodeResult]) -> dict[str, Any]:
    """The evaluation metrics: landing rate, why the others failed, and touchdown quality."""
    n = len(results)
    touchdowns = [r.touchdown for r in results if r.touchdown is not None]
    failed = Counter(
        check.rsplit(" ", 4)[0] for r in results for check in r.failed_checks
    )  # e.g. "vertical speed 8.00 > 1 m/s" -> "vertical speed"

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
        "landing_rate": sum(r.outcome == "landed" for r in results) / n,
        "outcomes": dict(Counter(r.outcome for r in results)),
        "failed_checks": dict(failed),
        "touchdown": {
            "x error [m]": spread([abs(t["x"]) for t in touchdowns]),
            "horizontal speed [m/s]": spread([abs(t["vx"]) for t in touchdowns]),
            "vertical speed [m/s]": spread([abs(t["vy"]) for t in touchdowns]),
            "tilt [deg]": spread([abs(t["tilt_deg"]) for t in touchdowns]),
            "spin [deg/s]": spread([abs(t["spin_deg_s"]) for t in touchdowns]),
        },
        "duration [s]": spread([r.duration_s for r in results]),
        "fuel [s of full thrust]": spread([r.fuel_s for r in results]),
    }


def can_stop_before_the_ground(env: RocketLandingEnv, observation: np.ndarray) -> bool:
    """Whether full main thrust could stop the fall before the legs reach the ground.

    A start that fails this is impossible for any controller, so it doesn't belong in a
    test set.
    """
    y, vy = float(observation[1]), float(observation[3])
    if vy >= 0:
        return True
    gravity = env.sim.gravity
    braking = env.rocket.nozzle("main").max_thrust / env.sim.mass - gravity  # [m/s²]
    return braking > 0 and vy**2 / (2 * braking) < y - env.standing_height


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
