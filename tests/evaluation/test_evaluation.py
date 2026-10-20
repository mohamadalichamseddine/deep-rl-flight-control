import math

import numpy as np
import pytest

from rocketlander.environment.env_config import EnvConfig
from rocketlander.environment.landing_judge import LandingCheck
from rocketlander.environment.outcome import Outcome
from rocketlander.environment.rocket_landing_env import RocketLandingEnv
from rocketlander.evaluation.evaluation import (
    EpisodeResult,
    can_stop_before_the_ground,
    format_summary,
    run_episode,
    run_test_set,
    seed_for_test_episode,
    summarize,
)
from rocketlander.evaluation.evaluation_plots import plot_evaluation
from rocketlander.pd.pd_controller import PDGains, build_pd_controller
from rocketlander.rocket.rocket_simulator import Touchdown
from rocketlander.rocket.rocket_spec import RocketSpec
from rocketlander.rocket.rocket_state import RocketState

ALL_OFF = np.array([-1.0, -1.0, -1.0], dtype=np.float32)


@pytest.fixture(scope="module")
def env(rocket_spec: RocketSpec, env_config: EnvConfig) -> RocketLandingEnv:
    return RocketLandingEnv(rocket_spec, env_config, level=0, render_mode=None)


def test_test_seeds_are_far_from_training_seeds() -> None:
    assert seed_for_test_episode(0) == 1_000_000
    assert seed_for_test_episode(137) == 1_000_137


def test_run_episode_records_the_flight(env: RocketLandingEnv) -> None:
    result = run_episode(env, lambda obs: ALL_OFF, seed_for_test_episode(0))

    start = env.reset(seed=seed_for_test_episode(0))[0]
    assert result.trajectory[0] == pytest.approx((start[0], start[1]))
    assert result.outcome == Outcome.UNSAFE_TOUCHDOWN  # free fall
    assert result.fuel_used == 0.0
    assert result.duration_s == pytest.approx(0.02 * (len(result.trajectory) - 1))

    env.reset(seed=seed_for_test_episode(0))
    rewards, done = [], False
    while not done:
        _, reward, terminated, truncated, _ = env.step(ALL_OFF)
        rewards.append(reward)
        done = terminated or truncated
    assert result.total_reward == pytest.approx(sum(rewards))


def test_fuel_counts_seconds_of_full_main_thrust(env: RocketLandingEnv) -> None:
    half_main = np.array([0.0, -1.0, -1.0], dtype=np.float32)  # main at 50%
    result = run_episode(env, lambda obs: half_main, seed_for_test_episode(0))
    assert result.fuel_used == pytest.approx(0.5 * result.duration_s)


def test_evaluation_is_repeatable(env: RocketLandingEnv, pd_gains: PDGains) -> None:
    controller = build_pd_controller(env.rocket_simulator, pd_gains).act
    first = summarize(run_test_set(controller, env, episodes=3))
    second = summarize(run_test_set(controller, env, episodes=3))
    assert first == second


def test_summary_counts_outcomes_and_failed_checks(env: RocketLandingEnv) -> None:
    results = [
        _result(Outcome.LANDED, []),
        _result(
            Outcome.UNSAFE_TOUCHDOWN,
            [
                LandingCheck("vertical speed", 8.0, 1.0, "m/s"),
                LandingCheck("tilt", 6.0, 5.0, "deg"),
            ],
        ),
        _result(Outcome.OUT_OF_BOUNDS, [], touchdown=False),
        _result(Outcome.LANDED, []),
    ]

    summary = summarize(results)

    assert summary["landing_rate"] == 0.5
    assert summary["outcomes"] == {"landed": 2, "unsafe_touchdown": 1, "out_of_bounds": 1}
    assert summary["failed_checks"] == {"vertical speed": 1, "tilt": 1}
    assert summary["touchdown"]["vertical speed [m/s]"]["max"] == pytest.approx(0.6)
    assert "landed 50.0% of 4 episodes" in format_summary(summary)


@pytest.mark.parametrize(
    ("y", "vy", "possible"),
    [
        (50.0, -10.0, True),  # needs 4.9 m to stop, has 48.6 m
        (10.0, -30.0, False),  # needs 44 m, has 8.6 m
        (5.0, 2.0, True),  # going up
    ],
)
def test_feasibility_compares_stopping_distance_with_height(
    env: RocketLandingEnv, y: float, vy: float, possible: bool
) -> None:
    start = RocketState(x=0.0, y=y, vx=0.0, vy=vy)
    assert can_stop_before_the_ground(env.rocket_simulator, start) is possible


def test_plot_is_saved(env: RocketLandingEnv, tmp_path) -> None:
    results = [
        _result(Outcome.LANDED, []),
        _result(Outcome.UNSAFE_TOUCHDOWN, [LandingCheck("off center", 0.8, 0.5, "m")]),
    ]
    path = plot_evaluation(results, env.env_config.landing, "test", tmp_path / "plot.png")
    assert path.exists() and path.stat().st_size > 10_000


def _result(
    outcome: Outcome, failed_checks: list[LandingCheck], touchdown: bool = True
) -> EpisodeResult:
    contact = RocketState(
        x=0.1, y=1.44, vx=0.05, vy=-0.6, tilt=math.radians(1.0), spin=math.radians(0.5)
    )
    return EpisodeResult(
        seed=seed_for_test_episode(0),
        outcome=outcome,
        failed_checks=failed_checks,
        touchdown=Touchdown(contact, frozenset({"leg"})) if touchdown else None,
        duration_s=10.0,
        total_reward=50.0,
        fuel_used=5.0,
        trajectory=[(0.0, 50.0), (0.0, 1.5)],
    )
