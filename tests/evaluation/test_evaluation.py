import numpy as np
import pytest

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
from rocketlander.pd.pd_controller import PDController

ALL_OFF = np.array([-1.0, -1.0, -1.0], dtype=np.float32)


@pytest.fixture(scope="module")
def env() -> RocketLandingEnv:
    return RocketLandingEnv()


def test_test_seeds_are_far_from_training_seeds() -> None:
    assert seed_for_test_episode(0) == 1_000_000
    assert seed_for_test_episode(137) == 1_000_137


def test_run_episode_records_the_flight(env: RocketLandingEnv) -> None:
    result = run_episode(env, lambda obs: ALL_OFF, seed_for_test_episode(0))

    start = env.reset(seed=seed_for_test_episode(0))[0]
    assert result.trajectory[0] == pytest.approx((start[0], start[1]))
    assert result.outcome == "unsafe_touchdown"  # free fall
    assert result.fuel_used == 0.0
    assert result.duration_s == pytest.approx(0.02 * (len(result.trajectory) - 1))


def test_fuel_counts_seconds_of_full_main_thrust(env: RocketLandingEnv) -> None:
    half_main = np.array([0.0, -1.0, -1.0], dtype=np.float32)  # main at 50%
    result = run_episode(env, lambda obs: half_main, seed_for_test_episode(0))
    assert result.fuel_used == pytest.approx(0.5 * result.duration_s)


def test_evaluation_is_repeatable(env: RocketLandingEnv) -> None:
    controller = PDController.for_env(env).act
    first = summarize(run_test_set(controller, env, episodes=3))
    second = summarize(run_test_set(controller, env, episodes=3))
    assert first == second


def test_summary_counts_outcomes_and_failed_checks(env: RocketLandingEnv) -> None:
    results = [
        _result("landed", []),
        _result("unsafe_touchdown", ["vertical speed 8.00 > 1 m/s", "tilt 6.00 > 5 deg"]),
        _result("out_of_bounds", [], touchdown=False),
        _result("landed", []),
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
    observation = np.array([0, y, 0, vy, 0, 1, 0, 100], dtype=np.float32)
    assert can_stop_before_the_ground(env, observation) is possible


def test_plot_is_saved(env: RocketLandingEnv, tmp_path) -> None:
    results = [_result("landed", []), _result("unsafe_touchdown", ["off center 0.80 > 0.5 m"])]
    path = plot_evaluation(results, env.config.landing, "test", tmp_path / "plot.png")
    assert path.exists() and path.stat().st_size > 10_000


def _result(outcome: str, failed: list[str], touchdown: bool = True) -> EpisodeResult:
    return EpisodeResult(
        seed=seed_for_test_episode(0),
        outcome=outcome,
        failed_checks=failed,
        touchdown={"x": 0.1, "vx": 0.05, "vy": -0.6, "tilt_deg": 1.0, "spin_deg_s": 0.5}
        if touchdown
        else None,
        duration_s=10.0,
        fuel_used=5.0,
        trajectory=[(0.0, 50.0), (0.0, 1.5)],
    )
