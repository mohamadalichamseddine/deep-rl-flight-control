import dataclasses
import math

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from rocketlander.environment.env_config import EnvConfig
from rocketlander.environment.outcome import Outcome
from rocketlander.environment.rocket_landing_env import RocketLandingEnv
from rocketlander.rocket.rocket_spec import RocketSpec
from rocketlander.rocket.rocket_state import RocketState

STANDING = 1.44  # center-of-mass height when standing on the legs [m]
HOVER = 100 * 9.81 / 2000  # main throttle whose thrust equals the weight
HOVER_ACTION = np.array([2 * HOVER - 1, -1.0, -1.0], dtype=np.float32)  # nozzles off = -1


def _start(env: RocketLandingEnv, state: RocketState) -> np.ndarray:
    obs, _ = env.reset(seed=0, options={"state": state})
    return obs


def _run(env: RocketLandingEnv, action: np.ndarray, max_steps: int = 5000):
    """Apply the same action until the episode ends; return the total reward and last step."""
    total = 0.0
    for _ in range(max_steps):
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        if terminated or truncated:
            return total, (obs, reward, terminated, truncated, info)
    raise AssertionError("episode did not end")


@pytest.mark.filterwarnings("ignore:.*observation space m(in|ax)imum value is:UserWarning")
def test_passes_gymnasium_env_checker(rocket_spec: RocketSpec, env_config: EnvConfig) -> None:
    check_env(
        RocketLandingEnv(rocket_spec, env_config, level=0, render_mode=None), skip_render_check=True
    )


@pytest.mark.parametrize("level", [0, 1, 2])
def test_random_starts_stay_within_the_level_ranges(
    rocket_spec: RocketSpec, env_config: EnvConfig, level: int
) -> None:
    env, ranges = (
        RocketLandingEnv(rocket_spec, env_config, level=level, render_mode=None),
        env_config.levels[level],
    )
    for seed in range(100):
        x, y, vx, vy, sin_tilt, _, spin, mass = env.reset(seed=seed)[0]
        assert ranges.x[0] <= x <= ranges.x[1]
        assert ranges.altitude[0] <= y <= ranges.altitude[1]
        assert ranges.vx[0] <= vx <= ranges.vx[1]
        assert ranges.vy[0] <= vy <= ranges.vy[1]
        tilt_deg = math.degrees(math.asin(sin_tilt))
        assert ranges.tilt_deg[0] - 1e-3 <= tilt_deg <= ranges.tilt_deg[1] + 1e-3
        spin_deg_s = math.degrees(spin)
        assert ranges.spin_deg_s[0] - 1e-3 <= spin_deg_s <= ranges.spin_deg_s[1] + 1e-3
        assert mass == pytest.approx(100.0)


def test_same_seed_and_actions_give_the_same_flight(
    rocket_spec: RocketSpec, env_config: EnvConfig
) -> None:
    def flight(seed: int) -> list[np.ndarray]:
        env = RocketLandingEnv(rocket_spec, env_config, level=1, render_mode=None)
        env.reset(seed=seed)
        actions = np.random.default_rng(seed).uniform(-1, 1, size=(200, 3)).astype(np.float32)
        return [env.step(a)[0] for a in actions]

    assert all(np.array_equal(a, b) for a, b in zip(flight(7), flight(7), strict=True))
    assert not np.array_equal(flight(7)[-1], flight(8)[-1])


def test_actions_map_to_throttles(rocket_spec: RocketSpec, env_config: EnvConfig) -> None:
    env = RocketLandingEnv(rocket_spec, env_config, level=0, render_mode=None)
    start = RocketState(0.0, 50.0, 0.0, 0.0)

    _start(env, start)
    vy_off = env.step(np.array([-1, -1, -1], dtype=np.float32))[0][3]  # all nozzles off
    _start(env, start)
    vy_full = env.step(np.array([1, -1, -1], dtype=np.float32))[0][3]  # main at full

    assert vy_off == pytest.approx(-9.81 * 0.02, rel=1e-3)
    assert vy_full == pytest.approx((2000 / 100 - 9.81) * 0.02, rel=1e-3)


def test_gentle_upright_touchdown_on_the_pad_is_a_landing(
    rocket_spec: RocketSpec, env_config: EnvConfig
) -> None:
    env = RocketLandingEnv(rocket_spec, env_config, level=0, render_mode=None)
    _start(env, RocketState(0.1, STANDING + 0.3, 0.0, -0.15))

    total, (_, reward, terminated, truncated, info) = _run(env, HOVER_ACTION)

    assert (terminated, truncated) == (True, False)
    assert info["outcome"] == Outcome.LANDED
    assert info["failed_checks"] == []
    assert info["touchdown"].state.vy == pytest.approx(-0.15, abs=0.05)
    assert reward > env_config.reward.landed_bonus - 5


@pytest.mark.parametrize("vy", [-0.4, -1.0, -8.0, -40.0])  # near miss, harder, crash, capped
def test_unsafe_touchdown_penalty_grows_with_the_miss(
    rocket_spec: RocketSpec, env_config: EnvConfig, vy: float
) -> None:
    reward_settings = env_config.reward
    missed = abs(vy) / 0.3 - 1  # limits' worth past the 0.3 m/s limit
    expected = max(
        reward_settings.unsafe_touchdown_penalty
        + reward_settings.penalty_per_limit_missed * missed,
        reward_settings.lost_control_penalty,
    )
    env = RocketLandingEnv(rocket_spec, env_config, level=0, render_mode=None)
    _start(env, RocketState(0.0, STANDING + 0.5, 0.0, vy))

    _, (_, reward, terminated, _, info) = _run(env, HOVER_ACTION)

    assert terminated
    assert info["outcome"] == Outcome.UNSAFE_TOUCHDOWN
    # Hover thrust cancels gravity, so the rocket keeps its speed down to the ground.
    [check] = info["failed_checks"]
    assert (check.name, check.limit, check.unit) == ("vertical speed", 0.3, "m/s")
    assert check.value == pytest.approx(abs(vy), abs=0.005)
    assert reward == pytest.approx(expected, abs=1.0)


@pytest.mark.parametrize(
    ("x", "tilt_deg", "failed"),
    [
        (2.0, 0.0, ["off center"]),  # soft and upright, but 2 m off the pad
        (0.0, 8.0, ["tilt"]),  # soft and on the pad, but tilted 8° (limit 5°)
    ],
)
def test_soft_touchdown_off_the_pad_or_tilted_is_unsafe(
    rocket_spec: RocketSpec, env_config: EnvConfig, x: float, tilt_deg: float, failed: list[str]
) -> None:
    env = RocketLandingEnv(rocket_spec, env_config, level=0, render_mode=None)
    _start(env, RocketState(x, STANDING + 0.1, 0.0, -0.15, tilt=math.radians(tilt_deg)))

    _, (*_, info) = _run(env, HOVER_ACTION)

    assert info["outcome"] == Outcome.UNSAFE_TOUCHDOWN
    names = [check.name for check in info["failed_checks"]]
    assert names == failed


def test_body_hitting_the_ground_is_unsafe(rocket_spec: RocketSpec, env_config: EnvConfig) -> None:
    # Upside down just above the ground, with the tilt limit raised so the nose hits first.
    limits = dataclasses.replace(env_config.limits, max_tilt_deg=180.0)
    env = RocketLandingEnv(
        rocket_spec, dataclasses.replace(env_config, limits=limits), level=0, render_mode=None
    )
    _start(env, RocketState(0.0, 1.76 + 0.1, 0.0, -3.0, tilt=math.pi))

    _, (*_, info) = _run(env, np.array([-1, -1, -1], dtype=np.float32))

    assert info["outcome"] == Outcome.UNSAFE_TOUCHDOWN
    assert "body" in info["touchdown"].parts


def test_tilting_past_the_limit_loses_control(
    rocket_spec: RocketSpec, env_config: EnvConfig
) -> None:
    env = RocketLandingEnv(rocket_spec, env_config, level=0, render_mode=None)
    _start(env, RocketState(0.0, 100.0, 0.0, 0.0, tilt=math.radians(89), spin=1.0))

    _, (_, reward, terminated, _, info) = _run(env, HOVER_ACTION)

    assert terminated and info["outcome"] == Outcome.LOST_CONTROL
    assert reward == pytest.approx(env_config.reward.lost_control_penalty, abs=1.0)


def test_leaving_the_flight_region_ends_the_episode(
    rocket_spec: RocketSpec, env_config: EnvConfig
) -> None:
    env = RocketLandingEnv(rocket_spec, env_config, level=0, render_mode=None)
    _start(env, RocketState(env_config.limits.max_x - 0.05, 100.0, 5.0, 0.0))

    _, (*_, terminated, _, info) = _run(env, HOVER_ACTION)

    assert terminated and info["outcome"] == Outcome.OUT_OF_BOUNDS


def test_running_out_of_time_is_a_failure(rocket_spec: RocketSpec, env_config: EnvConfig) -> None:
    limits = dataclasses.replace(env_config.limits, time_limit=1.0)
    env = RocketLandingEnv(
        rocket_spec, dataclasses.replace(env_config, limits=limits), level=0, render_mode=None
    )
    _start(env, RocketState(0.0, 100.0, 0.0, 0.0))

    steps = 0
    while True:
        _, reward, terminated, truncated, info = env.step(HOVER_ACTION)
        steps += 1
        if terminated or truncated:
            break

    assert (terminated, truncated, steps) == (True, False, 50)
    assert info["outcome"] == Outcome.TIME_LIMIT
    assert reward == pytest.approx(env_config.reward.time_limit_penalty, abs=1.0)


def test_every_second_in_the_air_costs_time_weight(
    rocket_spec: RocketSpec, env_config: EnvConfig
) -> None:
    env = RocketLandingEnv(rocket_spec, env_config, level=0, render_mode=None)
    _start(env, RocketState(0.0, 50.0, 0.0, 0.0))

    _, reward, _, _, info = env.step(HOVER_ACTION)

    # Hovering in place keeps the landing score, so only the fuel and the time cost.
    reward_settings = env_config.reward
    fuel = reward_settings.fuel_weight * info["fuel_used"]
    assert reward == pytest.approx(-fuel - reward_settings.time_weight * 0.02, abs=1e-4)


@pytest.mark.parametrize(("vy", "sign"), [(-5.0, +1), (+5.0, -1)])
def test_moving_toward_the_pad_is_rewarded_and_away_is_penalized(
    rocket_spec: RocketSpec, env_config: EnvConfig, vy: float, sign: int
) -> None:
    env = RocketLandingEnv(rocket_spec, env_config, level=0, render_mode=None)
    _start(env, RocketState(0.0, 20.0, 0.0, vy))

    reward = env.step(HOVER_ACTION)[1]

    assert math.copysign(1, reward) == sign
