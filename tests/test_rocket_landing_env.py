import dataclasses
import math

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from rocketlander.env_config import EnvConfig, load_env_config
from rocketlander.rocket_landing_env import DEFAULT_ENV_CONFIG, RocketLandingEnv
from rocketlander.rocket_state import RocketState

STANDING = 1.44  # center-of-mass height when standing on the legs [m]
HOVER = 100 * 9.81 / 2000  # main throttle whose thrust equals the weight
HOVER_ACTION = np.array([2 * HOVER - 1, -1.0, -1.0], dtype=np.float32)  # nozzles off = -1


@pytest.fixture(scope="module")
def config() -> EnvConfig:
    return load_env_config(DEFAULT_ENV_CONFIG)


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
def test_passes_gymnasium_env_checker() -> None:
    check_env(RocketLandingEnv(), skip_render_check=True)


@pytest.mark.parametrize("level", [0, 1, 2])
def test_random_starts_stay_within_the_level_ranges(config: EnvConfig, level: int) -> None:
    env, ranges = RocketLandingEnv(level=level), config.levels[level]
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


def test_same_seed_and_actions_give_the_same_flight() -> None:
    def flight(seed: int) -> list[np.ndarray]:
        env = RocketLandingEnv(level=1)
        env.reset(seed=seed)
        actions = np.random.default_rng(seed).uniform(-1, 1, size=(200, 3)).astype(np.float32)
        return [env.step(a)[0] for a in actions]

    assert all(np.array_equal(a, b) for a, b in zip(flight(7), flight(7), strict=True))
    assert not np.array_equal(flight(7)[-1], flight(8)[-1])


def test_actions_map_to_throttles() -> None:
    env = RocketLandingEnv()
    start = RocketState(0.0, 50.0, 0.0, 0.0)

    _start(env, start)
    vy_off = env.step(np.array([-1, -1, -1], dtype=np.float32))[0][3]  # all nozzles off
    _start(env, start)
    vy_full = env.step(np.array([1, -1, -1], dtype=np.float32))[0][3]  # main at full

    assert vy_off == pytest.approx(-9.81 * 0.02, rel=1e-3)
    assert vy_full == pytest.approx((2000 / 100 - 9.81) * 0.02, rel=1e-3)


def test_gentle_upright_touchdown_on_the_pad_is_a_landing(config: EnvConfig) -> None:
    env = RocketLandingEnv()
    _start(env, RocketState(0.1, STANDING + 0.3, 0.0, -0.5))

    total, (_, reward, terminated, truncated, info) = _run(env, HOVER_ACTION)

    assert (terminated, truncated) == (True, False)
    assert info["outcome"] == "landed"
    assert info["failed_checks"] == []
    assert info["touchdown"]["vy"] == pytest.approx(-0.5, abs=0.1)
    assert reward > config.reward.landed - 5


def test_fast_touchdown_is_unsafe_and_penalized_by_impact_speed(config: EnvConfig) -> None:
    env = RocketLandingEnv()
    _start(env, RocketState(0.0, STANDING + 0.5, 0.0, -8.0))

    _, (_, reward, terminated, _, info) = _run(env, HOVER_ACTION)

    assert terminated
    assert info["outcome"] == "unsafe_touchdown"
    # Hover thrust cancels gravity, so the rocket keeps its 8 m/s down to the ground.
    assert info["failed_checks"] == ["vertical speed 8.00 > 1 m/s"]
    impact_speed = abs(info["touchdown"]["vy"])
    expected = (
        config.reward.unsafe_touchdown + config.reward.unsafe_touchdown_per_speed * impact_speed
    )
    assert reward == pytest.approx(expected, abs=1.0)


@pytest.mark.parametrize(
    ("x", "tilt_deg", "failed"),
    [
        (2.0, 0.0, ["x error"]),  # soft and upright, but 2 m off the pad
        # Tilted 8°: the main engine also pushes sideways, so it drifts on the way down.
        (0.0, 8.0, ["horizontal speed", "tilt"]),
    ],
)
def test_soft_touchdown_off_the_pad_or_tilted_is_unsafe(
    x: float, tilt_deg: float, failed: list[str]
) -> None:
    env = RocketLandingEnv()
    _start(env, RocketState(x, STANDING + 0.3, 0.0, -0.5, theta=math.radians(tilt_deg)))

    _, (*_, info) = _run(env, HOVER_ACTION)

    assert info["outcome"] == "unsafe_touchdown"
    names = [check.rsplit(" ", 4)[0] for check in info["failed_checks"]]
    assert names == failed


def test_body_hitting_the_ground_is_unsafe(config: EnvConfig) -> None:
    # Upside down just above the ground, with the tilt limit raised so the nose hits first.
    limits = dataclasses.replace(config.limits, max_tilt_deg=180.0)
    env = RocketLandingEnv(config=dataclasses.replace(config, limits=limits))
    _start(env, RocketState(0.0, 1.76 + 0.1, 0.0, -3.0, theta=math.pi))

    _, (*_, info) = _run(env, np.array([-1, -1, -1], dtype=np.float32))

    assert info["outcome"] == "unsafe_touchdown"
    assert "body touched the ground" in info["failed_checks"]


def test_tilting_past_the_limit_loses_control(config: EnvConfig) -> None:
    env = RocketLandingEnv()
    _start(env, RocketState(0.0, 100.0, 0.0, 0.0, theta=math.radians(89), omega=1.0))

    _, (_, reward, terminated, _, info) = _run(env, HOVER_ACTION)

    assert terminated and info["outcome"] == "lost_control"
    assert reward == pytest.approx(config.reward.lost_control, abs=1.0)


def test_leaving_the_flight_region_ends_the_episode(config: EnvConfig) -> None:
    env = RocketLandingEnv()
    _start(env, RocketState(config.limits.max_x - 0.05, 100.0, 5.0, 0.0))

    _, (*_, terminated, _, info) = _run(env, HOVER_ACTION)

    assert terminated and info["outcome"] == "out_of_bounds"


def test_time_limit_truncates_the_episode(config: EnvConfig) -> None:
    limits = dataclasses.replace(config.limits, time_limit=1.0)
    env = RocketLandingEnv(config=dataclasses.replace(config, limits=limits))
    _start(env, RocketState(0.0, 100.0, 0.0, 0.0))

    steps = 0
    while True:
        *_, terminated, truncated, info = env.step(HOVER_ACTION)
        steps += 1
        if terminated or truncated:
            break

    assert (terminated, truncated, steps) == (False, True, 50)
    assert info["outcome"] == "time_limit"


@pytest.mark.parametrize(("vy", "sign"), [(-5.0, +1), (+5.0, -1)])
def test_moving_toward_the_pad_is_rewarded_and_away_is_penalized(vy: float, sign: int) -> None:
    env = RocketLandingEnv()
    _start(env, RocketState(0.0, 20.0, 0.0, vy))

    reward = env.step(HOVER_ACTION)[1]

    assert math.copysign(1, reward) == sign
