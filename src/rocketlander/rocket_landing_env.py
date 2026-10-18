"""Gymnasium environment: land the rocket on the pad at x = 0."""

import math
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rocketlander.env_config import EnvConfig, load_env_config
from rocketlander.rocket_simulator import RocketSimulator, Touchdown
from rocketlander.rocket_spec import RocketSpec, load_rocket
from rocketlander.rocket_state import RocketState

CONFIGS = Path(__file__).resolve().parents[2] / "configs"
DEFAULT_ROCKET = CONFIGS / "rocket" / "r100.yaml"
DEFAULT_ENV_CONFIG = CONFIGS / "env" / "default.yaml"


class RocketLandingEnv(gym.Env):
    """Land the rocket on the pad at x = 0, starting from a random state in the sky.

    Observation, in physical units: ``[x, y, vx, vy, sin(tilt), cos(tilt), spin, mass]``,
    where (x, y) is the center of mass. sin/cos avoids a jump when the tilt wraps around.

    Action: one value in [-1, 1] per nozzle (main, left, right), mapped to throttles in [0, 1].

    The episode ends (terminated) at the first ground contact, judged as "landed" or
    "unsafe_touchdown"; when the rocket tilts past the limit ("lost_control"); or when it
    leaves the flight region ("out_of_bounds"); or at the time limit ("time_limit"), which
    stands in for running out of fuel. The final step's ``info`` holds the outcome and,
    after a touchdown, the touchdown state and any failed landing checks.

    ``reset(options={"state": RocketState(...)})`` starts from a given state instead of a
    random one.

    Rendering is off by default (fastest, for training). ``render_mode="human"`` shows a live
    window paced to real time; ``render_mode="rgb_array"`` makes ``render()`` return images.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        level: int = 0,
        rocket: RocketSpec | None = None,
        config: EnvConfig | None = None,
        render_mode: str | None = None,
    ) -> None:
        self.rocket = rocket or load_rocket(DEFAULT_ROCKET)
        self.config = config or load_env_config(DEFAULT_ENV_CONFIG)
        if not 0 <= level < len(self.config.levels):
            raise ValueError(f"level must be 0-{len(self.config.levels) - 1}, got {level}")
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"render_mode must be one of {self.metadata['render_modes']}")
        self.level = level
        self.render_mode = render_mode
        self.renderer = None  # created on the first render
        self.sim = RocketSimulator(self.rocket, dt=self.config.dt, substeps=self.config.substeps)

        n_nozzles = len(self.rocket.nozzles)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(n_nozzles,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32)

        # Center-of-mass height when standing on the legs: the target of the shaping reward.
        lowest_tip = min(leg.tip[1] for leg in self.rocket.legs)
        self.standing_height = self.sim.com_height - lowest_tip
        self._max_thrust = max(nozzle.max_thrust for nozzle in self.rocket.nozzles)
        self._max_steps = round(self.config.limits.time_limit / self.config.dt)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        start = (options or {}).get("state") or self._random_start()
        self.sim.reset(start)
        self._steps = 0
        self._potential = self._shaping_potential(start)
        self._throttles = [0.0] * len(self.rocket.nozzles)
        self._info: dict[str, Any] = {}
        if self.renderer is not None:
            self.renderer.start_episode()
        if self.render_mode == "human":
            self.render()
        return self._observation(start), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        throttles = (np.clip(action, -1.0, 1.0) + 1.0) / 2.0
        state = self.sim.step(throttles.tolist())
        self._steps += 1

        touchdown = self.sim.touchdown
        if touchdown is not None:
            # The episode ends at the moment of contact.
            state = touchdown.state
        outcome, info = self._outcome(state, touchdown)

        potential = self._shaping_potential(state)
        reward = potential - self._potential - self._fuel_cost(throttles)
        self._potential = potential
        reward += self._final_reward(outcome, touchdown)

        # Every ending is a real ending, including the time limit (out of fuel), so nothing
        # is ever cut off early: truncated stays False.
        terminated, truncated = outcome is not None, False
        self._throttles, self._info = throttles.tolist(), info
        if self.render_mode == "human":
            self.render()
        return self._observation(state), float(reward), terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode is None:
            gym.logger.warn("render() called with render_mode=None; nothing to draw")
            return None
        if self.renderer is None:
            from rocketlander.rocket_renderer import RocketRenderer  # pygame only when drawing

            self.renderer = RocketRenderer(self.render_mode, self.metadata["render_fps"])
            self.renderer.start_episode()
        time_s = self._steps * self.config.dt
        self.renderer.draw(self.sim, self._throttles, time_s, self.level, self._info)
        if self.render_mode == "human":
            self.renderer.show(episode_over="outcome" in self._info)
            return None
        return self.renderer.to_array()

    @property
    def window_closed(self) -> bool:
        """True once the person watching has closed the window (or pressed Esc)."""
        return self.renderer is not None and self.renderer.closed

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def _random_start(self) -> RocketState:
        ranges = self.config.levels[self.level]

        def draw(bounds: tuple[float, float]) -> float:
            return float(self.np_random.uniform(*bounds))

        return RocketState(
            x=draw(ranges.x),
            y=draw(ranges.altitude),
            vx=draw(ranges.vx),
            vy=draw(ranges.vy),
            theta=math.radians(draw(ranges.tilt_deg)),
            omega=math.radians(draw(ranges.spin_deg_s)),
        )

    def _observation(self, s: RocketState) -> np.ndarray:
        return np.array(
            [s.x, s.y, s.vx, s.vy, math.sin(s.theta), math.cos(s.theta), s.omega, self.sim.mass],
            dtype=np.float32,
        )

    def _outcome(
        self, state: RocketState, touchdown: Touchdown | None
    ) -> tuple[str | None, dict[str, Any]]:
        """How the episode ends at this step, if it does, with details for ``info``."""
        limits = self.config.limits
        if touchdown is not None:
            failed = self._failed_landing_checks(touchdown)
            outcome = "unsafe_touchdown" if failed else "landed"
            s = touchdown.state
            details = {
                "x": s.x,
                "vx": s.vx,
                "vy": s.vy,
                "tilt_deg": math.degrees(s.theta),
                "spin_deg_s": math.degrees(s.omega),
            }
            return outcome, {"outcome": outcome, "failed_checks": failed, "touchdown": details}
        if abs(state.theta) > math.radians(limits.max_tilt_deg):
            outcome = "lost_control"
        elif abs(state.x) > limits.max_x or state.y > limits.max_altitude:
            outcome = "out_of_bounds"
        elif self._steps >= self._max_steps:
            outcome = "time_limit"
        else:
            return None, {}
        return outcome, {"outcome": outcome}

    def _landing_checks(self, touchdown: Touchdown) -> list[tuple[str, float, float, str]]:
        """(name, value, limit, unit) of each check a safe landing must pass."""
        criteria, s = self.config.landing, touchdown.state
        return [
            ("x error", abs(s.x), criteria.max_x_error, "m"),
            ("horizontal speed", abs(s.vx), criteria.max_vx, "m/s"),
            ("vertical speed", abs(s.vy), criteria.max_vy, "m/s"),
            ("tilt", math.degrees(abs(s.theta)), criteria.max_tilt_deg, "deg"),
            ("spin", math.degrees(abs(s.omega)), criteria.max_spin_deg_s, "deg/s"),
        ]

    def _failed_landing_checks(self, touchdown: Touchdown) -> list[str]:
        failed = [f"{part} touched the ground" for part in sorted(touchdown.parts - {"leg"})]
        failed += [
            f"{name} {value:.2f} > {limit:g} {unit}"
            for name, value, limit, unit in self._landing_checks(touchdown)
            if value > limit
        ]
        return failed

    def _shaping_potential(self, s: RocketState) -> float:
        """Higher (closer to 0) when near the landing point, slow, and upright."""
        w = self.config.reward
        distance = math.hypot(s.x, s.y - self.standing_height)
        speed = math.hypot(s.vx, s.vy)
        return -(w.distance * distance + w.speed * speed + w.tilt * abs(s.theta))

    def _fuel_cost(self, throttles: np.ndarray) -> float:
        thrust = sum(u * n.max_thrust for u, n in zip(throttles, self.rocket.nozzles, strict=True))
        return self.config.reward.fuel * thrust / self._max_thrust * self.config.dt

    def _final_reward(self, outcome: str | None, touchdown: Touchdown | None) -> float:
        w = self.config.reward
        if outcome == "landed":
            return w.landed
        if outcome == "unsafe_touchdown":
            if touchdown.parts - {"leg"}:  # the body hit the ground: as bad as it gets
                return w.lost_control
            # Graded: a near miss costs a little, a crash a lot. Each failed check adds how far
            # past its limit the touchdown was, in limits (0.6 m off center = 0.2 of 0.5 m).
            missed = sum(
                value / limit - 1
                for _, value, limit, _ in self._landing_checks(touchdown)
                if value > limit
            )
            return max(w.unsafe_touchdown + w.per_limit_missed * missed, w.lost_control)
        if outcome in ("lost_control", "out_of_bounds"):
            return w.lost_control
        if outcome == "time_limit":
            return w.time_limit
        return 0.0
