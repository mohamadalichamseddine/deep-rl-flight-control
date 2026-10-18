"""Gymnasium environment: land the rocket on the pad at x = 0."""

import math
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rocketlander.environment.env_config import EnvConfig, load_env_config
from rocketlander.rocket.rocket_simulator import RocketSimulator, Touchdown
from rocketlander.rocket.rocket_spec import RocketSpec, load_rocket
from rocketlander.rocket.rocket_state import RocketState

CONFIGS = Path(__file__).resolve().parents[3] / "configs"
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
        self.simulator = RocketSimulator(
            self.rocket, dt=self.config.dt, substeps=self.config.substeps
        )

        n_nozzles = len(self.rocket.nozzles)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(n_nozzles,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32)

        # The center of mass's y when the rocket stands on its legs: where the landing score
        # is highest.
        lowest_tip = min(leg.tip[1] for leg in self.rocket.legs)
        self.y_when_standing = self.simulator.center_of_mass_height - lowest_tip
        self._max_thrust = max(nozzle.max_thrust for nozzle in self.rocket.nozzles)
        self._max_steps = round(self.config.limits.time_limit / self.config.dt)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        start = (options or {}).get("state") or self._random_start()
        self.simulator.reset(start)
        self._steps = 0
        self._previous_score = self._landing_score(start)
        self._throttles = [0.0] * len(self.rocket.nozzles)
        self._info: dict[str, Any] = {}
        if self.renderer is not None:
            self.renderer.start_episode()
        if self.render_mode == "human":
            self.render()
        return self._observation(start), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        throttles = (np.clip(action, -1.0, 1.0) + 1.0) / 2.0
        state = self.simulator.step(throttles.tolist())
        self._steps += 1

        touchdown = self.simulator.touchdown
        if touchdown is not None:
            # The episode ends at the moment of contact.
            state = touchdown.state
        outcome, info = self._episode_outcome(state, touchdown)

        # Reward: how much the landing score improved in this step, minus the fuel used, plus
        # a bonus or penalty when the episode ends.
        score = self._landing_score(state)
        reward = score - self._previous_score - self._fuel_cost(throttles)
        self._previous_score = score
        reward += self._end_of_episode_reward(outcome, touchdown)

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
            from rocketlander.visualization.rocket_renderer import (
                RocketRenderer,  # pygame only when drawing
            )

            self.renderer = RocketRenderer(self.render_mode, self.metadata["render_fps"])
            self.renderer.start_episode()
        time_s = self._steps * self.config.dt
        self.renderer.draw(self.simulator, self._throttles, time_s, self.level, self._info)
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
            tilt=math.radians(draw(ranges.tilt_deg)),
            spin=math.radians(draw(ranges.spin_deg_s)),
        )

    def _observation(self, state: RocketState) -> np.ndarray:
        return np.array(
            [
                state.x,
                state.y,
                state.vx,
                state.vy,
                math.sin(state.tilt),
                math.cos(state.tilt),
                state.spin,
                self.simulator.mass,
            ],
            dtype=np.float32,
        )

    def _episode_outcome(
        self, state: RocketState, touchdown: Touchdown | None
    ) -> tuple[str | None, dict[str, Any]]:
        """How the episode ends at this step, if it does, with details for ``info``."""
        limits = self.config.limits
        if touchdown is not None:
            failed = self._failed_landing_checks(touchdown)
            outcome = "unsafe_touchdown" if failed else "landed"
            contact = touchdown.state
            details = {
                "x": contact.x,
                "vx": contact.vx,
                "vy": contact.vy,
                "tilt_deg": math.degrees(contact.tilt),
                "spin_deg_s": math.degrees(contact.spin),
            }
            return outcome, {"outcome": outcome, "failed_checks": failed, "touchdown": details}
        if abs(state.tilt) > math.radians(limits.max_tilt_deg):
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
        criteria, contact = self.config.landing, touchdown.state
        return [
            ("off center", abs(contact.x), criteria.max_off_center, "m"),
            ("horizontal speed", abs(contact.vx), criteria.max_vx, "m/s"),
            ("vertical speed", abs(contact.vy), criteria.max_vy, "m/s"),
            ("tilt", math.degrees(abs(contact.tilt)), criteria.max_tilt_deg, "deg"),
            ("spin", math.degrees(abs(contact.spin)), criteria.max_spin_deg_s, "deg/s"),
        ]

    def _failed_landing_checks(self, touchdown: Touchdown) -> list[str]:
        failed = [f"{part} touched the ground" for part in sorted(touchdown.parts - {"leg"})]
        failed += [
            f"{name} {value:.2f} > {limit:g} {unit}"
            for name, value, limit, unit in self._landing_checks(touchdown)
            if value > limit
        ]
        return failed

    def _landing_score(self, state: RocketState) -> float:
        """Higher (closer to 0) when near the landing point, slow, and upright."""
        reward = self.config.reward
        distance = math.hypot(state.x, state.y - self.y_when_standing)
        speed = math.hypot(state.vx, state.vy)
        return -(
            reward.distance_weight * distance
            + reward.speed_weight * speed
            + reward.tilt_weight * abs(state.tilt)
        )

    def _fuel_cost(self, throttles: np.ndarray) -> float:
        thrust = sum(u * n.max_thrust for u, n in zip(throttles, self.rocket.nozzles, strict=True))
        return self.config.reward.fuel_weight * thrust / self._max_thrust * self.config.dt

    def _end_of_episode_reward(self, outcome: str | None, touchdown: Touchdown | None) -> float:
        reward = self.config.reward
        if outcome == "landed":
            return reward.landed_bonus
        if outcome == "unsafe_touchdown":
            if touchdown.parts - {"leg"}:  # the body hit the ground: as bad as it gets
                return reward.lost_control_penalty
            # Graded: a near miss costs a little, a crash a lot. Each failed check adds how far
            # past its limit the touchdown was, in limits (0.6 m off center = 0.2 of 0.5 m).
            missed = sum(
                value / limit - 1
                for _, value, limit, _ in self._landing_checks(touchdown)
                if value > limit
            )
            penalty = reward.unsafe_touchdown_penalty + reward.penalty_per_limit_missed * missed
            return max(penalty, reward.lost_control_penalty)
        if outcome in ("lost_control", "out_of_bounds"):
            return reward.lost_control_penalty
        if outcome == "time_limit":
            return reward.time_limit_penalty
        return 0.0
