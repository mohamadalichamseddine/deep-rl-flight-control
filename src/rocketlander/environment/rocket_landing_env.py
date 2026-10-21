"""Gymnasium environment: land the rocket on the pad at x = 0."""

import math
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rocketlander.environment.action import action_to_throttles
from rocketlander.environment.env_config import EnvConfig
from rocketlander.environment.landing_judge import LandingJudge
from rocketlander.environment.landing_reward import LandingReward
from rocketlander.environment.observation import state_to_observation
from rocketlander.rocket.rocket_simulator import RocketSimulator
from rocketlander.rocket.rocket_spec import RocketSpec
from rocketlander.rocket.rocket_state import RocketState


class RocketLandingEnv(gym.Env):
    """Land the rocket on the pad at x = 0, starting from a random state in the sky.

    Observation, in physical units: ``[x, y, vx, vy, sin(tilt), cos(tilt), spin, mass]``,
    where (x, y) is the center of mass (see ``state_to_observation``).

    Action: one value in [-1, 1] per nozzle (main, left, right), mapped to throttles in [0, 1].

    The episode ends (terminated) at the first ground contact, judged as a safe landing or
    an unsafe touchdown; when the rocket tilts past the limit; when it leaves the flight
    region; or at the time limit, which stands in for running out of fuel (see ``Outcome``).

    ``info`` holds, every step, "fuel_used": seconds of full main-engine thrust used in the
    step. The final step adds "outcome" (an ``Outcome``), "touchdown" (the ``Touchdown``, or
    None) and "failed_checks" (the ``LandingCheck``s a touchdown went past).

    ``reset(options={"state": RocketState(...)})`` starts from a given state instead of a
    random one.

    ``render_mode=None`` draws nothing (fastest, for training); ``"human"`` shows a live
    window paced to real time; ``"rgb_array"`` makes ``render()`` return images.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        rocket_spec: RocketSpec,
        env_config: EnvConfig,
        level: int,
        render_mode: str | None,
    ) -> None:
        if not 0 <= level < len(env_config.levels):
            raise ValueError(f"level must be 0-{len(env_config.levels) - 1}, got {level}")
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"render_mode must be one of {self.metadata['render_modes']}")
        self.rocket_spec = rocket_spec
        self.env_config = env_config
        self.level = level
        self.render_mode = render_mode
        self.environment_renderer = None  # created on the first render
        self.rocket_simulator = RocketSimulator(
            rocket_spec, gravity=env_config.gravity, dt=env_config.dt, substeps=env_config.substeps
        )
        self.landing_judge = LandingJudge(
            env_config.landing,
            env_config.limits,
            max_steps=round(env_config.limits.time_limit / env_config.dt),
        )
        self.landing_reward = LandingReward(
            env_config.reward, self.rocket_simulator.y_when_standing, env_config.dt
        )

        n_nozzles = len(rocket_spec.nozzles)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(n_nozzles,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32)
        self._max_thrust = max(nozzle.max_thrust for nozzle in rocket_spec.nozzles)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        start = (options or {}).get("state") or self._random_start()
        self.rocket_simulator.reset(start)
        self.landing_reward.start(start)
        self._steps = 0
        self._throttles = [0.0] * len(self.rocket_spec.nozzles)
        self._info: dict[str, Any] = {}
        if self.environment_renderer is not None:
            self.environment_renderer.start_episode()
        if self.render_mode == "human":
            self.render()
        return state_to_observation(start, self.rocket_simulator.mass), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        throttles = action_to_throttles(action)
        state = self.rocket_simulator.step(throttles.tolist())
        self._steps += 1

        touchdown = self.rocket_simulator.touchdown
        if touchdown is not None:
            # The episode ends at the moment of contact.
            state = touchdown.state
        outcome = self.landing_judge.outcome(state, touchdown, self._steps)
        failed_checks = self.landing_judge.failed_checks(touchdown) if touchdown else []
        fuel_used = self._fuel_used(throttles)
        reward = self.landing_reward.step(state, fuel_used, outcome, touchdown, failed_checks)

        info: dict[str, Any] = {"fuel_used": fuel_used}
        if outcome is not None:
            info |= {"outcome": outcome, "touchdown": touchdown, "failed_checks": failed_checks}
        # Every ending is a real ending, including the time limit (out of fuel), so nothing
        # is ever cut off early: truncated stays False.
        terminated, truncated = outcome is not None, False
        self._throttles, self._info = throttles.tolist(), info
        if self.render_mode == "human":
            self.render()
        observation = state_to_observation(state, self.rocket_simulator.mass)
        return observation, float(reward), terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode is None:
            gym.logger.warn("render() called with render_mode=None; nothing to draw")
            return None
        if self.environment_renderer is None:
            from rocketlander.visualization.environment_renderer import (
                EnvironmentRenderer,  # pygame only when drawing
            )

            self.environment_renderer = EnvironmentRenderer(
                self.render_mode, self.metadata["render_fps"]
            )
            self.environment_renderer.start_episode()
        time_s = self._steps * self.env_config.dt
        self.environment_renderer.draw(
            self.rocket_simulator, self._throttles, time_s, self.level, self._info
        )
        if self.render_mode == "human":
            self.environment_renderer.show(episode_over="outcome" in self._info)
            return None
        return self.environment_renderer.to_array()

    @property
    def window_closed(self) -> bool:
        """True once the person watching has closed the window (or pressed Esc)."""
        return self.environment_renderer is not None and self.environment_renderer.closed

    def close(self) -> None:
        if self.environment_renderer is not None:
            self.environment_renderer.close()
            self.environment_renderer = None

    def _random_start(self) -> RocketState:
        ranges = self.env_config.levels[self.level]

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

    def _fuel_used(self, throttles: np.ndarray) -> float:
        """Seconds of full main-engine thrust used in one step (side nozzles count by thrust)."""
        thrust = sum(
            u * n.max_thrust for u, n in zip(throttles, self.rocket_spec.nozzles, strict=True)
        )
        return float(thrust / self._max_thrust * self.env_config.dt)
