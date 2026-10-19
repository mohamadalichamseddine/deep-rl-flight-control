"""The reward the network learns from."""

import math

from rocketlander.environment.env_config import RewardSettings
from rocketlander.environment.landing_judge import LandingCheck, parts_other_than_legs
from rocketlander.environment.outcome import Outcome
from rocketlander.rocket.rocket_simulator import Touchdown
from rocketlander.rocket.rocket_state import RocketState


class LandingReward:
    """Each step: how much the landing score improved, minus the fuel used. When the episode
    ends: a bonus for a safe landing, or a penalty.

    Call ``start`` with the first state of each episode, then ``step`` after every step.
    """

    def __init__(self, reward_settings: RewardSettings, y_when_standing: float) -> None:
        self.reward_settings = reward_settings
        self.y_when_standing = y_when_standing  # where the landing score is highest
        self._previous_score = 0.0

    def start(self, state: RocketState) -> None:
        self._previous_score = self.landing_score(state)

    def step(
        self,
        state: RocketState,
        fuel_used: float,
        outcome: Outcome | None,
        touchdown: Touchdown | None,
        failed_checks: list[LandingCheck],
    ) -> float:
        """The reward for the step that ended in ``state``."""
        score = self.landing_score(state)
        reward = score - self._previous_score - self.reward_settings.fuel_weight * fuel_used
        self._previous_score = score
        return reward + self.end_of_episode_reward(outcome, touchdown, failed_checks)

    def landing_score(self, state: RocketState) -> float:
        """Higher (closer to 0) when near the landing point, slow, and upright."""
        settings = self.reward_settings
        distance = math.hypot(state.x, state.y - self.y_when_standing)
        speed = math.hypot(state.vx, state.vy)
        return -(
            settings.distance_weight * distance
            + settings.speed_weight * speed
            + settings.tilt_weight * abs(state.tilt)
        )

    def end_of_episode_reward(
        self,
        outcome: Outcome | None,
        touchdown: Touchdown | None,
        failed_checks: list[LandingCheck],
    ) -> float:
        """The bonus or penalty for how the episode ended (0 while it goes on)."""
        settings = self.reward_settings
        if outcome is None:
            return 0.0
        if outcome == Outcome.LANDED:
            return settings.landed_bonus
        if outcome == Outcome.UNSAFE_TOUCHDOWN:
            if parts_other_than_legs(touchdown):  # the body hit the ground: as bad as it gets
                return settings.lost_control_penalty
            # Graded: a near miss costs a little, a crash a lot. Each failed check adds how far
            # past its limit the touchdown was, in limits (0.6 m off center = 0.2 of 0.5 m).
            missed = sum(check.value / check.limit - 1 for check in failed_checks)
            penalty = settings.unsafe_touchdown_penalty + settings.penalty_per_limit_missed * missed
            return max(penalty, settings.lost_control_penalty)
        if outcome in (Outcome.LOST_CONTROL, Outcome.OUT_OF_BOUNDS):
            return settings.lost_control_penalty
        return settings.time_limit_penalty  # Outcome.TIME_LIMIT
