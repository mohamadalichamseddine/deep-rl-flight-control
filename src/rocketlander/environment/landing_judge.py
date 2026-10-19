"""Judging how an episode ends: a safe landing, an unsafe touchdown, or another ending."""

import math
from dataclasses import dataclass

from rocketlander.environment.env_config import FlightLimits, LandingCriteria
from rocketlander.environment.outcome import Outcome
from rocketlander.rocket.rocket_simulator import Touchdown
from rocketlander.rocket.rocket_state import RocketState


@dataclass(frozen=True)
class LandingCheck:
    """One limit a safe landing must stay within, and the touchdown's value for it."""

    name: str
    value: float
    limit: float
    unit: str


class LandingJudge:
    """Decides after each step whether the episode ends, and how."""

    def __init__(
        self, landing_criteria: LandingCriteria, flight_limits: FlightLimits, max_steps: int
    ) -> None:
        self.landing_criteria = landing_criteria
        self.flight_limits = flight_limits
        self.max_steps = max_steps

    def outcome(
        self, state: RocketState, touchdown: Touchdown | None, steps: int
    ) -> Outcome | None:
        """How the episode ends after ``steps`` steps, or None if it goes on."""
        if touchdown is not None:
            safe = not parts_other_than_legs(touchdown) and not self.failed_checks(touchdown)
            return Outcome.LANDED if safe else Outcome.UNSAFE_TOUCHDOWN
        limits = self.flight_limits
        if abs(state.tilt) > math.radians(limits.max_tilt_deg):
            return Outcome.LOST_CONTROL
        if abs(state.x) > limits.max_x or state.y > limits.max_altitude:
            return Outcome.OUT_OF_BOUNDS
        if steps >= self.max_steps:
            return Outcome.TIME_LIMIT
        return None

    def landing_checks(self, touchdown: Touchdown) -> list[LandingCheck]:
        """Each limit a safe landing must stay within, with the touchdown's value."""
        criteria, contact = self.landing_criteria, touchdown.state
        return [
            LandingCheck("off center", abs(contact.x), criteria.max_off_center, "m"),
            LandingCheck("horizontal speed", abs(contact.vx), criteria.max_vx, "m/s"),
            LandingCheck("vertical speed", abs(contact.vy), criteria.max_vy, "m/s"),
            LandingCheck("tilt", math.degrees(abs(contact.tilt)), criteria.max_tilt_deg, "deg"),
            LandingCheck("spin", math.degrees(abs(contact.spin)), criteria.max_spin_deg_s, "deg/s"),
        ]

    def failed_checks(self, touchdown: Touchdown) -> list[LandingCheck]:
        """The landing checks the touchdown went past."""
        return [check for check in self.landing_checks(touchdown) if check.value > check.limit]


def parts_other_than_legs(touchdown: Touchdown) -> frozenset[str]:
    """The rocket parts besides the legs that touched the ground, e.g. {"body"}."""
    return touchdown.parts - {"leg"}


def describe_failures(touchdown: Touchdown | None, failed_checks: list[LandingCheck]) -> list[str]:
    """Why a touchdown was unsafe, one reason per line, e.g. "vertical speed 0.40 > 0.3 m/s"."""
    if touchdown is None:
        return []
    reasons = [f"{part} touched the ground" for part in sorted(parts_other_than_legs(touchdown))]
    reasons += [
        f"{check.name} {check.value:.2f} > {check.limit:g} {check.unit}" for check in failed_checks
    ]
    return reasons
