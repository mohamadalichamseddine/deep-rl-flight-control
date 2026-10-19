"""How an episode can end."""

from enum import StrEnum


class Outcome(StrEnum):
    LANDED = "landed"
    UNSAFE_TOUCHDOWN = "unsafe_touchdown"  # touched the ground outside the landing limits
    LOST_CONTROL = "lost_control"  # tilted past the limit
    OUT_OF_BOUNDS = "out_of_bounds"  # left the flight region
    TIME_LIMIT = "time_limit"  # ran out of time, which stands in for running out of fuel
