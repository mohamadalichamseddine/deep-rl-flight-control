from dataclasses import dataclass


@dataclass(frozen=True)
class RocketState:
    """Where the rocket is and how it moves, in the world frame.

    World frame: x to the right, y up, ground at y = 0, landing pad centered at x = 0.
    Angles are counterclockwise-positive (the same convention as Box2D): tilt > 0 means
    the nose is tilted to the left (toward -x).
    """

    x: float  # center of mass [m]
    y: float  # center of mass [m]
    vx: float  # [m/s]
    vy: float  # [m/s]
    tilt: float = 0.0  # angle from vertical [rad], in [-pi, pi]
    spin: float = 0.0  # how fast the tilt changes (angular velocity) [rad/s]
