"""The rocket's physical description, loaded from a YAML file such as configs/rocket/r100.yaml.

Positions are in the rocket's own (body) frame, in meters: origin at the center of the base,
+y up the rocket axis toward the nose, +x to the rocket's right when it stands upright.
Box2D calculates the center of mass and moment of inertia from the shapes (see rocket_simulator.py).
"""

import math
from dataclasses import dataclass
from pathlib import Path

from rocketlander.config_loader import load_config


@dataclass(frozen=True)
class Body:
    """The rocket body: a rectangle standing on the base, with its mass spread evenly."""

    height: float
    width: float
    mass: float

    def __post_init__(self) -> None:
        _require_positive(height=self.height, width=self.width, mass=self.mass)


@dataclass(frozen=True)
class EngineBlock:
    """Engine and legs: extra mass filling the bottom ``height`` meters of the body."""

    height: float
    mass: float

    def __post_init__(self) -> None:
        _require_positive(height=self.height, mass=self.mass)


@dataclass(frozen=True)
class Nozzle:
    name: str
    position: tuple[float, float]
    direction: tuple[float, float]  # unit vector: the way the force pushes the rocket
    max_thrust: float  # [N]

    def __post_init__(self) -> None:
        _require_positive(max_thrust=self.max_thrust)
        if not math.isclose(math.hypot(*self.direction), 1.0, abs_tol=1e-6):
            raise ValueError(f"nozzle {self.name!r}: direction must be a unit vector")


@dataclass(frozen=True)
class Leg:
    """A massless strut from the side of the body (top) down to the tip that touches ground."""

    top: tuple[float, float]
    tip: tuple[float, float]


@dataclass(frozen=True)
class RocketSpec:
    body: Body
    engine_block: EngineBlock
    nozzles: tuple[Nozzle, ...]  # order = order of the throttle commands
    legs: tuple[Leg, ...]

    def __post_init__(self) -> None:
        if self.engine_block.height > self.body.height:
            raise ValueError("engine_block.height must fit within the body")
        names = [n.name for n in self.nozzles]
        if len(set(names)) != len(names):
            raise ValueError(f"nozzle names must be unique, got {names}")
        if any(leg.tip[1] >= 0 for leg in self.legs):
            raise ValueError(
                "leg tips must reach below the base (y < 0), so the engine clears the ground"
            )

    def nozzle(self, name: str) -> Nozzle:
        for nozzle in self.nozzles:
            if nozzle.name == name:
                return nozzle
        raise KeyError(f"no nozzle named {name!r}; nozzles are {[n.name for n in self.nozzles]}")


def load_rocket(path: str | Path) -> RocketSpec:
    return load_config(path, RocketSpec)


def _require_positive(**values: float) -> None:
    for name, value in values.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
