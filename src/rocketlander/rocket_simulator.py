"""Rocket motion, computed by the Box2D physics engine.

The rocket is given to Box2D as shapes with masses. Box2D then calculates the center of mass
and moment of inertia, turns each nozzle's force (applied where the nozzle is mounted) into
pushing and turning, adds gravity, moves the rocket, and detects when it touches the ground.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import Box2D

from rocketlander.rocket_spec import Leg, RocketSpec
from rocketlander.rocket_state import RocketState

STANDARD_GRAVITY = 9.81  # [m/s²]

# Constraint-solver iterations, used when parts touch the ground.
_VELOCITY_ITERATIONS = 8
_POSITION_ITERATIONS = 3

_GROUND_HALF_WIDTH = 1000.0  # [m]
_LEG_THICKNESS = 0.04  # [m]


@dataclass(frozen=True)
class Touchdown:
    """The rocket's first contact with the ground."""

    state: RocketState  # just before the contact, before Box2D's contact forces act
    parts: frozenset[str]  # what touched: "leg", "body", and/or "engine block"


class RocketSimulator:
    """Moves the rocket under gravity and nozzle thrust above flat ground at y = 0, using Box2D.

    Each ``step`` holds the throttle commands for one control step of ``dt`` seconds, split
    into ``substeps`` smaller Box2D steps for accuracy. Box2D stores numbers as 32-bit floats,
    so states are accurate to about 7 digits.

    Safety cut-off: once the rocket has touched the ground, the engines are off, whatever
    throttles are commanded.
    """

    def __init__(
        self,
        spec: RocketSpec,
        gravity: float = STANDARD_GRAVITY,
        dt: float = 0.02,
        substeps: int = 5,
    ) -> None:
        if substeps < 1:
            raise ValueError(f"substeps must be >= 1, got {substeps}")
        self.spec = spec
        self.dt = dt
        self._gravity = gravity
        self._substeps = substeps
        self._build_world()

    @property
    def mass(self) -> float:
        """Total mass [kg], as calculated by Box2D."""
        return self._body.mass

    @property
    def com_height(self) -> float:
        """Height of the center of mass above the base [m], as calculated by Box2D."""
        return self._body.localCenter.y

    @property
    def moment_of_inertia(self) -> float:
        """Moment of inertia about the center of mass [kg·m²], as calculated by Box2D."""
        # Box2D reports it about the body's origin (the base); move it to the center of mass.
        body = self._body
        return body.inertia - body.mass * body.localCenter.lengthSquared

    def reset(self, state: RocketState) -> None:
        """Place the rocket in the given state, in a fresh world.

        Rebuilding the world clears contacts left over from the previous flight, so the same
        starting state always produces the same flight.
        """
        self._build_world()
        body = self._body
        body.angle = state.theta
        # The state gives the center of mass, but Box2D places a body by its origin.
        body.position = Box2D.b2Vec2(state.x, state.y) - body.GetWorldVector(body.localCenter)
        body.linearVelocity = (state.vx, state.vy)  # Box2D's velocity is the center of mass's
        body.angularVelocity = state.omega

    def step(self, throttles: Sequence[float]) -> RocketState:
        """Fire the nozzles at ``throttles`` (one per nozzle, in [0, 1]) for ``dt`` seconds.

        Throttles outside [0, 1] are clipped, as a real actuator saturates.
        """
        nozzles = self.spec.nozzles
        if len(throttles) != len(nozzles):
            raise ValueError(f"expected {len(nozzles)} throttles, got {len(throttles)}")
        forces = []
        for nozzle, throttle in zip(nozzles, throttles, strict=True):
            thrust = min(max(throttle, 0.0), 1.0) * nozzle.max_thrust
            force = (nozzle.direction[0] * thrust, nozzle.direction[1] * thrust)
            forces.append((force, nozzle.position))

        body = self._body
        h = self.dt / self._substeps
        for _ in range(self._substeps):
            before = self.state()
            if self.touchdown is None:
                # Box2D clears applied forces after every Step, so they are re-applied each
                # substep, where the nozzle is now and in the direction it now points.
                for force, position in forces:
                    body.ApplyForce(body.GetWorldVector(force), body.GetWorldPoint(position), True)
            self._world.Step(h, _VELOCITY_ITERATIONS, _POSITION_ITERATIONS)
            if self.touchdown is None and self._contacts.parts:
                self.touchdown = Touchdown(before, frozenset(self._contacts.parts))
        return self.state()

    def state(self) -> RocketState:
        """The rocket's current state."""
        body = self._body
        return RocketState(
            x=body.worldCenter.x,
            y=body.worldCenter.y,
            vx=body.linearVelocity.x,
            vy=body.linearVelocity.y,
            theta=math.remainder(body.angle, math.tau),  # Box2D angles are not wrapped
            omega=body.angularVelocity,
        )

    def shapes(self) -> list[tuple[str, list[tuple[float, float]]]]:
        """Each Box2D shape on the rocket: its name and corner points in world coordinates [m]."""
        body = self._body
        return [
            (fixture.userData, [tuple(body.GetWorldPoint(v)) for v in fixture.shape.vertices])
            for fixture in body.fixtures
        ]

    def to_world(self, point: tuple[float, float]) -> tuple[float, float]:
        """A point in rocket coordinates (e.g. a nozzle position), in world coordinates [m]."""
        return tuple(self._body.GetWorldPoint(point))

    def direction_to_world(self, direction: tuple[float, float]) -> tuple[float, float]:
        """A direction in rocket coordinates (e.g. a nozzle's), in world coordinates."""
        return tuple(self._body.GetWorldVector(direction))

    def _build_world(self) -> None:
        spec = self.spec
        self._world = Box2D.b2World(gravity=(0.0, -self._gravity), doSleep=False)
        self._contacts = _GroundContacts()
        self._world.contactListener = self._contacts
        self.touchdown: Touchdown | None = None

        ground = self._world.CreateStaticBody(position=(0.0, 0.0))
        ground.CreatePolygonFixture(
            box=(_GROUND_HALF_WIDTH, 1.0, (0.0, -1.0), 0.0), userData="ground"
        )

        # The body's origin is the center of the rocket's base, the same origin the spec
        # uses, so positions from the spec can be passed to Box2D as they are.
        self._body = self._world.CreateDynamicBody(position=(0.0, 0.0))
        _add_block(self._body, "body", spec.body.width, spec.body.height, spec.body.mass)
        _add_block(
            self._body,
            "engine block",
            spec.body.width,
            spec.engine_block.height,
            spec.engine_block.mass,
        )
        for leg in spec.legs:
            _add_leg(self._body, leg)


class _GroundContacts(Box2D.b2ContactListener):
    """Collects the names of rocket parts that touch the ground; Box2D calls BeginContact."""

    def __init__(self) -> None:
        super().__init__()
        self.parts: set[str] = set()

    def BeginContact(self, contact: Box2D.b2Contact) -> None:  # noqa: N802 (Box2D's name)
        names = {contact.fixtureA.userData, contact.fixtureB.userData}
        if "ground" in names:
            self.parts |= names - {"ground"}


def _add_block(body: Box2D.b2Body, name: str, width: float, height: float, mass: float) -> None:
    """Give Box2D a rectangle standing on the base, centered on the axis, with this mass."""
    body.CreatePolygonFixture(
        box=(width / 2, height / 2, (0.0, height / 2), 0.0),
        density=mass / (width * height),  # in 2D, Box2D's density is mass per area
        userData=name,
    )


def _add_leg(body: Box2D.b2Body, leg: Leg) -> None:
    """Give Box2D a thin, massless strut from the leg's top to its tip."""
    (top_x, top_y), (tip_x, tip_y) = leg.top, leg.tip
    half = _LEG_THICKNESS / 2
    body.CreatePolygonFixture(
        vertices=[
            (top_x - half, top_y),
            (top_x + half, top_y),
            (tip_x + half, tip_y),
            (tip_x - half, tip_y),
        ],
        density=0.0,  # the legs' mass is part of the engine block
        userData="leg",
    )
