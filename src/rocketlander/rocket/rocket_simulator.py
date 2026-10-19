"""Rocket motion, computed by the Box2D physics engine.

The rocket is given to Box2D as shapes with masses. Box2D then calculates the center of mass
and moment of inertia, turns each nozzle's force (applied where the nozzle is mounted) into
pushing and turning, adds gravity, moves the rocket, and detects when it touches the ground.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import Box2D

from rocketlander.rocket.rocket_spec import Leg, RocketSpec
from rocketlander.rocket.rocket_state import RocketState

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
        rocket_spec: RocketSpec,
        gravity: float,
        dt: float,
        substeps: int,
    ) -> None:
        if substeps < 1:
            raise ValueError(f"substeps must be >= 1, got {substeps}")
        self.rocket_spec = rocket_spec
        self.dt = dt
        self.gravity = gravity
        self._substeps = substeps
        self._initialize_world()

    @property
    def mass(self) -> float:
        """Total mass [kg], as calculated by Box2D."""
        return self._box2d_rocket.mass

    @property
    def center_of_mass_height(self) -> float:
        """Height of the center of mass above the base [m], as calculated by Box2D."""
        return self._box2d_rocket.localCenter.y

    @property
    def y_when_standing(self) -> float:
        """Height of the center of mass when the rocket stands on its legs [m]."""
        lowest_tip = min(leg.tip[1] for leg in self.rocket_spec.legs)
        return self.center_of_mass_height - lowest_tip

    @property
    def moment_of_inertia(self) -> float:
        """Moment of inertia about the center of mass [kg·m²], as calculated by Box2D."""
        # Box2D reports it about the rocket's origin (the base); move it to the center of mass.
        box2d_rocket = self._box2d_rocket
        return box2d_rocket.inertia - box2d_rocket.mass * box2d_rocket.localCenter.lengthSquared

    def reset(self, state: RocketState) -> None:
        """Place the rocket in the given state, in a fresh world.

        Rebuilding the world clears contacts left over from the previous flight, so the same
        starting state always produces the same flight.
        """
        self._initialize_world()
        box2d_rocket = self._box2d_rocket
        box2d_rocket.angle = state.tilt
        # The state gives the center of mass, but Box2D places the rocket by its origin.
        center_of_mass_offset = box2d_rocket.GetWorldVector(box2d_rocket.localCenter)
        box2d_rocket.position = Box2D.b2Vec2(state.x, state.y) - center_of_mass_offset
        # Box2D's velocity is the center of mass's.
        box2d_rocket.linearVelocity = (state.vx, state.vy)
        box2d_rocket.angularVelocity = state.spin

    def step(self, throttles: Sequence[float]) -> RocketState:
        """Fire the nozzles at ``throttles`` (one per nozzle, in [0, 1]) for ``dt`` seconds.

        Throttles outside [0, 1] are clipped, as a real actuator saturates.
        """
        nozzles = self.rocket_spec.nozzles
        if len(throttles) != len(nozzles):
            raise ValueError(f"expected {len(nozzles)} throttles, got {len(throttles)}")
        nozzle_forces = []
        for nozzle, throttle in zip(nozzles, throttles, strict=True):
            thrust = min(max(throttle, 0.0), 1.0) * nozzle.max_thrust
            force = (nozzle.direction[0] * thrust, nozzle.direction[1] * thrust)
            nozzle_forces.append((force, nozzle.position))

        box2d_rocket = self._box2d_rocket
        substep_dt = self.dt / self._substeps
        for _ in range(self._substeps):
            state_before = self.state()
            if self.touchdown is None:
                # Box2D clears applied forces after every Step, so they are re-applied each
                # substep, where the nozzle is now and in the direction it now points.
                for force, nozzle_position in nozzle_forces:
                    box2d_rocket.ApplyForce(
                        force=box2d_rocket.GetWorldVector(force),
                        point=box2d_rocket.GetWorldPoint(nozzle_position),
                        wake=True,
                    )
            self._world.Step(substep_dt, _VELOCITY_ITERATIONS, _POSITION_ITERATIONS)
            if self.touchdown is None and self._ground_contacts.parts:
                self.touchdown = Touchdown(state_before, frozenset(self._ground_contacts.parts))
        return self.state()

    def state(self) -> RocketState:
        """The rocket's current state."""
        box2d_rocket = self._box2d_rocket
        return RocketState(
            x=box2d_rocket.worldCenter.x,
            y=box2d_rocket.worldCenter.y,
            vx=box2d_rocket.linearVelocity.x,
            vy=box2d_rocket.linearVelocity.y,
            tilt=math.remainder(box2d_rocket.angle, math.tau),  # Box2D angles are not wrapped
            spin=box2d_rocket.angularVelocity,
        )

    def shapes(self) -> list[tuple[str, list[tuple[float, float]]]]:
        """Each Box2D shape on the rocket: its name and corner points in world coordinates [m]."""
        box2d_rocket = self._box2d_rocket
        return [
            (
                fixture.userData,
                [tuple(box2d_rocket.GetWorldPoint(vertex)) for vertex in fixture.shape.vertices],
            )
            for fixture in box2d_rocket.fixtures
        ]

    def point_to_world(self, point: tuple[float, float]) -> tuple[float, float]:
        """A point in rocket coordinates (e.g. a nozzle position), in world coordinates [m]."""
        return tuple(self._box2d_rocket.GetWorldPoint(point))

    def direction_to_world(self, direction: tuple[float, float]) -> tuple[float, float]:
        """A direction in rocket coordinates (e.g. a nozzle's), in world coordinates."""
        return tuple(self._box2d_rocket.GetWorldVector(direction))

    def _initialize_world(self) -> None:
        rocket_spec = self.rocket_spec
        self._world = Box2D.b2World(gravity=(0.0, -self.gravity), doSleep=False)
        self._ground_contacts = _GroundContacts()
        self._world.contactListener = self._ground_contacts
        self.touchdown: Touchdown | None = None

        ground = self._world.CreateStaticBody(position=(0.0, 0.0))
        ground.CreatePolygonFixture(
            box=(_GROUND_HALF_WIDTH, 1.0, (0.0, -1.0), 0.0), userData="ground"
        )

        # The Box2D rocket's origin is the center of the rocket's base, the same origin the
        # RocketSpec uses, so its positions can be passed to Box2D as they are.
        self._box2d_rocket = self._world.CreateDynamicBody(position=(0.0, 0.0))
        _add_block(
            self._box2d_rocket,
            "body",
            rocket_spec.body.width,
            rocket_spec.body.height,
            rocket_spec.body.mass,
        )
        _add_block(
            self._box2d_rocket,
            "engine block",
            rocket_spec.body.width,
            rocket_spec.engine_block.height,
            rocket_spec.engine_block.mass,
        )
        for leg in rocket_spec.legs:
            _add_leg(self._box2d_rocket, leg)


class _GroundContacts(Box2D.b2ContactListener):
    """Collects the names of rocket parts that touch the ground; Box2D calls BeginContact."""

    def __init__(self) -> None:
        super().__init__()
        self.parts: set[str] = set()

    def BeginContact(self, contact: Box2D.b2Contact) -> None:  # noqa: N802 (Box2D's name)
        part_names = {contact.fixtureA.userData, contact.fixtureB.userData}
        if "ground" in part_names:
            self.parts |= part_names - {"ground"}


def _add_block(
    box2d_rocket: Box2D.b2Body, name: str, width: float, height: float, mass: float
) -> None:
    """Give Box2D a rectangle standing on the base, centered on the axis, with this mass."""
    box2d_rocket.CreatePolygonFixture(
        box=(width / 2, height / 2, (0.0, height / 2), 0.0),
        density=mass / (width * height),  # in 2D, Box2D's density is mass per area
        userData=name,
    )


def _add_leg(box2d_rocket: Box2D.b2Body, leg: Leg) -> None:
    """Give Box2D a thin, massless strut from the leg's top to its tip."""
    (top_x, top_y), (tip_x, tip_y) = leg.top, leg.tip
    half_thickness = _LEG_THICKNESS / 2
    box2d_rocket.CreatePolygonFixture(
        vertices=[
            (top_x - half_thickness, top_y),
            (top_x + half_thickness, top_y),
            (tip_x + half_thickness, tip_y),
            (tip_x - half_thickness, tip_y),
        ],
        density=0.0,  # the legs' mass is part of the engine block
        userData="leg",
    )
