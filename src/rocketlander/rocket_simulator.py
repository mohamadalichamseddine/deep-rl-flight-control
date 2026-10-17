"""Rocket motion, computed by the Box2D physics engine.

The rocket is given to Box2D as shapes with masses. Box2D then calculates the center of mass
and moment of inertia, turns each nozzle's force (applied where the nozzle is mounted) into
pushing and turning, adds gravity, and moves the rocket.
"""

import math
from collections.abc import Sequence

import Box2D

from rocketlander.rocket_spec import RocketSpec
from rocketlander.rocket_state import RocketState

STANDARD_GRAVITY = 9.81  # [m/s²]

# Constraint-solver iterations; they only matter once there are contacts.
_VELOCITY_ITERATIONS = 8
_POSITION_ITERATIONS = 3


class RocketSimulator:
    """Moves the rocket under gravity and nozzle thrust, using Box2D.

    Each ``step`` holds the throttle commands for one control step of ``dt`` seconds, split
    into ``substeps`` smaller Box2D steps for accuracy. Box2D stores numbers as 32-bit floats,
    so states are accurate to about 7 digits.
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
        self._substeps = substeps
        self._world = Box2D.b2World(gravity=(0.0, -gravity), doSleep=False)

        # The Box2D body's origin is the center of the rocket's base, the same origin the
        # spec uses, so nozzle positions can be passed to Box2D as they are.
        self._body = self._world.CreateDynamicBody(position=(0.0, 0.0))
        _add_block(self._body, spec.body.width, spec.body.height, spec.body.mass)
        _add_block(self._body, spec.body.width, spec.engine_block.height, spec.engine_block.mass)

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
        """Place the rocket in the given state."""
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
            # Box2D clears applied forces after every Step, so they are re-applied each
            # substep, where the nozzle is now and in the direction it now points.
            for force, position in forces:
                body.ApplyForce(body.GetWorldVector(force), body.GetWorldPoint(position), True)
            self._world.Step(h, _VELOCITY_ITERATIONS, _POSITION_ITERATIONS)
        return self.state()

    def shapes(self) -> list[list[tuple[float, float]]]:
        """Corner points of each Box2D shape on the rocket, in world coordinates [m]."""
        body = self._body
        return [
            [tuple(body.GetWorldPoint(vertex)) for vertex in fixture.shape.vertices]
            for fixture in body.fixtures
        ]

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


def _add_block(body: Box2D.b2Body, width: float, height: float, mass: float) -> None:
    """Give Box2D a rectangle standing on the base, centered on the axis, with this mass."""
    body.CreatePolygonFixture(
        box=(width / 2, height / 2, (0.0, height / 2), 0.0),
        density=mass / (width * height),  # in 2D, Box2D's density is mass per area
    )
