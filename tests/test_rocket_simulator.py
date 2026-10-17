"""Checks that Box2D, given our rocket, matches hand calculations and known physics."""

import math

import pytest

from rocketlander.rocket_simulator import STANDARD_GRAVITY, RocketSimulator
from rocketlander.rocket_spec import RocketSpec
from rocketlander.rocket_state import RocketState

G = STANDARD_GRAVITY

# Hand calculations for configs/rocket/r100.yaml:
MASS = 80.0 + 20.0
COM_HEIGHT = (80.0 * 1.5 + 20.0 * 0.2) / MASS  # 1.24 m
INERTIA = (  # each rectangle m(w² + h²)/12, moved to the center of mass: + m·d²
    80.0 * (0.3**2 + 3.0**2) / 12
    + 80.0 * (1.5 - COM_HEIGHT) ** 2
    + 20.0 * (0.3**2 + 0.4**2) / 12
    + 20.0 * (0.2 - COM_HEIGHT) ** 2
)  # ≈ 88.06 kg·m²
SIDE_TORQUE = 25.0 * (2.7 - COM_HEIGHT)  # side nozzle thrust × lever arm = 36.5 N·m


def _simulator(rocket: RocketSpec, state: RocketState, substeps: int = 5) -> RocketSimulator:
    sim = RocketSimulator(rocket, substeps=substeps)
    sim.reset(state)
    return sim


def _fly(sim: RocketSimulator, throttles: list[float], seconds: float) -> RocketState:
    for _ in range(round(seconds / sim.dt)):
        sim.step(throttles)
    return sim.state()


HOVER = MASS * G / 2000.0  # main throttle whose thrust equals the weight


def test_box2d_calculates_mass_properties_from_the_shapes(rocket: RocketSpec) -> None:
    sim = RocketSimulator(rocket)
    assert sim.mass == pytest.approx(MASS, rel=1e-6)
    assert sim.com_height == pytest.approx(COM_HEIGHT, rel=1e-6)
    assert sim.moment_of_inertia == pytest.approx(INERTIA, rel=1e-6)


def test_shapes_have_the_rocket_dimensions(rocket: RocketSpec) -> None:
    # Upright with the base 10 m up: the body spans x ±0.15 m and 3 m up from the base, the
    # engine block fills the bottom 0.4 m, and the legs reach from 0.5 m up to 0.2 m below.
    base = 10.0
    sim = _simulator(rocket, RocketState(0.0, base + COM_HEIGHT, 0.0, 0.0))

    shapes = sim.shapes()

    assert [name for name, _ in shapes] == ["body", "engine block", "leg", "leg"]
    expected = [((-0.15, 0.15), (0.0, 3.0)), ((-0.15, 0.15), (0.0, 0.4))]
    for (_, corners), (x_span, y_span) in zip(shapes, expected, strict=False):
        xs, ys = [x for x, _ in corners], [y - base for _, y in corners]
        assert (min(xs), max(xs)) == pytest.approx(x_span, abs=1e-5)
        assert (min(ys), max(ys)) == pytest.approx(y_span, abs=1e-5)
    for _, corners in shapes[2:]:
        assert min(y - base for _, y in corners) == pytest.approx(-0.2, abs=1e-5)
        assert max(y - base for _, y in corners) == pytest.approx(0.5, abs=1e-5)


def test_touchdown_on_the_legs_records_the_state_just_before_contact(rocket: RocketSpec) -> None:
    # Standing height of the center of mass: 1.24 m above the base + 0.2 m legs below it.
    sim = _simulator(rocket, RocketState(0.0, 1.44 + 0.05, 0.0, -1.0))

    for _ in range(10):
        sim.step([0, 0, 0])

    assert sim.touchdown is not None
    assert sim.touchdown.parts == {"leg"}
    # It fell 3-5 cm (Box2D starts a contact up to 2 cm early, its collision margin), so just
    # before contact it was falling at sqrt(1² + 2·g·fall) = 1.2-1.4 m/s. Box2D's contact
    # forces then stopped it.
    assert -1.41 <= sim.touchdown.state.vy <= -1.18
    assert abs(sim.state().vy) < 0.5


def test_engines_are_cut_after_touchdown(rocket: RocketSpec) -> None:
    sim = _simulator(rocket, RocketState(0.0, 1.44 + 0.05, 0.0, -1.0))
    _fly(sim, [0, 0, 0], 0.2)
    assert sim.touchdown is not None

    state = _fly(sim, [1, 1, 0], 1.0)  # full main thrust would lift off at 10 m/s²

    assert state.y == pytest.approx(1.44, abs=0.05)
    assert state.theta == pytest.approx(0.0, abs=1e-3)


def test_touchdown_reports_the_body_hitting_the_ground(rocket: RocketSpec) -> None:
    # Upside down, the nose (1.76 m from the center of mass) hits the ground first.
    sim = _simulator(rocket, RocketState(0.0, 1.76 + 0.1, 0.0, -3.0, theta=math.pi))

    _fly(sim, [0, 0, 0], 0.2)

    assert sim.touchdown is not None
    assert sim.touchdown.parts == {"body"}


def test_reset_starts_a_fresh_flight(rocket: RocketSpec) -> None:
    start = RocketState(0.3, 1.44 + 0.5, 0.2, -2.0, theta=0.05)
    sim = _simulator(rocket, start)
    first = [sim.step([0.4, 0.1, 0.0]) for _ in range(60)]
    assert sim.touchdown is not None

    sim.reset(start)
    assert sim.touchdown is None
    second = [sim.step([0.4, 0.1, 0.0]) for _ in range(60)]

    assert second == first


def test_reset_then_state_returns_the_same_state(rocket: RocketSpec) -> None:
    state = RocketState(x=-3.0, y=42.0, vx=1.5, vy=-7.0, theta=0.1, omega=-0.2)
    result = _simulator(rocket, state).state()
    for field in ("x", "y", "vx", "vy", "theta", "omega"):
        assert getattr(result, field) == pytest.approx(getattr(state, field), rel=1e-6)


def test_free_fall_matches_the_analytic_solution(rocket: RocketSpec) -> None:
    substeps, t = 5, 2.0
    sim = _simulator(rocket, RocketState(0.0, 100.0, 0.0, 0.0), substeps=substeps)

    state = _fly(sim, [0, 0, 0], t)

    assert state.vy == pytest.approx(-G * t, rel=1e-5)
    # Stepping in time adds a small position error, at most g·h·t/2 for a substep of h.
    h = sim.dt / substeps
    assert abs(state.y - (100.0 - G * t**2 / 2)) <= G * h * t / 2 + 1e-3
    assert state.x == pytest.approx(0.0, abs=1e-6)
    assert (state.vx, state.theta) == (0.0, 0.0)


def test_more_substeps_reduce_the_integration_error(rocket: RocketSpec) -> None:
    def fall_error(substeps: int) -> float:
        sim = _simulator(rocket, RocketState(0.0, 100.0, 0.0, 0.0), substeps=substeps)
        return abs(_fly(sim, [0, 0, 0], 2.0).y - (100.0 - G * 2.0**2 / 2))

    assert fall_error(10) < fall_error(1) / 5


def test_hover_throttle_holds_altitude(rocket: RocketSpec) -> None:
    sim = _simulator(rocket, RocketState(0.0, 50.0, 0.0, 0.0))

    state = _fly(sim, [HOVER, 0, 0], 5.0)

    assert state.y == pytest.approx(50.0, abs=1e-3)
    assert state.vy == pytest.approx(0.0, abs=1e-3)


def test_side_nozzle_turns_the_rocket_at_torque_over_inertia(rocket: RocketSpec) -> None:
    # The left nozzle turns the nose toward +x: clockwise, which is negative.
    sim = _simulator(rocket, RocketState(0.0, 50.0, 0.0, 0.0))

    state = _fly(sim, [0, 1, 0], 1.0)

    assert state.omega == pytest.approx(-SIDE_TORQUE / INERTIA * 1.0, rel=1e-3)


def test_main_nozzle_alone_does_not_turn_the_rocket(rocket: RocketSpec) -> None:
    sim = _simulator(rocket, RocketState(0.0, 50.0, 0.0, 0.0, theta=0.2))

    state = _fly(sim, [1, 0, 0], 2.0)

    # Box2D computes the torque in 32-bit floats; what remains is rounding noise (~0.0002 °/s).
    assert state.theta == pytest.approx(0.2, abs=1e-5)
    assert state.omega == pytest.approx(0.0, abs=1e-4)


def test_main_nozzle_pushes_along_the_tilted_axis(rocket: RocketSpec) -> None:
    # theta > 0 tilts the nose left, so full thrust accelerates the rocket toward -x.
    theta = 0.2
    sim = _simulator(rocket, RocketState(0.0, 50.0, 0.0, 0.0, theta=theta))

    state = _fly(sim, [1, 0, 0], 1.0)

    accel = 2000.0 / MASS
    assert state.vx == pytest.approx(-math.sin(theta) * accel, rel=1e-4)
    assert state.vy == pytest.approx(math.cos(theta) * accel - G, rel=1e-4)


@pytest.mark.parametrize(("nozzle", "direction"), [("left", +1), ("right", -1)])
def test_side_nozzle_moves_and_tilts_the_rocket_the_same_way(
    rocket: RocketSpec, nozzle: str, direction: int
) -> None:
    # Mounted above the center of mass, a side nozzle pushes the rocket toward a side and
    # tilts its nose toward the same side (theta < 0 is nose toward +x).
    throttles = [HOVER, float(nozzle == "left"), float(nozzle == "right")]
    sim = _simulator(rocket, RocketState(0.0, 50.0, 0.0, 0.0))

    state = _fly(sim, throttles, 0.5)

    assert math.copysign(1, state.vx) == direction
    assert math.copysign(1, -state.theta) == direction


def test_opposing_side_nozzles_cancel_out(rocket: RocketSpec) -> None:
    sim = _simulator(rocket, RocketState(0.0, 50.0, 0.0, 0.0))

    state = _fly(sim, [HOVER, 1, 1], 1.0)

    assert state.vx == pytest.approx(0.0, abs=1e-6)
    assert state.omega == pytest.approx(0.0, abs=1e-6)


def test_throttles_outside_0_to_1_saturate(rocket: RocketSpec) -> None:
    start = RocketState(0.0, 50.0, 0.0, 0.0)
    clipped = _fly(_simulator(rocket, start), [1.5, -0.3, 0], 1.0)
    in_range = _fly(_simulator(rocket, start), [1.0, 0.0, 0], 1.0)
    assert clipped == in_range


def test_one_throttle_per_nozzle_is_required(rocket: RocketSpec) -> None:
    sim = RocketSimulator(rocket)
    with pytest.raises(ValueError, match="expected 3 throttles, got 2"):
        sim.step([1.0, 0.0])


def test_theta_is_wrapped_to_plus_minus_pi(rocket: RocketSpec) -> None:
    sim = _simulator(rocket, RocketState(0.0, 50.0, 0.0, 0.0, theta=3.0, omega=2.0))

    state = _fly(sim, [0, 0, 0], 1.0)  # turns to 5 rad, i.e. 5 - 2π

    # Tolerance allows for Box2D summing the angle in 32-bit floats over 250 substeps.
    assert state.theta == pytest.approx(5.0 - math.tau, abs=1e-4)
